import CoreML
import Vision
import UIKit
import Accelerate


// Per-frame values the post-processing needs. Captured when the frame is submitted and carried
// with that frame's results, so a later frame can never overwrite them mid-flight.
struct FrameInfo {
    let timestamp: UInt64
    let originalWidth: Int
    let originalHeight: Int
    let scaleX: Float
    let scaleY: Float
}

// YOLOPredictor Class
//
// Threading: the Vision request object is reused, so model runs (handler.perform) must never
// overlap. In synchronous mode the caller runs them (Unity calls from its main thread); in
// asynchronous mode they run back to back on inferenceQueue. Post-processing always runs on the
// serial postprocessQueue, so result callbacks never overlap and arrive in submission order —
// the Unity side reuses its detection objects across callbacks and is not reentrant.
class YOLOPredictor {
    let model: MLModel
    let detector: VNCoreMLModel
    let confidenceThreshold: Float
    let iouThreshold: Float
    let modelWidth: Int
    let modelHeight: Int
    let classNames: [Int: String]

    // When true, predict() returns right after submitting the frame and the model runs on
    // inferenceQueue. A frame submitted while the model is running waits in a one-frame slot,
    // replacing any older waiting frame (latest wins, the others get no callback), and runs as
    // soon as the current run finishes — so the model never idles waiting for the next camera
    // frame, and what it runs is never more than one frame old. When false, predict() blocks
    // the caller for the model run (the original behavior).
    var runsAsynchronously = false
    private let inferenceQueue = DispatchQueue(label: "YOLOUnity.inference", qos: .userInitiated)
    private let postprocessQueue = DispatchQueue(label: "YOLOUnity.postprocess", qos: .userInitiated)
    private typealias PendingFrame = (handler: VNImageRequestHandler, info: FrameInfo)
    private let frameLock = NSLock()  // guards modelRunning and waitingFrame
    private var modelRunning = false
    private var waitingFrame: PendingFrame? = nil
    
    lazy var visionRequest: YOLORequest = {
        let request = YOLORequest(
          model: detector,
          completionHandler: {
            [weak self] request, error in
            self?.processObservations(for: request, error: error)
          })
        // NOTE: BoundingBoxView object scaling depends on request.imageCropAndScaleOption https://developer.apple.com/documentation/vision/vnimagecropandscaleoption
        request.imageCropAndScaleOption = .scaleFill  // .scaleFit, .scaleFill, .centerCrop
        return request
    }()

    init?(
        modelName: String,
        confidanceThreshold: Float = 0.5,
        iouThreshold: Float = 0.5,
        scaleMethod: String = "scaleFill"
    ) {
        // Initialize the Core ML model
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine // save GPU for rendering
//        config.computeUnits = .all
        
        guard let model: MLModel = {
            switch modelName {
//            case "yolo11x_seg":
//                return try? yolo11x_seg(configuration: config).model
//            case "yolo11l_seg":
//                return try? yolo11l_seg(configuration: config).model
//            case "yolo11m_seg":
//                return try? yolo11m_seg(configuration: config).model
//            case "yolo11s_seg":
//                return try? yolo11s_seg(configuration: config).model
//            case "yolo11n_seg":
//                return try? yolo11n_seg(configuration: config).model
            case "vistas_l":
                NSLog("Info: 'vistas_l' is routed to the vistas_l_v2 model.")
                return try? vistas_l_v2(configuration: config).model
            case "vistas_m":
                NSLog("Info: 'vistas_m' is routed to the vistas_m_v2 model.")
                return try? vistas_m_v2(configuration: config).model
            case "vistas_s":
                NSLog("Info: 'vistas_s' is routed to the vistas_s_v2 model.")
                return try? vistas_s_v2(configuration: config).model
            default:
                NSLog("Error: Unknown model name '\(modelName)'.")
                return nil
            }
        }() else {
            return nil
        }
        
        guard let detector = try? VNCoreMLModel(for: model) else {
            NSLog("Error: Failed to initialize the detector.")
            return nil
        }
        
        self.model = model
        self.detector = detector
        self.detector.featureProvider = ThresholdProvider()
        self.confidenceThreshold = confidanceThreshold
        self.iouThreshold = iouThreshold
        
        (self.modelWidth, self.modelHeight, self.classNames) = parseModelSizeAndNames(model: model)
        
        let request = YOLORequest(
            model: detector,
            completionHandler: { [weak self] request, error in
                self?.processObservations(for: request, error: error)
        })
        
//        request.imageCropAndScaleOption = .scaleFill  // .scaleFit, .scaleFill, .centerCrop
        switch scaleMethod {
        case "scaleFit":
            request.imageCropAndScaleOption = VNImageCropAndScaleOption.scaleFit
        case "scaleFill":
            request.imageCropAndScaleOption = VNImageCropAndScaleOption.scaleFill
        case "centerCrop":
            request.imageCropAndScaleOption = VNImageCropAndScaleOption.centerCrop
        default:
            NSLog("Cannot parse scaleMethod: \(scaleMethod), defaulting to scaleFit")
            request.imageCropAndScaleOption = VNImageCropAndScaleOption.scaleFit
        }
        
        visionRequest = request
        
        NSLog("OpenCV version: \(OpenCVWrapper.opencvVersion())")
        
        NSLog("Initialized model \(modelName) with scaleMethod=\(scaleMethod), score threshold=\(confidenceThreshold), iou threshold=\(iouThreshold), Model width=\(modelWidth), height=\(modelHeight), numClasses=\(classNames.count)")
    }
    
    
    func predict(cgImage: CGImage, timestamp: UInt64, scaleX: Float = 1.0, scaleY: Float = 1.0) {
        let info = FrameInfo(timestamp: timestamp, originalWidth: cgImage.width,
                             originalHeight: cgImage.height, scaleX: scaleX, scaleY: scaleY)
        submit(VNImageRequestHandler(cgImage: cgImage, options: [:]), info)
    }
    
    // The pixel buffer must already be an independent copy of the caller's pixels (see
    // bytesToCVPixelBuffer): in asynchronous mode it is read after this returns.
    func predict(cvPixelBuffer: CVPixelBuffer, timestamp: UInt64, scaleX: Float = 1.0, scaleY: Float = 1.0) {
        let info = FrameInfo(timestamp: timestamp, originalWidth: CVPixelBufferGetWidth(cvPixelBuffer),
                             originalHeight: CVPixelBufferGetHeight(cvPixelBuffer), scaleX: scaleX, scaleY: scaleY)
        submit(VNImageRequestHandler(cvPixelBuffer: cvPixelBuffer, options: [:]), info)
    }

    private func submit(_ handler: VNImageRequestHandler, _ info: FrameInfo) {
        guard runsAsynchronously else {
            runModel(handler, info)
            return
        }
        frameLock.lock()
        if modelRunning {
            waitingFrame = (handler, info)  // latest wins
            frameLock.unlock()
            return
        }
        modelRunning = true
        frameLock.unlock()

        inferenceQueue.async { self.runFrames(startingWith: (handler, info)) }
    }

    // Runs frames back to back on inferenceQueue until no frame is waiting.
    private func runFrames(startingWith first: PendingFrame) {
        var next: PendingFrame? = first
        while let frame = next {
            runModel(frame.handler, frame.info)
            frameLock.lock()
            next = waitingFrame
            waitingFrame = nil
            if next == nil { modelRunning = false }
            frameLock.unlock()
        }
    }

    // Runs the model; Vision calls processObservations synchronously before perform returns.
    private func runModel(_ handler: VNImageRequestHandler, _ info: FrameInfo) {
        visionRequest.frameInfo = info
        do {
            try handler.perform([visionRequest])
        } catch {
            NSLog("Prediction failed: \(error.localizedDescription)")
        }
    }
    
    func processObservations(for request: VNRequest, error: Error?) {
        // Still inside handler.perform: snapshot the results and this frame's info now, since
        // the next run reuses the same request object.
        let observations = request.results
        let frameInfo = (request as? YOLORequest)?.frameInfo
        postprocessQueue.async {
            autoreleasepool {
                //            let startTime = CACurrentMediaTime()
                if let error = error {
                    NSLog("Error in processing observations: \(error.localizedDescription)")
                    return
                }
                
                // Access results
                guard let results = observations as? [VNCoreMLFeatureValueObservation],
                      results.count >= 2 else {
                    NSLog("Error: Insufficient results. Found \(observations?.count ?? 0) results.")
                    return
                }
                
                // Get both multi-arrays
                guard let array0 = results[0].featureValue.multiArrayValue,
                      let array1 = results[1].featureValue.multiArrayValue else {
                    NSLog("Error: Could not get multi-arrays from results")
                    return
                }
                
                // Check dimensions to determine which is which
                let array0Rank = array0.shape.count
                let array1Rank = array1.shape.count
                
                // YOLO boxes typically have rank 3, masks have rank 4
                // Assign boxes and masks based on their dimensions
                let (boxes, masks) = (array0Rank == 3 && array1Rank == 4) ? (array0, array1) :
                (array0Rank == 4 && array1Rank == 3) ? (array1, array0) :
                {
                    NSLog("Error: Unexpected array dimensions. Array 0: \(array0.shape), Array 1: \(array1.shape)")
                    return (array0, array1) // Default case, may not work correctly
                }()
                
                guard let info = frameInfo else {
                    NSLog("Missing image properties")
                    return
                }
                let originalWidth = info.originalWidth, originalHeight = info.originalHeight
                let timestamp = info.timestamp, scaleX = info.scaleX, scaleY = info.scaleY
                
                let coordinateRestorer = getCoordinateRestorer(
                    originalSize: (Float(originalWidth), Float(originalHeight)),
                    targetSize: (Float(self.modelWidth), Float(self.modelHeight)),
                    option: self.visionRequest.imageCropAndScaleOption
                )
                
                //            NSLog("Original size: \(originalWidth)x\(originalHeight)")
                
                let numMasks = masks.shape[1].intValue
                let numClasses = boxes.shape[1].intValue - 4 - numMasks
                let maskWidth = masks.shape[2].intValue
                let maskHeight = masks.shape[3].intValue
                let maskSize = maskWidth * maskHeight
                
                let boxPredictions: [BoxPrediction] = parseBoundingBoxes(
                    multiArray: boxes,
                    numClasses: numClasses,
                    confidenceThreshold: self.confidenceThreshold
                )
                
                // apply NMS
                
                let groupedPredictions = Dictionary(grouping: boxPredictions) { prediction in
                    prediction.classIndex
                }
                
                var nmsPredictions: [BoxPrediction] = []
                nmsPredictions.reserveCapacity(100)
                let _ = groupedPredictions.mapValues { predictions in
                    nmsPredictions.append(
                        contentsOf: nonMaximumSuppression(
                            predictions: predictions,
                            iouThreshold: self.iouThreshold,
                            limit: 20
                        )
                    )
                }
                
                let numPredictions = nmsPredictions.count
                var coefficients = [Float](repeating: 0.0, count: numPredictions * numMasks)
                for (i, prediction) in nmsPredictions.enumerated() {
                    prediction.maskCoefficients.withUnsafeBufferPointer { ptr in
                        coefficients.withUnsafeMutableBufferPointer { destPtr in
                            // Copy each prediction's coefficients to the correct offset
                            memcpy(
                                destPtr.baseAddress!.advanced(by: i * numMasks),
                                ptr.baseAddress!,
                                numMasks * MemoryLayout<Float>.stride
                            )
                        }
                    }
                }
                
                let flatMasks = masks.flatArray()
                
                var allMasks = [Float](repeating: 0.0, count: numPredictions * maskSize)
                vDSP_mmul(
                    coefficients, 1,
                    flatMasks, 1,
                    &allMasks, 1,
                    vDSP_Length(numPredictions),
                    vDSP_Length(maskSize),
                    vDSP_Length(numMasks)
                )
                
                let sigmoidAllMasks = getSigmoidMask(mask: allMasks.withUnsafeBufferPointer { $0.baseAddress! },
                                                     maskSize: numPredictions * maskSize)
                
                //            let maskProtos: [[Float]] = getMaskProtos(masks: masks, numMasks: numMasks)
                
                //            var i = 0
                //            print("Got \(nmsPredictions.count) predictions")
                
                let contourList: [OpenCVWrapper.ContoursResult] = nmsPredictions.concurrentEnumeratedMap { (i, box) in
                    //                let mask = getMasksFromProtos(
                    //                    maskProtos: maskProtos,
                    //                    coefficients: box.maskCoefficients
                    //                )
                    let sigmoidMask = sigmoidAllMasks.withUnsafeBufferPointer { ptr in
                        ptr.baseAddress!.advanced(by: i * maskSize)
                    }
                    
                    //                let sigmoidMask = getSigmoidMask(mask: mask, maskSize: maskSize)
                    
                    let upsampledMask = upsampleMask(
                        mask: sigmoidMask,
                        width: maskWidth,
                        height: maskHeight,
                        newWidth: self.modelWidth,
                        newHeight: self.modelHeight
                    )
                    
                    let (croppedMask, (boxWidth, boxHeight)) = cropMaskPhysical(
                        //                let croppedMask = cropMask(
                        mask: upsampledMask,
                        width: self.modelWidth,
                        height: self.modelHeight,
                        bbox: box.xyxy
                    )
                    
                    let zeroedMask = removeBelowThreshold(mask: croppedMask, threshold: 0.5)
                    
                    let contours: OpenCVWrapper.ContoursResult = OpenCVWrapper.findContours(mask: zeroedMask, width: boxWidth, height: boxHeight, corner: (box.xyxy.x1, box.xyxy.y1), coordinateRestorer: coordinateRestorer)
                    
                    return contours
                }
                
                //            let endTime = CACurrentMediaTime()
                //            print("Processing used \(endTime - startTime) seconds (\(1.0 / (endTime - startTime)) FPS")
                
                if let callback = yoloCallback {
                    let indexData = nmsPredictions.map { Int32($0.classIndex) }
                    let scores = nmsPredictions.map { $0.score }
                    
                    let boxes = nmsPredictions.flatMap { box in
                        let p1 = coordinateRestorer(box.xyxy.x1, box.xyxy.y1)
                        let p2 = coordinateRestorer(box.xyxy.x2, box.xyxy.y2)
                        return [Int32(Float(p1.0) * scaleX), Int32(Float(p1.1) * scaleY), Int32(Float(p2.0) * scaleX), Int32(Float(p2.1) * scaleY)]
                    }
                    
                    // flatten all contour points
                    let contourPoints = contourList.flatMap { contours in
                        contours.contours.flatMap { contour in
                            contour.flatMap { [Int32(Float($0.0) * scaleX), Int32(Float($0.1) * scaleY)] }
                        }
                    }
                    
                    var contourIndices: [Int32] = []
                    var currentIndex: Int32 = 0
                    for contours in contourList {
                        contourIndices.append(currentIndex)
                        for contour in contours.contours {
                            contourIndices.append(currentIndex + Int32(contour.count))
                            currentIndex += Int32(contour.count)
                        }
                        contourIndices.append(-1)
                    }
                    
                    let centroids = contourList.flatMap { contours in
                        [
                            Int32(contours.centroid.0 * scaleX),
                            Int32(contours.centroid.1 * scaleY)
                        ]
                    }
                    
                    indexData.withUnsafeBufferPointer { indexPtr in
                        scores.withUnsafeBufferPointer { scoresPtr in
                            boxes.withUnsafeBufferPointer { boxesPtr in
                                contourPoints.withUnsafeBufferPointer { pointsPtr in
                                    contourIndices.withUnsafeBufferPointer { indicesPtr in
                                        centroids.withUnsafeBufferPointer { centroidPtr in
                                            callback(
                                                Int32(nmsPredictions.count),
                                                indexPtr.baseAddress!,
                                                scoresPtr.baseAddress!,
                                                boxesPtr.baseAddress!,
                                                pointsPtr.baseAddress!,
                                                Int32(contourPoints.count),
                                                indicesPtr.baseAddress!,
                                                Int32(contourIndices.count),
                                                centroidPtr.baseAddress!,
                                                timestamp
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


class ThresholdProvider: MLFeatureProvider {
  /// Stores IoU and confidence thresholds as MLFeatureValue objects.
  var values: [String: MLFeatureValue]

  /// The set of feature names provided by this provider.
  var featureNames: Set<String> {
    return Set(values.keys)
  }

  /// Initializes the provider with specified IoU and confidence thresholds.
  /// - Parameters:
  ///   - iouThreshold: The IoU threshold for determining object overlap.
  ///   - confidenceThreshold: The minimum confidence for considering a detection valid.
  init(iouThreshold: Double = 0.45, confidenceThreshold: Double = 0.25) {
    values = [
      "iouThreshold": MLFeatureValue(double: iouThreshold),
      "confidenceThreshold": MLFeatureValue(double: confidenceThreshold),
    ]
  }

  /// Returns the feature value for the given feature name.
  /// - Parameter featureName: The name of the feature.
  /// - Returns: The MLFeatureValue object corresponding to the feature name.
  func featureValue(for featureName: String) -> MLFeatureValue? {
    return values[featureName]
  }
}


class YOLORequest: VNCoreMLRequest {
    // Set before each run; read synchronously in the completion handler.
    var frameInfo: FrameInfo?
}
