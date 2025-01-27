import CoreML
import Vision
import UIKit


// YOLOPredictor Class
class YOLOPredictor {
    let model: MLModel
    let detector: VNCoreMLModel
    let confidenceThreshold: Float
    let iouThreshold: Float
    let modelWidth: Int
    let modelHeight: Int
    let classNames: [Int: String]
    
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
        
        guard let model: MLModel = {
            switch modelName {
            case "yolo11x_seg":
                return try? yolo11x_seg(configuration: config).model
            case "yolo11l_seg":
                return try? yolo11l_seg(configuration: config).model
            case "yolo11m_seg":
                return try? yolo11m_seg(configuration: config).model
            case "yolo11s_seg":
                return try? yolo11s_seg(configuration: config).model
            case "yolo11n_seg":
                return try? yolo11n_seg(configuration: config).model
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
            request.imageCropAndScaleOption = .scaleFit
        case "scaleFill":
            request.imageCropAndScaleOption = .scaleFill
        case "centerCrop":
            request.imageCropAndScaleOption = .centerCrop
        default:
            NSLog("Cannot parse scaleMethod: \(scaleMethod), defaulting to scaleFit")
            request.imageCropAndScaleOption = .scaleFit
        }
        
        visionRequest = request
        
        NSLog("OpenCV version: \(OpenCVWrapper.opencvVersion())")
        
        NSLog("Initialized model \(modelName) with scaleMethod=\(scaleMethod), score threshold=\(confidenceThreshold), iou threshold=\(iouThreshold), Model width=\(modelWidth), height=\(modelHeight), numClasses=\(classNames.count)")
    }
    
    
    func predict(cgImage: CGImage, timestamp: UInt64) {
        visionRequest.userData["timestamp"] = timestamp
        visionRequest.userData["originalWidth"] = cgImage.width
        visionRequest.userData["originalHeight"] = cgImage.height
        
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
//            let startTime = CACurrentMediaTime()
            try handler.perform([visionRequest])
//            let endTime = CACurrentMediaTime()
//            print("CGImage prediction used \(endTime - startTime) seconds (\(1.0 / (endTime - startTime)) FPS")
        } catch {
            NSLog("Prediction failed: \(error.localizedDescription)")
        }
    }
    
    func predict(cvPixelBuffer: CVPixelBuffer, timestamp: UInt64) {
        visionRequest.userData["originalWidth"] = CVPixelBufferGetWidth(cvPixelBuffer)
        visionRequest.userData["originalHeight"] = CVPixelBufferGetHeight(cvPixelBuffer)
        visionRequest.userData["timestamp"] = timestamp
        
        let handler = VNImageRequestHandler(cvPixelBuffer: cvPixelBuffer, options: [:])
        do {
//            let startTime = CACurrentMediaTime()
            try handler.perform([visionRequest])
//            let endTime = CACurrentMediaTime()
//            print("CVPixelBuffer prediction used \(endTime - startTime) seconds (\(1.0 / (endTime - startTime)) FPS")
        } catch {
            NSLog("Prediction failed: \(error.localizedDescription)")
        }
    }
    
    func processObservations(for request: VNRequest, error: Error?) {
        DispatchQueue.main.async {
//            let startTime = CACurrentMediaTime()
            
            if let error = error {
                NSLog("Error in processing observations: \(error.localizedDescription)")
                return
            }
            
            // Access results
            guard let results = request.results as? [VNCoreMLFeatureValueObservation] else {
                NSLog("No results found or unexpected result type.")
                return
            }
            
            guard let yoloRequest = request as? YOLORequest,
                  let originalWidth = yoloRequest.userData["originalWidth"] as? Int,
                  let originalHeight = yoloRequest.userData["originalHeight"] as? Int,
                  let timestamp = yoloRequest.userData["timestamp"] as? UInt64 else {
                NSLog("Missing image properties")
                return
            }
            
            let coordinateRestorer = getCoordinateRestorer(
                originalSize: (Float(originalWidth), Float(originalHeight)),
                targetSize: (Float(self.modelWidth), Float(self.modelHeight)),
                option: self.visionRequest.imageCropAndScaleOption
            )
            
//            NSLog("Original size: \(originalWidth)x\(originalHeight)")
            
            let boxes: MLMultiArray = results[0].featureValue.multiArrayValue!
            let masks: MLMultiArray = results[1].featureValue.multiArrayValue!
            
            let numMasks = masks.shape[1].intValue
            let numClasses = boxes.shape[1].intValue - 4 - numMasks
            let maskWidth = masks.shape[2].intValue
            let maskHeight = masks.shape[3].intValue

            let boxPredictions: [BoxPrediction] = parseBoundingBoxes(
                multiArray: boxes,
                numClasses: numClasses,
                confidenceThreshold: self.confidenceThreshold
            )
            
//            print("Got \(boxPredictions.count) raw predictions")
            
            guard !boxPredictions.isEmpty else {
                return
            }
            
            // apply NMS
            
            let groupedPredictions = Dictionary(grouping: boxPredictions) { prediction in
                prediction.classIndex
            }
            
            var nmsPredictions: [BoxPrediction] = []
            let _ = groupedPredictions.mapValues { predictions in
                nmsPredictions.append(
                    contentsOf: nonMaximumSuppression(
                        predictions: predictions,
                        iouThreshold: self.iouThreshold,
                        limit: 100))
            }
            
            let maskProtos: [[Float]] = getMaskProtos(masks: masks, numMasks: numMasks)
            
//            var i = 0
//            print("Got \(nmsPredictions.count) predictions")
            
            for box in nmsPredictions {
                let mask = getMasksFromProtos(
                    maskProtos: maskProtos,
                    coefficients: box.maskCoefficients
                )
                
                let sigmoidMask = getSigmoidMask(mask: mask)
                
                let upsampledMask = upsampleMask(
                    mask: sigmoidMask,
                    width: maskWidth,
                    height: maskHeight,
                    newWidth: self.modelWidth,
                    newHeight: self.modelHeight
                )
                
                let (croppedMask, (boxWidth, boxHeight)) = cropMaskPhysical(
                    mask: upsampledMask,
                    width: self.modelWidth,
                    height: self.modelHeight,
                    bbox: box.xyxy
                )
                
                let zeroedMask = removeBelowThreshold(mask: croppedMask, threshold: 0.5)
                
//                let filename: String = "\(i)_\(self.classNames[box.classIndex, default: "Unknown"])_mask.png"
//                i += 1
//                
//                let exportPath = saveGrayscaleImage(mask: zeroedMask, width: boxWidth, height: boxHeight, filename: filename)
//
//                print("Exported to \(exportPath)")
                
                print("Recognized \(self.classNames[box.classIndex, default: "Unknown"]) at \(coordinateRestorer(box.xyxy.x1, box.xyxy.y1)), \(coordinateRestorer(box.xyxy.x2, box.xyxy.y2))")
                
            }
            
//            let endTime = CACurrentMediaTime()
//            print("Processing used \(endTime - startTime) seconds (\(1.0 / (endTime - startTime)) FPS")
            
            if let callback = yoloCallback {
                callback(nmsPredictions.count)
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
    var userData: [String: Any] = [:]
}
