import CoreML
import Vision
import UIKit
import Accelerate

typealias XYXY = (x1: Float, y1: Float, x2: Float, y2: Float)

struct BoxPrediction : Identifiable {
    let id = UUID()
    
    let classIndex: Int
    let score: Float
    let xyxy: XYXY
    let maskCoefficients: [Float]
}

struct Prediction : Identifiable {
    let id = UUID()
    
    let classIndex: Int
    let score: Float
    let box: XYXY
    let mask: [UInt8]
    let maskSize: (Int, Int)
}

func parseBoundingBoxes(
    multiArray: MLMultiArray,
    numClasses: Int,
    confidenceThreshold: Float) -> [BoxPrediction] {
    let shape = multiArray.shape.map { $0.intValue }
    let boxCount = shape[2] // number of boxes
    let featureCount = shape[1] // 116 features per box
    
    var boxPredictions: [BoxPrediction] = []

    for i in 0..<boxCount {
        // Extract class confidences
        let classConfidences = (4..<(4+numClasses)).map { multiArray[$0*boxCount + i].floatValue }
        let (bestClassIndex, bestClassConfidence) = classConfidences.enumerated().max(by: { $0.element < $1.element })!

        // Check if the best confidence exceeds the threshold
        guard bestClassConfidence > confidenceThreshold else { continue }
        
        let cx = Float(truncating: multiArray[0*boxCount + i])
        let cy = Float(truncating: multiArray[1*boxCount + i])
        let width   = Float(truncating: multiArray[2*boxCount + i])
        let height  = Float(truncating: multiArray[3*boxCount + i])
        
        let xMin = cx - width / 2
        let yMin = cy - height / 2
        let xMax = cx + width / 2
        let yMax = cy + height / 2
        
        // Extract mask weights (32 weights)
        let maskWeights = ((4+numClasses)..<featureCount).map {
            multiArray[$0*boxCount + i].floatValue
        }
        
        let prediction = BoxPrediction(
            classIndex: bestClassIndex,
            score: bestClassConfidence,
            xyxy: XYXY(x1: xMin, y1: yMin, x2: xMax, y2: yMax),
            maskCoefficients: maskWeights
        )
        boxPredictions.append(prediction)
    }
    
    return boxPredictions
}


func nonMaximumSuppression(
    predictions: [BoxPrediction],
    iouThreshold: Float,
    limit: Int
) -> [BoxPrediction] {
    guard !predictions.isEmpty else { return [] }
    
    let sortedIndices = predictions.indices.sorted {
        predictions[$0].score > predictions[$1].score
    }

    var selected: [BoxPrediction] = []
    var active = [Bool](repeating: true, count: predictions.count)
    var numActive = active.count

    // Start with the box that has the highest score.
    // Remove any remaining boxes that overlap it more than the given threshold
    // amount. If there are any boxes left (i.e. these did not overlap with any
    // previous boxes), then repeat this procedure, until no more boxes remain
    // or the limit has been reached.
    outer: for i in 0..<predictions.count {
        
        if active[i] {
            
            let boxA = predictions[sortedIndices[i]]
            selected.append(boxA)
            
            if selected.count >= limit { break }

            for j in i+1..<predictions.count {
            
                if active[j] {
            
                    let boxB = predictions[sortedIndices[j]]
                    
                    if IOU(box1: boxA.xyxy, box2: boxB.xyxy) > iouThreshold {
                        active[j] = false
                        numActive -= 1
                       
                        if numActive <= 0 { break outer }
                    
                    }
                
                }
            
            }
        }
        
    }
    return selected
}


func getMaskProtos(masks: MLMultiArray, numMasks: Int) -> [[Float]] {
    let width = masks.shape[2].intValue
    let height = masks.shape[3].intValue
    let maskSize = height * width
    
    let pointer = masks.dataPointer.assumingMemoryBound(to: Float.self)
    let maskStride = masks.strides[1].intValue
    
    return (0..<numMasks).map { maskIdx -> [Float] in
        let start = pointer.advanced(by: maskIdx * maskStride)
        return Array(UnsafeBufferPointer(start: start, count: maskSize))
    }
}


func getMaskProtos_Naive(masks: MLMultiArray, numMasks: Int) -> [[Float]] {
    var finalMasks: [[Float]] = []
    let rows = masks.shape[3].intValue
    let columns = masks.shape[2].intValue
    for tube in 0..<numMasks {
        var mask: [Float] = []
        for i in 0..<(rows*columns) {
            let index = tube*(rows*columns)+i
            mask.append(Float(truncating: masks[index]))
        }
        finalMasks.append(mask)
    }
    return finalMasks
}


func getMasksFromProtos(
    maskProtos: [[Float]],
    coefficients: [Float]
) -> [Float] {
    guard maskProtos.count == coefficients.count else {
        print("Unmatched length: \(maskProtos.count) vs \(coefficients.count)")
        return []
    }
    
    guard !maskProtos.isEmpty else {
        print("No masks provided")
        return []
    }
    
    let maskSize = maskProtos[0].count
    
    var summedMask = [Float](repeating: 0, count: maskSize)
    for (maskProto, var coefficient) in zip(maskProtos, coefficients) {
        var multipliedMask = [Float](repeating: 0, count: maskSize)
        vDSP_vsmul(maskProto, 1, &coefficient, &multipliedMask, 1, vDSP_Length(maskSize))
        vDSP_vadd(summedMask, 1, multipliedMask, 1, &summedMask, 1, vDSP_Length(maskSize))
    }
    
    return summedMask
}


func getSigmoidMask(mask: [Float]) -> [Float] {
    let maskSize = mask.count
    var count = Int32(maskSize)
    var onef: Float = 1.0
    
    // step 1: x -> e^-x
    let negatedMask = mask.map { -$0 }
    var expNegatedMask = [Float](repeating: 0, count: maskSize)
    vvexpf(&expNegatedMask, negatedMask, &count)
    
    // step 2: e^-x -> 1+e^-x
    vDSP_vsadd(expNegatedMask, 1, &onef, &expNegatedMask, 1, vDSP_Length(maskSize))
    
    // step 3: 1+e^-x -> 1/(1+e^-x)
    var sigmoidMask = [Float](repeating: 0, count: maskSize)
    vvrecf(&sigmoidMask, &expNegatedMask, &count)
    
    return sigmoidMask
}


func cropMask(mask: [Float], width: Int, height: Int, bbox: XYXY) -> [Float] {
    var boxMask = [Float](repeating: 0, count: width * height)
    let x1 = max(0, Int(bbox.x1))
    let y1 = max(0, Int(bbox.y1))
    let x2 = min(width, Int(bbox.x2))
    let y2 = min(height, Int(bbox.y2))
    
    for y in y1..<y2 {
        let rowStart = y * width + x1
        vDSP_vfill([1.0], &boxMask[rowStart], 1, vDSP_Length(x2 - x1))
    }
    
    var result = [Float](repeating: 0, count: width * height)
    vDSP_vmul(mask, 1, boxMask, 1, &result, 1, vDSP_Length(width * height))
    return result
}


func upsampleMask(mask: [Float], width: Int, height: Int, newWidth: Int, newHeight: Int) -> [Float] {
    let sourceRowBytes = width * MemoryLayout<Float>.stride
    let sourceByteCount = sourceRowBytes * height
    
    let sourceData = UnsafeMutablePointer<Float>.allocate(capacity: width * height)
    sourceData.initialize(from: mask, count: width * height)
    
    var sourceBuffer = vImage_Buffer(
        data: sourceData,
        height: vImagePixelCount(height),
        width: vImagePixelCount(width),
        rowBytes: sourceRowBytes
    )
    
    var destinationBuffer = try! vImage_Buffer(
        width: Int(newWidth),
        height: Int(newHeight),
        bitsPerPixel: 32
    )
    
    let error = vImageScale_PlanarF(
        &sourceBuffer,
        &destinationBuffer,
        nil,
        vImage_Flags(kvImageNoFlags)
    )
    
    guard error == kvImageNoError else {
        sourceData.deallocate()
        destinationBuffer.free()
        fatalError("Error during upsampling: \(error)")
    }
    
    let result = Array(
        UnsafeBufferPointer(
            start: destinationBuffer.data.assumingMemoryBound(to: Float.self),
            count: newWidth * newHeight
        )
    )
    
    sourceData.deallocate()
    destinationBuffer.free()
    return result
}



func sigmoid(value: Float) -> Float {
    return 1.0 / (1.0 + exp(-value))
}

func IOU(box1: XYXY, box2: XYXY) -> Float {
    let xA = max(box1.x1, box2.x1)
    let yA = max(box1.y1, box2.y1)
    let xB = min(box1.x2, box2.x2)
    let yB = min(box1.y2, box2.y2)
    
    if xA >= xB || yA >= yB {
        return 0.0
    }
    
    let interArea = (xB - xA) * (yB - yA)
    let box1Area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
    let box2Area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
    
    let unionArea = box1Area + box2Area - interArea
    
    return interArea / unionArea
}
