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

    // Transpose the array to shape [1, 8400, 116] (essentially swapping the last two dimensions)
    let stride = multiArray.strides.map { $0.intValue }
    let boxCount = shape[2]
    let featureCount = shape[1] // 116 features per box
    
    var boxPredictions: [BoxPrediction] = []

    for boxIndex in 0..<boxCount {
        // Transpose index calculation
        let cx = multiArray[boxIndex * stride[2]].floatValue
        let cy = multiArray[boxIndex * stride[2] + stride[1]].floatValue
        let width = multiArray[boxIndex * stride[2] + 2 * stride[1]].floatValue
        let height = multiArray[boxIndex * stride[2] + 3 * stride[1]].floatValue

        // Extract class confidences
//        let classConfidences = (4..<84).map { multiArray[boxIndex * stride[2] + $0 * stride[1]].floatValue }
        let classConfidences = (4..<(4+numClasses)).map { multiArray[boxIndex * stride[2] + $0 * stride[1]].floatValue }
        let (bestClassIndex, bestClassConfidence) = classConfidences.enumerated().max(by: { $0.element < $1.element })!

        // Check if the best confidence exceeds the threshold
        guard bestClassConfidence > confidenceThreshold else { continue }

        // Extract mask weights (32 weights)
         let maskWeights = ((4+numClasses)..<featureCount).map { multiArray[boxIndex * stride[2] + $0 * stride[1]].floatValue }
        
        let xMin = cx - width / 2
        let yMin = cy - height / 2
        let xMax = cx + width / 2
        let yMax = cy + height / 2
        
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


func getMaskProtos(
    masks: MLMultiArray,
    numMasks: Int
) -> [[Float]] {
    guard numMasks == masks.shape[1].intValue else {
        print("Incorrect mask shape: \(masks.shape)")
        return []
    }
    
    let stride = masks.strides[1].intValue
    
    let masks = (0..<numMasks).map { i -> [Float] in
        let start = i * stride
        return Array(UnsafeBufferPointer(start: masks.dataPointer.assumingMemoryBound(to: Float.self).advanced(by: start),
                                         count: stride))
    }
    
    return masks
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
