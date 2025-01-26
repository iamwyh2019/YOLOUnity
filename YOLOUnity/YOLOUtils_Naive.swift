import CoreML
import Vision
import UIKit
import Accelerate

func parseBoundingBoxes_Naive(
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


func getMasksFromProtos_Naive(
    maskProtos: [[Float]],
    coefficients: [Float]
) -> [Float] {
    guard maskProtos.count == coefficients.count else {
        NSLog("Unmatched length: \(maskProtos.count) vs \(coefficients.count)")
        return []
    }
    
    guard !maskProtos.isEmpty else {
        NSLog("No masks provided")
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
