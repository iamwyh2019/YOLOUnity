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

   let boxCount = multiArray.shape[2].intValue
   let featureCount = multiArray.shape[1].intValue
   let numMasks = featureCount - (4 + numClasses)
   let pointer = multiArray.dataPointer.assumingMemoryBound(to: Float.self)
   
   let predictions = Array(0..<boxCount).concurrentMap { i -> BoxPrediction? in
       var maxConfidence: Float = 0
       var bestClassIndex = 0
       for c in 0..<numClasses {
           let conf = pointer[(4+c)*boxCount + i]
           if conf > maxConfidence {
               maxConfidence = conf
               bestClassIndex = c
           }
       }
       
       guard maxConfidence > confidenceThreshold else { return nil }
       
       let cx = pointer[0*boxCount + i]
       let cy = pointer[1*boxCount + i]
       let width = pointer[2*boxCount + i]
       let height = pointer[3*boxCount + i]
       
       let maskWeights = (0..<numMasks).map { m ->
           Float in pointer[(4 + numClasses + m)*boxCount + i]
       }
       
       return BoxPrediction(
           classIndex: bestClassIndex,
           score: maxConfidence,
           xyxy: XYXY(
               x1: cx - width/2,
               y1: cy - height/2,
               x2: cx + width/2,
               y2: cy + height/2
           ),
           maskCoefficients: maskWeights
       )
   }.compactMap { $0 }

   return predictions
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


func getMasksFromProtos(maskProtos: [[Float]], coefficients: [Float]) -> [Float] {
    guard !maskProtos.isEmpty, maskProtos.count == coefficients.count else { return [] }
    
    let maskSize = maskProtos[0].count
    var result = [Float](repeating: 0, count: maskSize)
    
    for (proto, coefficient) in zip(maskProtos, coefficients) {
        // Multiply proto by the scalar coefficient and add the result to 'result'
        vDSP_vsma(proto, 1, [coefficient], result, 1, &result, 1, vDSP_Length(maskSize))
    }
    
    return result
}


func getSigmoidMask(mask: [Float]) -> [Float] {
    let maskSize = mask.count
    var count = Int32(maskSize)
    var onef: Float = 1.0
    
    // step 1: x -> e^-x
    let negatedMask = vDSP.multiply(-1.0, mask)
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
    var result = [Float](repeating: 0, count: width * height)
    let x1 = max(0, Int(bbox.x1))
    let y1 = max(0, Int(bbox.y1))
    let x2 = min(width - 1, Int(bbox.x2))
    let y2 = min(height - 1, Int(bbox.y2))
    let rowWidth = x2 - x1 + 1

    DispatchQueue.concurrentPerform(iterations: y2 - y1 + 1) { i in
       let y = y1 + i
       let offset = y * width + x1
       mask.withUnsafeBufferPointer { ptr in
           vDSP_vsmul(ptr.baseAddress! + offset, 1, [1.0], &result[offset], 1, vDSP_Length(rowWidth))
       }
    }

    return result
}


func upsampleMask(mask: [Float], width: Int, height: Int, newWidth: Int, newHeight: Int) -> [Float] {
    let sourceRowBytes = width * MemoryLayout<Float>.stride
    
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
        NSLog("Error during upsampling: \(error)")
        return []
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
