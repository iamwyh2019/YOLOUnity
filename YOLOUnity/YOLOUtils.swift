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


func cropMaskPhysical(mask: [Float], width: Int, height: Int, bbox: XYXY) -> ([Float], (width: Int, height: Int)) {
   let x1 = max(0, Int(bbox.x1))
   let y1 = max(0, Int(bbox.y1))
   let x2 = min(width - 1, Int(bbox.x2))
   let y2 = min(height - 1, Int(bbox.y2))
   let cropWidth = x2 - x1 + 1
   let cropHeight = y2 - y1 + 1
   
   var cropped = [Float](repeating: 0, count: cropWidth * cropHeight)
   
   mask.withUnsafeBufferPointer { srcPtr in
       cropped.withUnsafeMutableBufferPointer { dstPtr in
           DispatchQueue.concurrentPerform(iterations: cropHeight) { y in
               let srcOffset = (y + y1) * width + x1
               let dstOffset = y * cropWidth
               memcpy(dstPtr.baseAddress! + dstOffset,
                     srcPtr.baseAddress! + srcOffset,
                     cropWidth * MemoryLayout<Float>.stride)
           }
       }
   }
   
   return (cropped, (cropWidth, cropHeight))
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


func removeBelowThreshold(mask: [Float], threshold: Float = 0.5) -> [Float] {
    /// makes all entries lower than threshold to be 0
    /// values above threshold is left as it is, but it doesn't affect finding contours
    return vDSP.threshold(mask, to: threshold, with: .zeroFill)
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


func parseModelSizeAndNames(model: MLModel) -> (width: Int, height: Int, names: [Int: String]) {
    // this is representation in Python, need to change to standard JSON
    var classNames: [Int: String] = [:]
    var width: Int = -1
    var height: Int = -1
    
    if let metadata = model.modelDescription.metadata[.creatorDefinedKey] as? [String: Any] {
        if let namesData = metadata["names"] as? String {
            // Preprocess the JSON string
            var fixedJSONString = namesData.replacingOccurrences(of: "'", with: "\"") // Convert single quotes to double quotes

            // Use NSRegularExpression to quote keys
            if let regex = try? NSRegularExpression(pattern: "(\\d+):", options: []) {
                let range = NSRange(location: 0, length: fixedJSONString.utf16.count)
                fixedJSONString = regex.stringByReplacingMatches(in: fixedJSONString, options: [], range: range, withTemplate: "\"$1\":")
            }

            // Decode the fixed JSON
            if let jsonData = fixedJSONString.data(using: .utf8) {
                do {
                    classNames = try JSONDecoder().decode([Int: String].self, from: jsonData)
                } catch {
                    NSLog("Error decoding JSON: \(error)")
                }
            } else {
                NSLog("Error: Could not convert string to data.")
            }
        } else {
            NSLog("Error: `names` field not found or invalid.")
        }
        
        if let sizeData = metadata["imgsz"] as? String {
            // sizeData is in the form of "[%d, %d]"
            let pattern = #"^\[\s*(\d+),\s*(\d+)\s*\]$"#
            if let regex = try? NSRegularExpression(pattern: pattern) {
                // Search for matches
                if let match = regex.firstMatch(in: sizeData, range: NSRange(sizeData.startIndex..., in: sizeData)) {
                    // Extract the matched groups
                    if let range1 = Range(match.range(at: 1), in: sizeData),
                       let range2 = Range(match.range(at: 2), in: sizeData) {
                        let number1 = Int(sizeData[range1])
                        let number2 = Int(sizeData[range2])
                        
                        if let number1 = number1, let number2 = number2 {
                            width = number1
                            height = number2
                        } else {
                            NSLog("Error: failed to parse `imgsz` field as integers: \"\(sizeData)\".")
                        }
                    } else {
                        NSLog("Error: failed to parse `imgsz` field as integers: \"\(sizeData)\".")
                    }
                }
                else {
                    NSLog("Error: failed to parse `imgsz` field as integers: \"\(sizeData)\".")
                }
            } else {
                NSLog("Error: failed to parse `imgsz` field as integers: \"\(sizeData)\".")
            }
        } else {
            NSLog("Error: `imgsz` field not found or invalid.")
        }
    } else {
        NSLog("Cannot find metadata in model description.")
    }
    
    return (width: width, height: height, names: classNames)
}
