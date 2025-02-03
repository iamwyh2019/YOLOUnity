import Foundation
import Vision
import UIKit

// Callback type
//public typealias YOLOCallback = @convention(c) (UnsafePointer<Float>, Int, UnsafePointer<Float>, Int) -> Void
public typealias YOLOCallback = @convention(c) (
    Int32,                          // number of detections
    UnsafePointer<Int32>,           // classIndex
    UnsafePointer<UInt8>, Int32,   // names data, total name bytes
    UnsafePointer<Float>,          // scores (length = numDetections)
    UnsafePointer<Int32>,          // boxes (length = numDetections * 4)
    UnsafePointer<Int32>, Int32,   // contour points, count
    UnsafePointer<Int32>, Int32,   // contour indices, count
    UnsafePointer<Int32>,           // centroids (length = numDetections * 2)
    UInt64                          // timestamp
) -> Void

// Global variables
var predictor: YOLOPredictor? = nil
var yoloCallback: YOLOCallback? = nil

// Register the callback
@_cdecl("RegisterYOLOCallback")
public func RegisterYOLOCallback(callback: @escaping YOLOCallback) {
    yoloCallback = callback
}


@_cdecl("InitializeYOLO")
public func InitializeYOLO(
    modelName: UnsafePointer<CChar>,
    confidenceThreshold: Float,
    iouThreshold: Float,
    scaleMethod: UnsafePointer<CChar>
) -> Bool {
    let name = String(cString: modelName)
    let scaleMethodStr = String(cString: scaleMethod)
    predictor = YOLOPredictor(
        modelName: name,
        confidanceThreshold: confidenceThreshold,
        iouThreshold: iouThreshold,
        scaleMethod: scaleMethodStr
    )
    return predictor != nil
}


@_cdecl("RunYOLO")
public func RunYOLO(
    imageData: UnsafePointer<Float>,
    width: Int,
    height: Int,
    timestamp: UInt64 = 0,
    scaleX: Float = 1.0,
    scaleY: Float = 1.0
) {
    guard let predictor = predictor else {
        NSLog("Error: YOLOPredictor not initialized.")
        return
    }
    
    guard let cvPixelBuffer = floatArrayToCVPixelBuffer(data: imageData, width: width, height: height) else {
        NSLog("Error: Failed to convert image data.")
        return
    }
    
    predictor.predict(
        cvPixelBuffer: cvPixelBuffer,
        timestamp: timestamp == 0 ? getCurrentTimestamp() : timestamp,
        scaleX: scaleX,
        scaleY: scaleY
    )
}

@_cdecl("RunYOLO_Byte")
public func RunYOLO_Byte(
    imageData: UnsafePointer<UInt8>,
    width: Int,
    height: Int,
    timestamp: UInt64 = 0,
    scaleX: Float = 1.0,
    scaleY: Float = 1.0
) {
    guard let predictor = predictor else {
        NSLog("Error: YOLOPredictor not initialized.")
        return
    }
    
    guard let cvPixelBuffer = bytesToCVPixelBuffer(data: imageData, width: width, height: height) else {
        NSLog("Error: Failed to convert image data.")
        return
    }
    
    predictor.predict(
        cvPixelBuffer: cvPixelBuffer,
        timestamp: timestamp == 0 ? getCurrentTimestamp() : timestamp,
        scaleX: scaleX,
        scaleY: scaleY
    )
}
