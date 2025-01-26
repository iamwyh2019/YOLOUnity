import Foundation
import Vision
import UIKit

// Callback type
//public typealias YOLOCallback = @convention(c) (UnsafePointer<Float>, Int, UnsafePointer<Float>, Int) -> Void
public typealias YOLOCallback = @convention(c) (Int) -> Void

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
    height: Int
) {
    guard let predictor = predictor else {
        NSLog("Error: YOLOPredictor not initialized.")
        return
    }
    
    guard let cgImage = floatArrayToCGImage(data: imageData, width: width, height: height) else {
        NSLog("Error: Failed to convert image data.")
        return
    }

    predictor.predict(cgImage: cgImage)
}
