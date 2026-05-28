import Foundation
import Vision
import UIKit

// Callback type
//public typealias YOLOCallback = @convention(c) (UnsafePointer<Float>, Int, UnsafePointer<Float>, Int) -> Void
public typealias YOLOCallback = @convention(c) (
    Int32,                          // number of detections
    UnsafePointer<Int32>,           // classIndex
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

// Cached null-separated UTF-8 buffer of class names, indexed by classIndex.
// Allocated lazily on first GetClassNames call; freed and rebuilt when a new
// predictor is initialized via InitializeYOLO.
private var cachedClassNamesPtr: UnsafeMutablePointer<UInt8>? = nil
private var cachedClassNamesByteLength: Int32 = 0
private var cachedClassNamesCount: Int32 = 0

private func releaseClassNamesCache() {
    if let ptr = cachedClassNamesPtr {
        ptr.deallocate()
        cachedClassNamesPtr = nil
    }
    cachedClassNamesByteLength = 0
    cachedClassNamesCount = 0
}

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
    releaseClassNamesCache()
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

// Returns a pointer to a null-separated UTF-8 buffer of class names, indexed by
// classIndex (0..count-1). The buffer is owned by the plugin and remains valid
// until the next InitializeYOLO call. Missing/sparse class indices are encoded
// as empty strings (consecutive null bytes).
@_cdecl("GetClassNames")
public func GetClassNames(
    outCount: UnsafeMutablePointer<Int32>,
    outByteLength: UnsafeMutablePointer<Int32>
) -> UnsafePointer<UInt8>? {
    if cachedClassNamesPtr == nil {
        guard let predictor = predictor else {
            outCount.pointee = 0
            outByteLength.pointee = 0
            return nil
        }
        let maxIdx = predictor.classNames.keys.max() ?? -1
        let count = maxIdx + 1
        var bytes: [UInt8] = []
        for i in 0..<count {
            if let name = predictor.classNames[i] {
                bytes.append(contentsOf: name.utf8)
            }
            bytes.append(0)
        }
        let ptr = UnsafeMutablePointer<UInt8>.allocate(capacity: bytes.count)
        for (i, b) in bytes.enumerated() {
            ptr[i] = b
        }
        cachedClassNamesPtr = ptr
        cachedClassNamesByteLength = Int32(bytes.count)
        cachedClassNamesCount = Int32(count)
    }
    outCount.pointee = cachedClassNamesCount
    outByteLength.pointee = cachedClassNamesByteLength
    return UnsafePointer(cachedClassNamesPtr!)
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
