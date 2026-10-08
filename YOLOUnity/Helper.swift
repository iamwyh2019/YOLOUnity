import Foundation
import Vision
import Accelerate

func bytesToCVPixelBuffer(data: UnsafePointer<UInt8>, width: Int, height: Int, flipY: Bool = true) -> CVPixelBuffer? {
    var pixelBuffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, nil, &pixelBuffer)
   
    guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
        NSLog("Failed to create pixel buffer: \(status)")
        return nil
    }

    CVPixelBufferLockBaseAddress(buffer, .init(rawValue: 0))
    defer { CVPixelBufferUnlockBaseAddress(buffer, .init(rawValue: 0)) }

    let baseAddress = CVPixelBufferGetBaseAddress(buffer)!
    let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)

    DispatchQueue.concurrentPerform(iterations: height) { y in
        let sourceY = flipY ? (height - 1 - y) : y
        let sourceStart = sourceY * width * 4
        let destStart = y * bytesPerRow
        
        // Direct memory copy for each row
        memcpy(baseAddress + destStart, data + sourceStart, width * 4)
    }

    return buffer
}


/// Return the timestamp in milliseconds
func getCurrentTimestamp() -> UInt64 {
    return UInt64(Date().timeIntervalSince1970 * 1000)
}
