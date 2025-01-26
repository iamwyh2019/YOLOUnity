import Foundation
import Vision
import UniformTypeIdentifiers
import Accelerate
import UIKit

func floatArrayToCGImage(data: UnsafePointer<Float>, width: Int, height: Int, flipY: Bool = true) -> CGImage? {
    let bytesPerPixel = 4
    let byteCount = width * height * bytesPerPixel
    let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: byteCount)
    let retainedBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: byteCount)
    defer { buffer.deallocate() }
    
    DispatchQueue.concurrentPerform(iterations: height) { y in
        for x in 0..<width {
            let sourceY = flipY ? (height - 1 - y) : y
            let sourceIndex = (sourceY * width + x) * 4
            let destIndex = y * width * 4 + x * 4

            buffer[destIndex + 0] = UInt8(max(0, min(255, data[sourceIndex + 0] * 255.0)))
            buffer[destIndex + 1] = UInt8(max(0, min(255, data[sourceIndex + 1] * 255.0)))
            buffer[destIndex + 2] = UInt8(max(0, min(255, data[sourceIndex + 2] * 255.0)))
            buffer[destIndex + 3] = UInt8(max(0, min(255, data[sourceIndex + 3] * 255.0)))
        }
    }
    
    memcpy(retainedBuffer, buffer, byteCount)
    
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue)
    
    guard let provider = CGDataProvider(dataInfo: nil, data: retainedBuffer, size: byteCount, releaseData: { _, ptr, _ in
        let bufferPtr = ptr.assumingMemoryBound(to: UInt8.self)
        bufferPtr.deallocate()
    }) else {
        retainedBuffer.deallocate()
        return nil
    }
    
    return CGImage(width: width, height: height,
                  bitsPerComponent: 8, bitsPerPixel: bytesPerPixel * 8,
                  bytesPerRow: width * bytesPerPixel,
                  space: colorSpace, bitmapInfo: bitmapInfo,
                  provider: provider, decode: nil,
                  shouldInterpolate: false, intent: .defaultIntent)
}


func cgImageToFloatArray(_ image: CGImage, width: Int, height: Int, flipY: Bool = true) -> [Float]? {
    let bytesPerPixel = 4
    let byteCount = width * height * bytesPerPixel
    let context = CGContext(data: nil,
                         width: width,
                         height: height,
                         bitsPerComponent: 8,
                         bytesPerRow: width * bytesPerPixel,
                         space: CGColorSpaceCreateDeviceRGB(),
                         bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)

    guard let context = context else { return nil }
    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

    guard let ptr = context.data?.assumingMemoryBound(to: UInt8.self) else { return nil }
    let buffer = UnsafeBufferPointer(start: ptr, count: byteCount)
    let floatArray = buffer.map { Float($0) / 255.0 }

    return flipY ? flipYAxis(image: floatArray, width: width, height: height) : floatArray
}


func floatArrayToCVPixelBuffer(data: UnsafePointer<Float>, width: Int, height: Int, flipY: Bool = true) -> CVPixelBuffer? {
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
        let uint8Buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
        let sourceY = flipY ? (height - 1 - y) : y
        for x in 0..<width {
        //           let sourceIndex = (y * width + x) * 4
        //           let destIndex = y * bytesPerRow + x * 4
            let sourceIndex = (sourceY * width + x) * 4
            let destIndex = y * bytesPerRow + x * 4
           
            uint8Buffer[destIndex + 0] = UInt8(max(0, min(255, data[sourceIndex + 0] * 255.0))) // R
            uint8Buffer[destIndex + 1] = UInt8(max(0, min(255, data[sourceIndex + 1] * 255.0))) // G
            uint8Buffer[destIndex + 2] = UInt8(max(0, min(255, data[sourceIndex + 2] * 255.0))) // B
            uint8Buffer[destIndex + 3] = UInt8(max(0, min(255, data[sourceIndex + 3] * 255.0))) // A
        }
    }

    return buffer
}



func saveCGImageToDisk(cgImage: CGImage, filename: String) {
    // Create a retained copy of the CGImage
    guard let imageCopy = cgImage.copy() else {
        NSLog("Error: Failed to copy CGImage")
        return
    }

    let url = FileManager.default.temporaryDirectory.appendingPathComponent(filename)

    guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
        NSLog("Error: Failed to create image destination.")
        return
    }

    CGImageDestinationAddImage(destination, imageCopy, nil)
    if CGImageDestinationFinalize(destination) {
        NSLog("Image successfully saved to \(url.path)")
    } else {
        NSLog("Error: Failed to save image to disk.")
    }
}


func saveGrayscaleImage(mask: [Float], width: Int, height: Int, filename: String) -> String {
    let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    let fullPath = documentsPath.appendingPathComponent(filename).path
    
    let colorSpace = CGColorSpaceCreateDeviceGray()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
    
    let context = CGContext(
        data: nil,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: width,
        space: colorSpace,
        bitmapInfo: bitmapInfo.rawValue
    )!
    
    var clipped = Array(repeating: Float(0), count: mask.count)
    vDSP_vclip(mask, 1, [0], [1], &clipped, 1, vDSP_Length(mask.count))
    
    let binarized_mask: [Float] = mask.map { $0 > 0.5 ? 1 : 0 }
    
    var scaled = Array(repeating: Float(0), count: mask.count)
    vDSP_vsmul(binarized_mask, 1, [255], &scaled, 1, vDSP_Length(mask.count))
    
    let byteArray = scaled.map { UInt8($0) }
    context.data?.copyMemory(from: byteArray, byteCount: width * height)
    
    let image = context.makeImage()!
    let uiImage = UIImage(cgImage: image)
    let data = uiImage.pngData()!
    try! data.write(to: URL(fileURLWithPath: fullPath))
    return fullPath
}


func flipYAxis(image: [Float], width: Int, height: Int) -> [Float] {
    var flipped = [Float](repeating: 0, count: width * height * 4)
    for y in 0..<height {
        let srcRow = (height - 1 - y) * width * 4
        let dstRow = y * width * 4
        for x in 0..<(width * 4) {
            flipped[dstRow + x] = image[srcRow + x]
        }
    }
    return flipped
}
