import Foundation
import Vision
import UniformTypeIdentifiers

func floatArrayToCGImage(data: UnsafePointer<Float>, width: Int, height: Int) -> CGImage? {
    let bytesPerPixel = 4
    let byteCount = width * height * bytesPerPixel
    let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: byteCount)
    let retainedBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: byteCount)
    defer { buffer.deallocate() }
    
    DispatchQueue.concurrentPerform(iterations: height) { y in
        let rowStart = y * width
        for x in 0..<width {
            let i = rowStart + x
            let baseIndex = i * 4
            buffer[baseIndex + 0] = UInt8(max(0, min(255, data[baseIndex + 0] * 255.0)))
            buffer[baseIndex + 1] = UInt8(max(0, min(255, data[baseIndex + 1] * 255.0)))
            buffer[baseIndex + 2] = UInt8(max(0, min(255, data[baseIndex + 2] * 255.0)))
            buffer[baseIndex + 3] = UInt8(max(0, min(255, data[baseIndex + 3] * 255.0)))
        }
    }
    
    memcpy(retainedBuffer, buffer, byteCount)
    
    let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
    
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


func cgImageToFloatArray(_ image: CGImage, width: Int, height: Int) -> [Float]? {
    let bytesPerPixel = 4 // RGBA
    let byteCount = width * height * bytesPerPixel
    var rawData = [UInt8](repeating: 0, count: byteCount)

    let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    let bitmapInfo: CGBitmapInfo = [
        .byteOrder32Big, // Ensure proper byte order
        CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
    ]

    guard let context = CGContext(data: &rawData,
                                  width: width,
                                  height: height,
                                  bitsPerComponent: 8,
                                  bytesPerRow: width * bytesPerPixel,
                                  space: colorSpace,
                                  bitmapInfo: bitmapInfo.rawValue) else {
        print("Error: Failed to create CGContext.")
        return nil
    }

    // Draw the image into the context
    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

    // Debug: Verify raw data size
    if rawData.count != byteCount {
        print("Error: Mismatched raw data size. Expected \(byteCount), got \(rawData.count).")
        return nil
    }

    // Normalize the data to float [0.0, 1.0]
    return rawData.map { Float($0) / 255.0 }
}


func floatArrayToCVPixelBuffer(data: UnsafePointer<Float>, width: Int, height: Int) -> CVPixelBuffer? {
   var pixelBuffer: CVPixelBuffer?
   CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, nil, &pixelBuffer)
   
   guard let buffer = pixelBuffer else { return nil }
   
   CVPixelBufferLockBaseAddress(buffer, .init(rawValue: 0))
   defer { CVPixelBufferUnlockBaseAddress(buffer, .init(rawValue: 0)) }
   
   let baseAddress = CVPixelBufferGetBaseAddress(buffer)!
   let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
   
   DispatchQueue.concurrentPerform(iterations: height) { y in
       let uint8Buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
       for x in 0..<width {
           let sourceIndex = (y * width + x) * 4
           let destIndex = y * bytesPerRow + x * 4
           
           uint8Buffer[destIndex + 0] = UInt8(max(0, min(255, data[sourceIndex + 0] * 255.0))) // B
           uint8Buffer[destIndex + 1] = UInt8(max(0, min(255, data[sourceIndex + 1] * 255.0))) // G
           uint8Buffer[destIndex + 2] = UInt8(max(0, min(255, data[sourceIndex + 2] * 255.0))) // R
           uint8Buffer[destIndex + 3] = UInt8(max(0, min(255, data[sourceIndex + 3] * 255.0))) // A
       }
   }
   
   return buffer
}



func saveCGImageToDisk(cgImage: CGImage, filename: String) {
    // Create a retained copy of the CGImage
    guard let imageCopy = cgImage.copy() else {
        print("Error: Failed to copy CGImage")
        return
    }

    let url = FileManager.default.temporaryDirectory.appendingPathComponent(filename)

    guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
        print("Error: Failed to create image destination.")
        return
    }

    CGImageDestinationAddImage(destination, imageCopy, nil)
    if CGImageDestinationFinalize(destination) {
        print("Image successfully saved to \(url.path)")
    } else {
        print("Error: Failed to save image to disk.")
    }
}

