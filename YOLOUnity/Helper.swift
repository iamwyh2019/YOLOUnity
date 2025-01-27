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


func saveImage(mask: [Float], width: Int, height: Int, filename: String, grayscale: Bool = true) -> String {
    let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    let fullPath = documentsPath.appendingPathComponent(filename).path

    let colorSpace = grayscale ? CGColorSpaceCreateDeviceGray() : CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
    let bytesPerRow = grayscale ? width : width * 3
    
    guard mask.count == bytesPerRow * height else {
        print("Inconsistent mask size: \(mask.count) vs \(bytesPerRow * height)")
        return ""
    }

    let context = CGContext(
       data: nil,
       width: width,
       height: height,
       bitsPerComponent: 8,
       bytesPerRow: bytesPerRow,
       space: colorSpace,
       bitmapInfo: bitmapInfo.rawValue
    )!

    var clipped = Array(repeating: Float(0), count: mask.count)
    vDSP_vclip(mask, 1, [0], [1], &clipped, 1, vDSP_Length(mask.count))

    var scaled = Array(repeating: Float(0), count: mask.count)
    vDSP_vsmul(clipped, 1, [255], &scaled, 1, vDSP_Length(mask.count))

    let byteArray = scaled.map { UInt8($0) }
    context.data?.copyMemory(from: byteArray, byteCount: bytesPerRow * height)

    let image = context.makeImage()!
    let uiImage = UIImage(cgImage: image)
    let data = uiImage.pngData()!
    try! data.write(to: URL(fileURLWithPath: fullPath))
    return fullPath
}


func grayscaleToRGB(mask: [Float], width: Int, height: Int) -> [Float] {
   var rgb = [Float](repeating: 0, count: width * height * 3)
   for i in 0..<mask.count {
       let rgbIndex = i * 3
       rgb[rgbIndex] = mask[i]
       rgb[rgbIndex + 1] = mask[i]
       rgb[rgbIndex + 2] = mask[i]
   }
   return rgb
}


func drawContour(
   mask: [Float],
   width: Int,
   height: Int,
   contours: [[(x: Int, y: Int)]],
   color: (r: Float, g: Float, b: Float) = (1, 0, 0),
   contourWidth: Int = 1
) -> [Float] {
   var result = mask
   
   for contour in contours {
       for i in 0..<contour.count {
           let p1 = contour[i]
           let p2 = contour[(i + 1) % contour.count]
           
           // Bresenham's line algorithm
           var x = p1.x
           var y = p1.y
           let dx = abs(p2.x - p1.x)
           let dy = abs(p2.y - p1.y)
           let sx = p1.x < p2.x ? 1 : -1
           let sy = p1.y < p2.y ? 1 : -1
           var err = dx - dy
           
           while true {
               // Draw thick line by filling a square around each point
               for offsetY in -contourWidth/2...contourWidth/2 {
                   for offsetX in -contourWidth/2...contourWidth/2 {
                       let px = x + offsetX
                       let py = y + offsetY
                       if px >= 0 && px < width && py >= 0 && py < height {
                           let idx = (py * width + px) * 3
                           result[idx] = color.r
                           result[idx + 1] = color.g
                           result[idx + 2] = color.b
                       }
                   }
               }
               
               if x == p2.x && y == p2.y { break }
               let e2 = 2 * err
               if e2 > -dy {
                   err -= dy
                   x += sx
               }
               if e2 < dx {
                   err += dx
                   y += sy
               }
           }
       }
   }
   return result
}

func fillPoly(mask: [Float], width: Int, height: Int, contours: [[(x: Int, y: Int)]], color: (r: Float, g: Float, b: Float) = (1, 0, 0)) -> [Float] {
   var result = mask
   
   for contour in contours {
       // For each scanline
       var minY = Int.max
       var maxY = Int.min
       for point in contour {
           minY = min(minY, point.y)
           maxY = max(maxY, point.y)
       }
       
       for y in max(0, minY)...min(height-1, maxY) {
           var intersections: [Int] = []
           
           // Find intersections with scanline
           for i in 0..<contour.count {
               let p1 = contour[i]
               let p2 = contour[(i + 1) % contour.count]
               
               if (p1.y > y && p2.y <= y) || (p2.y > y && p1.y <= y) {
                   let x = p1.x + (p2.x - p1.x) * (y - p1.y) / (p2.y - p1.y)
                   intersections.append(x)
               }
           }
           
           // Sort intersections
           intersections.sort()
           
           // Fill between pairs of intersections
           for i in stride(from: 0, to: intersections.count-1, by: 2) {
               let startX = max(0, intersections[i])
               let endX = min(width-1, intersections[i+1])
               
               for x in startX...endX {
                   let idx = (y * width + x) * 3
                   result[idx] = color.r
                   result[idx + 1] = color.g
                   result[idx + 2] = color.b
               }
           }
       }
   }
   return result
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

/// Return the timestamp in milliseconds
func getCurrentTimestamp() -> UInt64 {
    return UInt64(Date().timeIntervalSince1970 * 1000)
}
