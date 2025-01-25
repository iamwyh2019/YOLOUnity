import Foundation
import Vision

func convertFloatArrayToCGImage(data: UnsafePointer<Float>, width: Int, height: Int) -> CGImage? {
    let bitsPerComponent = 8
    let bitsPerPixel = 32 // 4 channels (RGBA) * 8 bits
    let bytesPerRow = width * 4 // 4 bytes per pixel (RGBA)
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

    // Allocate a buffer for RGBA UInt8 data
    let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: width * height * 4)
    defer { buffer.deallocate() }

    // Convert float data (normalized [0.0, 1.0]) to UInt8 (0-255)
    for i in 0..<(width * height) {
        let r = UInt8(max(0, min(255, data[i * 4 + 0] * 255.0)))
        let g = UInt8(max(0, min(255, data[i * 4 + 1] * 255.0)))
        let b = UInt8(max(0, min(255, data[i * 4 + 2] * 255.0)))
        let a = UInt8(max(0, min(255, data[i * 4 + 3] * 255.0))) // Alpha channel

        buffer[i * 4 + 0] = r
        buffer[i * 4 + 1] = g
        buffer[i * 4 + 2] = b
        buffer[i * 4 + 3] = a
    }

    // Create a CGImage from the buffer
    let provider = CGDataProvider(dataInfo: nil, data: buffer, size: width * height * 4, releaseData: { _, _, _ in })!
    return CGImage(width: width,
                   height: height,
                   bitsPerComponent: bitsPerComponent,
                   bitsPerPixel: bitsPerPixel,
                   bytesPerRow: bytesPerRow,
                   space: colorSpace,
                   bitmapInfo: bitmapInfo,
                   provider: provider,
                   decode: nil,
                   shouldInterpolate: true,
                   intent: .defaultIntent)
}


func cgImageToFloatArray(_ image: CGImage, width: Int, height: Int) -> [Float]? {
    // Create a raw RGBA buffer
    let bytesPerPixel = 4 // RGBA
    let byteCount = width * height * bytesPerPixel
    var rawData = [UInt8](repeating: 0, count: byteCount)

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let context = CGContext(data: &rawData,
                            width: width,
                            height: height,
                            bitsPerComponent: 8,
                            bytesPerRow: width * bytesPerPixel,
                            space: colorSpace,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)

    // Draw the image into the context
    context?.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

    // Normalize the data to float [0.0, 1.0]
    return rawData.map { Float($0) / 255.0 }
}
