import XCTest
@testable import YOLOUnity
import UIKit

class YOLOUnityTests: XCTestCase {
    func testFramework() {
        guard InitializeYOLO(modelName: "yolo11l_seg", confidenceThreshold: 0.5,
                             iouThreshold: 0.5, scaleMethod: "scaleFit") else {
            print("Failed to initialize YOLO model.")
            return
        }
        
        guard let imagePath = Bundle(for: type(of: self)).path(forResource: "demo", ofType: "jpg") else {
            print("Failed to find demo.jpg in the test bundle.")
            return
        }

        guard let image = UIImage(contentsOfFile: imagePath),
              let cgImage = image.cgImage else {
            print("Failed to load image at \(imagePath)")
            return
        }
        
        let width = cgImage.width
        let height = cgImage.height
        guard let floatArray = cgImageToFloatArray(cgImage, width: width, height: height) else {
            print("Failed to convert image to float array.")
            return
        }
        
        print("Testing with image \(width)x\(height)")

        floatArray.withUnsafeBufferPointer { buffer in
//            guard let cgImage = floatArrayToCGImage(data: buffer.baseAddress!, width: width, height: height) else { return }
//            predictor!.predict(cgImage: cgImage)
            let cvPixelBuffer = try! floatArrayToCVPixelBuffer(data: buffer.baseAddress!, width: width, height: height)!
            predictor!.predict(cvPixelBuffer: cvPixelBuffer, timestamp: getCurrentTimestamp())
        }
//        predictor!.predict(cgImage: cgImage)
        
//        print("Prediction completed.")
    }
}
