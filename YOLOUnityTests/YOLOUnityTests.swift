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
        
        RegisterYOLOCallback { result1, count1, result2, count2 in
            let array1 = Array(UnsafeBufferPointer(start: result1, count: count1))
            let array2 = Array(UnsafeBufferPointer(start: result2, count: count2))
            print("Result 1: \(array1)")
            print("Result 2: \(array2)")
        }

        floatArray.withUnsafeBufferPointer { buffer in
            guard let cgImage = floatArrayToCGImage(data: buffer.baseAddress!, width: width, height: height) else { return }
            predictor!.predict(cgImage: cgImage)
        }
        
        print("Prediction completed.")
    }
}
