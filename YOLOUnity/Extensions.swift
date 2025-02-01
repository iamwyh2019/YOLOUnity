import CoreML

extension Array {
    func concurrentMap<B>(_ transform: @escaping (Element) -> B) -> [B] {
        var result = [B?](repeating: nil, count: count)
        DispatchQueue.concurrentPerform(iterations: count) { idx in
            result[idx] = transform(self[idx])
        }
        return result.map { $0! }
    }
    
    func concurrentEnumeratedMap<B>(_ transform: @escaping (Int, Element) -> B) -> [B] {
        var result = [B?](repeating: nil, count: count)
        DispatchQueue.concurrentPerform(iterations: count) { idx in
            result[idx] = transform(idx, self[idx])
        }
        return result.map { $0! }
    }
}

extension Range where Bound == Int {
    func concurrentMap<B>(_ transform: @escaping (Int) -> B) -> [B] {
        let count = upperBound - lowerBound
        var result = [B?](repeating: nil, count: count)
        DispatchQueue.concurrentPerform(iterations: count) { idx in
            result[idx] = transform(idx + lowerBound)
        }
        return result.map { $0! }
    }
}

// MLMultiArray extension to flatten the array
extension MLMultiArray {
    func flatArray() -> [Float] {
        let pointer = UnsafeMutablePointer<Float>(OpaquePointer(self.dataPointer))
        let buffer = UnsafeBufferPointer(start: pointer, count: self.count)
        return Array(buffer)
    }
}

// Extend OpenCVWrapper for easier Swift usage
extension OpenCVWrapper {
    static func findContours(mask: [Float], width: Int, height: Int, corner: (x: Float, y: Float) = (0.0, 0.0), coordinateRestorer: ((Float, Float) -> (Float, Float))? = nil) -> [[(Int, Int)]] {
        let nsContours = OpenCVWrapper.findContours(mask, width: Int32(width), height: Int32(height))
        return nsContours.concurrentMap { contour in
            stride(from: 0, to: (contour as! [NSNumber]).count, by: 2).map { i in
                let numbers = contour as! [NSNumber]
                let point = (Float(numbers[i].intValue) + corner.x, Float(numbers[i+1].intValue) + corner.y)
                if let restorer = coordinateRestorer {
                    let returnPoint = restorer(point.0, point.1)
                    return (Int(returnPoint.0), Int(returnPoint.1))
                }
                return (Int(point.0), Int(point.1))
            }
        }
    }
}
