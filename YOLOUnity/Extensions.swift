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
    struct ContoursResult {
        let contours: [[(Int, Int)]]
        let centroid: (Float, Float)
    }
    
    static func findContours(
        mask: [Float],
        width: Int,
        height: Int,
        corner: (x: Float, y: Float) = (0.0, 0.0),
        coordinateRestorer: ((Float, Float) -> (Float, Float))? = nil
    ) -> ContoursResult {
        let result = OpenCVWrapper.findContours(mask, width: Int32(width), height: Int32(height))
        
        guard let dict = result as? [String: Any],
              let nsContours = dict["contours"] as? [NSArray],
              let centroidArray = dict["centroid"] as? [NSNumber] else {
            return ContoursResult(contours: [], centroid: (0, 0))
        }
        
        // Process contours
        let contours = nsContours.concurrentMap { contour in
            stride(from: 0, to: (contour as! [NSNumber]).count, by: 2).map { i in
                let numbers = contour as! [NSNumber]
                let point = (Float(numbers[i].intValue) + corner.x,
                           Float(numbers[i+1].intValue) + corner.y)
                
                if let restorer = coordinateRestorer {
                    let restored = restorer(point.0, point.1)
                    return (Int(restored.0), Int(restored.1))
                }
                return (Int(point.0), Int(point.1))
            }
        }
        
        // Process centroid
        let rawCentroid = (Float(centroidArray[0].doubleValue) + corner.x,
                          Float(centroidArray[1].doubleValue) + corner.y)
        let centroid = coordinateRestorer?(rawCentroid.0, rawCentroid.1) ?? rawCentroid
        
        return ContoursResult(contours: contours, centroid: centroid)
    }
}
