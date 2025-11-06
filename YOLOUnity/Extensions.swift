import CoreML

// Extension for using sequence operations with memory safety
extension Collection {
    func concurrentMap<T>(_ transform: (Element) -> T) -> [T] {
        let n = count
        if n == 0 {
            return []
        }
        
        // For small collections, don't parallelize
        if n < 32 {
            return map(transform)
        }
        
        // Use a specific number of threads based on system
        let threadCount = Swift.min(ProcessInfo.processInfo.activeProcessorCount, n)
        let jobsPerThread = Swift.max(1, n / threadCount)
        
        var result = [T?](repeating: nil, count: n)
        DispatchQueue.concurrentPerform(iterations: threadCount) { thread in
            let start = thread * jobsPerThread
            let end = (thread == threadCount - 1) ? n : start + jobsPerThread
            for i in start..<end {
                let index = self.index(self.startIndex, offsetBy: i)
                result[i] = transform(self[index])
            }
        }
        
        return result.compactMap { $0 }
    }
    
    func concurrentEnumeratedMap<T>(_ transform: (Int, Element) -> T) -> [T] {
        let n = count
        if n == 0 {
            return []
        }
        
        // For small collections, don't parallelize
        if n < 32 {
            return enumerated().map { transform($0.offset, $0.element) }
        }
        
        // Use a specific number of threads based on system
        let threadCount = Swift.min(ProcessInfo.processInfo.activeProcessorCount, n)
        let jobsPerThread = Swift.max(1, n / threadCount)
        
        var result = [T?](repeating: nil, count: n)
        DispatchQueue.concurrentPerform(iterations: threadCount) { thread in
            let start = thread * jobsPerThread
            let end = (thread == threadCount - 1) ? n : start + jobsPerThread
            for i in start..<end {
                let index = self.index(self.startIndex, offsetBy: i)
                result[i] = transform(i, self[index])
            }
        }
        
        return result.compactMap { $0 }
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
