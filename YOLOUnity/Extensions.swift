import CoreML

extension Collection {
    // map, split across cores in contiguous chunks. Small collections run serially.
    func concurrentMap<T>(_ transform: (Element) -> T) -> [T] {
        concurrentEnumeratedMap { transform($1) }
    }

    func concurrentEnumeratedMap<T>(_ transform: (Int, Element) -> T) -> [T] {
        let n = count
        if n < 32 {
            return enumerated().map { transform($0.offset, $0.element) }
        }

        let elements = Array(self)
        let threadCount = Swift.min(ProcessInfo.processInfo.activeProcessorCount, n)
        let jobsPerThread = n / threadCount

        // Each thread initializes only its own slots, through a raw pointer: writing to a shared
        // Swift array from several threads at once is a data race.
        return [T](unsafeUninitializedCapacity: n) { buffer, initializedCount in
            let base = buffer.baseAddress!
            DispatchQueue.concurrentPerform(iterations: threadCount) { thread in
                let start = thread * jobsPerThread
                let end = (thread == threadCount - 1) ? n : start + jobsPerThread
                for i in start..<end {
                    (base + i).initialize(to: transform(i, elements[i]))
                }
            }
            initializedCount = n
        }
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
