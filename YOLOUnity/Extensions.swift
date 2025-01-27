import CoreML

extension Array {
    func concurrentMap<B>(_ transform: @escaping (Element) -> B) -> [B] {
        var result = [B?](repeating: nil, count: count)
        DispatchQueue.concurrentPerform(iterations: count) { idx in
            result[idx] = transform(self[idx])
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
