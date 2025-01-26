extension Array {
    func concurrentMap<B>(_ transform: @escaping (Element) -> B) -> [B] {
        var result = [B?](repeating: nil, count: count)
        DispatchQueue.concurrentPerform(iterations: count) { idx in
            result[idx] = transform(self[idx])
        }
        return result.map { $0! }
    }
}
