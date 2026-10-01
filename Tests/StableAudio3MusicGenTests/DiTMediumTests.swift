import XCTest
import MLX
import MLXRandom
@testable import StableAudio3MusicGen

/// DiT-Medium shape smoke tests for the `.mediumInt8` family path.
/// Forward passes need the compiled MLX metallib (built by
/// `scripts/build_mlx_metallib.sh debug` before `swift test`).
final class DiTMediumTests: XCTestCase {

    func testDiTMediumForwardShape() throws {
        let model = DiTMedium(tLat: 22, bits: 8)
        let x = MLXRandom.normal([1, DiTMediumDims.ioChannels, 22])
        let t = MLXArray([Float(0.5)])
        let cross = MLXRandom.normal([1, 257, DiTMediumDims.condTokenDim])
        let g = MLXRandom.normal([1, DiTMediumDims.globalCondDim])
        let v = model.callAsFunction(x, t: t, crossAttnCondRaw: cross,
                                     globalCondRaw: g, localAddCond: nil)
        eval(v)
        XCTAssertEqual(v.shape, [1, DiTMediumDims.ioChannels, 22])
    }

    /// Regression: local-add-cond zeros must follow the INPUT length, not the
    /// baked `tLat` from load time. MusicForge loads with a 15 s hint
    /// (tLat 162) but the first real chunk is the 60 s opener (tLat 646) —
    /// the baked size made every Medium song longer than 15 s crash with a
    /// shape mismatch at eval.
    func testDiTMediumForwardRunsAtAnyLatentLength() throws {
        let model = DiTMedium(tLat: 162, bits: 8)
        let x = MLXRandom.normal([1, DiTMediumDims.ioChannels, 33])
        let t = MLXArray([Float(0.5)])
        let cross = MLXRandom.normal([1, 257, DiTMediumDims.condTokenDim])
        let g = MLXRandom.normal([1, DiTMediumDims.globalCondDim])
        let v = model.callAsFunction(x, t: t, crossAttnCondRaw: cross,
                                     globalCondRaw: g, localAddCond: nil)
        eval(v)
        XCTAssertEqual(v.shape, [1, DiTMediumDims.ioChannels, 33])
    }
}
