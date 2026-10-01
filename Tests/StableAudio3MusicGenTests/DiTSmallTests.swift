import XCTest
import MLX
import MLXRandom
@testable import StableAudio3MusicGen

/// DiT-Small + SAME-S architecture, key rewriting, and shape smoke tests for
/// the `smallMusicInt4` port. Forward passes need the compiled MLX metallib
/// (built by `scripts/build_mlx_metallib.sh debug` before `swift test`).
final class DiTSmallTests: XCTestCase {

    // MARK: - Dims

    func testDiTSmallDimsMatchUpstream() {
        XCTAssertEqual(DiTSmallDims.ioChannels, 256)
        XCTAssertEqual(DiTSmallDims.embedDim, 1024)
        XCTAssertEqual(DiTSmallDims.depth, 20)
        XCTAssertEqual(DiTSmallDims.numHeads, 16)
        XCTAssertEqual(DiTSmallDims.headDim, 64)
        XCTAssertEqual(DiTSmallDims.ffInner, 4096)
        XCTAssertEqual(DiTSmallDims.numHeads * DiTSmallDims.headDim, DiTSmallDims.embedDim)
    }

    func testSAMESDimsMatchUpstream() {
        XCTAssertEqual(SAMESDims.dim, 768)
        XCTAssertEqual(SAMESDims.numHeads, 12)
        XCTAssertEqual(SAMESDims.numBlocks, 6)
        XCTAssertEqual(SAMESDims.ffInner, 2304)
        XCTAssertEqual(SAMESDims.outChannels, 512)
        XCTAssertEqual(SAMESDims.subChunkSize, 17)
        XCTAssertEqual(SAMESDims.sinPerPos, 16)
        // Chunk-midpoint-shift constants from same_s_decoder.py
        XCTAssertEqual(SAMESDims.chunkSizeLat, 32)
        XCTAssertEqual(SAMESDims.effectiveChunk, 34)
        XCTAssertEqual(SAMESDims.shift, 17)
    }

    func testVariantIOChannels() {
        XCTAssertEqual(StableAudio3Variant.mediumInt8.ioChannels, 256)
        XCTAssertEqual(StableAudio3Variant.smallMusicInt4.ioChannels, 256)
    }

    // MARK: - Key rewriting (shared by both DiT families)

    func testRewriteDiTKeysMapsListIndicesToNamedChildren() {
        let zeros = MLXArray.zeros([1])
        let input: [String: MLXArray] = [
            // top-level conditioner lists → inProj/outProj
            "to_cond_embed.0.weight": zeros,
            "to_cond_embed.2.weight": zeros,
            "to_global_embed.0.weight": zeros,
            "to_timestep_embed.2.bias": zeros,
            "transformer.global_cond_embedder.0.weight": zeros,
            "transformer.global_cond_embedder.2.scales": zeros,
            // per-layer lists → glu/out and inProj/outProj
            "transformer.layers.3.ff.ff.0.proj.weight": zeros,
            "transformer.layers.3.ff.ff.2.bias": zeros,
            "transformer.layers.7.to_local_embed.seq.0.weight": zeros,
            "transformer.layers.7.to_local_embed.seq.2.scales": zeros,
            // untouched keys
            "transformer.layers.3.pre_norm.weight": zeros,
            "preprocess_conv.weight": zeros,
            "transformer.memory_tokens": zeros,
        ]
        let out = sa3RewriteDiTKeys(input)
        XCTAssertEqual(out.count, input.count)
        XCTAssertNotNil(out["to_cond_embed.inProj.weight"])
        XCTAssertNotNil(out["to_cond_embed.outProj.weight"])
        XCTAssertNotNil(out["to_global_embed.inProj.weight"])
        XCTAssertNotNil(out["to_timestep_embed.outProj.bias"])
        XCTAssertNotNil(out["transformer.global_cond_embedder.inProj.weight"])
        XCTAssertNotNil(out["transformer.global_cond_embedder.outProj.scales"])
        XCTAssertNotNil(out["transformer.layers.3.ff.ff.glu.proj.weight"])
        XCTAssertNotNil(out["transformer.layers.3.ff.ff.out.bias"])
        XCTAssertNotNil(out["transformer.layers.7.to_local_embed.seq.inProj.weight"])
        XCTAssertNotNil(out["transformer.layers.7.to_local_embed.seq.outProj.scales"])
        XCTAssertNotNil(out["transformer.layers.3.pre_norm.weight"])
        XCTAssertNotNil(out["preprocess_conv.weight"])
        XCTAssertNotNil(out["transformer.memory_tokens"])
        // No numeric-index children survive.
        XCTAssertNil(out["to_cond_embed.0.weight"])
        XCTAssertNil(out["transformer.layers.3.ff.ff.2.bias"])
    }

    // MARK: - SAME-S mapping weight permutation

    func testPermuteSAMEMappingPermutesPyTorchLayout() {
        let pytorch = MLXArray.zeros([512, 768, 3])
        let out = sa3PermuteSAMEMapping(["mapping.weight": pytorch,
                                         "mapping.bias": MLXArray.zeros([512])])
        XCTAssertEqual(out["mapping.weight"]?.shape, [512, 3, 768])
        XCTAssertEqual(out["mapping.bias"]?.shape, [512])
    }

    func testPermuteSAMEMappingNoOpsOnMLXLayout() {
        let mlx = MLXArray.zeros([512, 3, 768])
        let out = sa3PermuteSAMEMapping(["mapping.weight": mlx])
        XCTAssertEqual(out["mapping.weight"]?.shape, [512, 3, 768])
    }

    // MARK: - Forward shape smoke tests

    func testDiTSmallForwardShape() throws {
        let model = DiTSmall(tLat: 22, bits: 4)
        let x = MLXRandom.normal([1, 256, 22])
        let t = MLXArray([Float(0.5)])
        let cross = MLXRandom.normal([1, 257, 768])
        let g = MLXRandom.normal([1, 768])
        let v = model.callAsFunction(x, t: t, crossAttnCondRaw: cross,
                                     globalCondRaw: g, localAddCond: nil)
        eval(v)
        XCTAssertEqual(v.shape, [1, 256, 22])
    }

    func testDiTSmallForwardRunsAtAnyLatentLength() throws {
        // Local-add-cond zeros are taken from the INPUT length, not `tLat`.
        let model = DiTSmall(tLat: 162, bits: 4)
        let x = MLXRandom.normal([1, 256, 33])
        let t = MLXArray([Float(0.5)])
        let cross = MLXRandom.normal([1, 257, 768])
        let g = MLXRandom.normal([1, 768])
        let v = model.callAsFunction(x, t: t, crossAttnCondRaw: cross,
                                     globalCondRaw: g, localAddCond: nil)
        eval(v)
        XCTAssertEqual(v.shape, [1, 256, 33])
    }

    func testSAMESDecoderForwardShape() throws {
        let model = SAMESDecoder()
        let latents = MLXRandom.normal([1, 256, 32])
        let out = model.callAsFunction(latents, fullAttention: false)
        eval(out)
        XCTAssertEqual(out.shape, [1, 512, 32 * 16])
    }

    func testSameSDecodeChunkedMatchesUnchunkedOnEvenT() throws {
        // Even T ≤ kernel takes the un-chunked fallback → identical output.
        let model = SAMESDecoder()
        let latents = MLXRandom.normal([1, 256, 32])
        let whole = model.callAsFunction(latents, fullAttention: false)
        let chunked = sameSDecodeChunked(model, latents: latents, chunkSize: 128, overlap: 8)
        eval(whole, chunked)
        XCTAssertEqual(chunked.shape, whole.shape)
        let diff = (whole - chunked).abs().max()
        eval(diff)
        XCTAssertEqual(diff.item(Float.self), 0, accuracy: 1e-5)
    }

    func testSameSDecodeChunkedSegmentsLongLatents() throws {
        let model = SAMESDecoder()
        // T = 160 > kernel (144) forces the three-segment path; every window
        // is a full even kernel so the decoder invariant holds.
        let latents = MLXRandom.normal([1, 256, 160])
        let out = sameSDecodeChunked(model, latents: latents, chunkSize: 128, overlap: 8)
        eval(out)
        XCTAssertEqual(out.shape, [1, 512, 160 * 16])
    }
}
