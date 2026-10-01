import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Differential attention (SAME-S: no sliding-window mask — the
// chunk-midpoint-shift reshaping below plays that role)

/// Differential SDPA: `out = SDPA(q,k,v) - SDPA(q_diff,k_diff,v)`.
/// `to_qkv` chunk(5) order: q, k, v, q_diff, k_diff. Attention is always
/// full over whatever window the caller reshapes to (SAME-L is the variant
/// with the 17×51 SWA mask).
public final class DiffAttentionS: Module {
    @ModuleInfo(key: "to_qkv") public var toQKV: Linear      // FP32 in SAME-S
    @ModuleInfo(key: "to_out") public var toOut: Linear
    @ModuleInfo(key: "q_norm") public var qNorm: DyT
    @ModuleInfo(key: "k_norm") public var kNorm: DyT

    public let scale: Float

    public override init() {
        let D = SAMESDims.dim
        self._toQKV.wrappedValue = Linear(D, 5 * D, bias: false)
        self._toOut.wrappedValue = Linear(D, D, bias: false)
        self._qNorm.wrappedValue = DyT(dim: SAMESDims.headDim)
        self._kNorm.wrappedValue = DyT(dim: SAMESDims.headDim)
        self.scale = 1.0 / Float(SAMESDims.headDim).squareRoot()
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0), T = x.dim(1)
        let H = SAMESDims.numHeads, D = SAMESDims.headDim
        let qkv = toQKV(x)
        let parts = MLX.split(qkv, parts: 5, axis: -1)
        var q = parts[0], k = parts[1]
        let v = parts[2]
        var qDiff = parts[3], kDiff = parts[4]

        func toHeads(_ t: MLXArray) -> MLXArray {
            t.reshaped([B, T, H, D]).transposed(0, 2, 1, 3)
        }
        q = toHeads(q); k = toHeads(k)
        let vh = toHeads(v)
        qDiff = toHeads(qDiff); kDiff = toHeads(kDiff)

        q = qNorm(q); k = kNorm(k)
        qDiff = qNorm(qDiff); kDiff = kNorm(kDiff)

        q      = MLXFast.RoPE(q,      dimensions: SAMESDims.ropeDims, traditional: false,
                               base: SAMESDims.ropeBase, scale: 1.0, offset: 0)
        k      = MLXFast.RoPE(k,      dimensions: SAMESDims.ropeDims, traditional: false,
                               base: SAMESDims.ropeBase, scale: 1.0, offset: 0)
        qDiff  = MLXFast.RoPE(qDiff,  dimensions: SAMESDims.ropeDims, traditional: false,
                               base: SAMESDims.ropeBase, scale: 1.0, offset: 0)
        kDiff  = MLXFast.RoPE(kDiff,  dimensions: SAMESDims.ropeDims, traditional: false,
                               base: SAMESDims.ropeBase, scale: 1.0, offset: 0)

        let outMain = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: vh,
                                                         scale: scale, mask: nil)
        let outDiff = MLXFast.scaledDotProductAttention(queries: qDiff, keys: kDiff, values: vh,
                                                         scale: scale, mask: nil)
        let outF = (outMain - outDiff).transposed(0, 2, 1, 3).reshaped([B, T, SAMESDims.dim])
        return toOut(outF)
    }
}

// MARK: - GeGLU feed-forward (SiLU in EVERY block — SAME-S has no sinusoidal gate)

public final class FeedForwardS: Module {
    @ModuleInfo(key: "glu_proj") public var gluProj: Linear
    @ModuleInfo(key: "proj_out") public var projOut: Linear

    public override init() {
        self._gluProj.wrappedValue = Linear(SAMESDims.dim, SAMESDims.ffInner * 2, bias: true)
        self._projOut.wrappedValue = Linear(SAMESDims.ffInner, SAMESDims.dim, bias: true)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let p = gluProj(x)
        let parts = MLX.split(p, parts: 2, axis: -1)
        let value = parts[0], gate = parts[1]
        return projOut(value * silu(gate))
    }
}

public final class TransformerBlockS: Module {
    @ModuleInfo(key: "pre_norm") public var preNorm: DyT
    @ModuleInfo public var attn: DiffAttentionS
    @ModuleInfo(key: "ff_norm") public var ffNorm: DyT
    @ModuleInfo public var ff: FeedForwardS

    public override init() {
        self._preNorm.wrappedValue = DyT(dim: SAMESDims.dim)
        self._attn.wrappedValue = DiffAttentionS()
        self._ffNorm.wrappedValue = DyT(dim: SAMESDims.dim)
        self._ff.wrappedValue = FeedForwardS()
        super.init()
    }

    public func callAsFunction(_ xIn: MLXArray) -> MLXArray {
        var x = xIn
        x = x + attn(preNorm(x))
        x = x + ff(ffNorm(x))
        return x
    }
}

// MARK: - SAME-S decoder

/// ~90 M params, FP32. Input [B, 256, T_lat] → output [B, 512, T_lat*16].
///
/// Structure mirrors upstream `same_s_decoder.py`:
/// - one learnable `new_tokens` row broadcast to the 16 non-latent positions
///   of each 17-token group;
/// - **blocks 0–2** run over non-overlapping `EFFECTIVE_CHUNK = 34` windows;
/// - **blocks 3–5** run over the same sequence shifted by `SHIFT = 17` on
///   both ends (chunk-midpoint-shift), then crop back — this replaces the
///   sliding-window mask SAME-L uses;
/// - `mapping` is a k=3, pad=1 Conv1d (weight permuted from the PyTorch
///   `(out, in, k)` layout at load time).
///
/// Invariant: the internal length `T_lat × 17` must be divisible by 34 —
/// i.e. every call must see an **even** `T_lat`. `StableAudio3MusicGen`
/// rounds the requested latent count up to even for this family, and the
/// chunked path only ever passes full (even) kernels.
public final class SAMESDecoder: Module {
    @ParameterInfo(key: "running_std") public var runningStd: MLXArray   // (1,)
    @ModuleInfo(key: "project_in") public var projectIn: Linear
    @ParameterInfo(key: "new_tokens") public var newTokens: MLXArray     // (1, 1, dim)
    @ModuleInfo public var blocks: [TransformerBlockS]
    @ModuleInfo public var mapping: Conv1d   // k=3, padding=1

    public override init() {
        self._runningStd.wrappedValue = MLXArray([Float(1.0)])
        self._projectIn.wrappedValue = Linear(SAMESDims.latentDim, SAMESDims.dim, bias: true)
        self._newTokens.wrappedValue = MLXArray.zeros([1, 1, SAMESDims.dim])
        self._blocks.wrappedValue = (0..<SAMESDims.numBlocks).map { _ in TransformerBlockS() }
        self._mapping.wrappedValue = Conv1d(
            inputChannels: SAMESDims.dim, outputChannels: SAMESDims.outChannels,
            kernelSize: 3, padding: 1, bias: true)
        super.init()
    }

    /// `fullAttention` exists for protocol parity with SAME-L; SAME-S always
    /// attends over the window the caller supplies, so it is ignored.
    public func callAsFunction(_ latents: MLXArray, fullAttention: Bool = false) -> MLXArray {
        let B = latents.dim(0)
        let tLat = latents.dim(2)
        precondition(tLat % 2 == 0,
                     "SAME-S decode requires an even latent length (got \(tLat)): "
                   + "the internal T×17 must align to 34. StableAudio3MusicGen rounds "
                   + "small-family latents up to even before reaching the decoder.")

        // Softnorm bottleneck decode (scalar) — runningStd is shape [1].
        var x = latents * runningStd.asType(latents.dtype)

        // [B, 256, T_lat] → [B, T_lat, 256] → [B, T_lat, DIM]
        x = projectIn(x.transposed(0, 2, 1))

        // Single new_token broadcast to 16 positions per latent slot, with the
        // original latent at position 0 of each 17-group.
        let xE = x.expandedDimensions(axis: 2)                                   // [B, T_lat, 1, DIM]
        let nt = MLX.broadcast(
            newTokens.expandedDimensions(axis: 0).asType(x.dtype),
            to: [B, tLat, SAMESDims.sinPerPos, SAMESDims.dim])
        x = MLX.concatenated([xE, nt], axis: 2)                                  // [B, T_lat, 17, DIM]
        x = x.reshaped([B, tLat * SAMESDims.subChunkSize, SAMESDims.dim])

        let internalT = tLat * SAMESDims.subChunkSize        // T_lat × 17 (divisible by 34)

        // First half — blocks 0…2 over non-overlapping 34-token chunks.
        let nc1 = internalT / SAMESDims.effectiveChunk
        var h = x.reshaped([B * nc1, SAMESDims.effectiveChunk, SAMESDims.dim])
        h = blocks[0](h)
        h = blocks[1](h)
        h = blocks[2](h)
        h = h.reshaped([B, internalT, SAMESDims.dim])

        // Shift by 17 on both ends → second half — blocks 3…5, then crop back.
        let left  = h[0..., 0..<SAMESDims.shift, 0...]
        let right = h[0..., (internalT - SAMESDims.shift)..<internalT, 0...]
        h = MLX.concatenated([left, h, right], axis: 1)       // [B, internalT+34, DIM]
        let nc2 = (internalT + SAMESDims.effectiveChunk) / SAMESDims.effectiveChunk
        h = h.reshaped([B * nc2, SAMESDims.effectiveChunk, SAMESDims.dim])
        h = blocks[3](h)
        h = blocks[4](h)
        h = blocks[5](h)
        h = h.reshaped([B, internalT + SAMESDims.effectiveChunk, SAMESDims.dim])
        h = h[0..., SAMESDims.shift..<(internalT + SAMESDims.shift), 0...]

        // Drop the original latent slot at index 0 of each 17-group; keep 16.
        h = h.reshaped([B * tLat, SAMESDims.subChunkSize, SAMESDims.dim])
        h = h[0..., 1..., 0...]
        h = h.reshaped([B, tLat * SAMESDims.sinPerPos, SAMESDims.dim])

        // MLX Conv1d expects [B, T, C] input → [B, T_lat*16, 512] → [B, 512, T_lat*16]
        return mapping(h).transposed(0, 2, 1)
    }
}

// MARK: - Chunked decode for long sequences

/// Uniform-kernel chunked decode for SAME-S. Mirrors `decode_chunked` in
/// upstream `same_s_decoder.py` — no zero-padding, three segments
/// (first/interior/last) each see `kernel = chunk + 2*overlap` real latents.
///
/// Every window handed to the model has length `kernel`, so the decoder's
/// even-`T_lat` invariant reduces to: **kernel even** (the assertion below)
/// plus an even total length on the un-chunked fallback.
public func sameSDecodeChunked(_ model: SA3AudioDecoder, latents: MLXArray,
                               chunkSize: Int, overlap: Int) -> MLXArray {
    let T = latents.dim(2)
    let kernel = chunkSize + 2 * overlap
    precondition(kernel % 2 == 0,
                 "SAME-S needs an even kernel (chunk + 2*overlap); got \(kernel)")
    if T <= kernel {
        precondition(T % 2 == 0,
                     "SAME-S un-chunked decode requires an even latent length "
                   + "(got \(T)) and T ≤ kernel (\(kernel)) prevents the chunked path.")
        return model.callAsFunction(latents, fullAttention: false)
    }

    var pieces: [MLXArray] = []

    // 1) First decode covers output positions [0, chunk + overlap)
    let firstOut = model.callAsFunction(latents[0..., 0..., 0..<kernel], fullAttention: false)
    let validFirst = chunkSize + overlap
    pieces.append(firstOut[0..., 0..., 0..<(validFirst * SAMESDims.sinPerPos)])
    var i = validFirst

    // 2) Interior: stride by chunk
    while i + chunkSize + overlap <= T {
        let lo = i - overlap
        let hi = i + chunkSize + overlap
        let out = model.callAsFunction(latents[0..., 0..., lo..<hi], fullAttention: false)
        let pieceLo = overlap * SAMESDims.sinPerPos
        let pieceHi = (overlap + chunkSize) * SAMESDims.sinPerPos
        pieces.append(out[0..., 0..., pieceLo..<pieceHi])
        i += chunkSize
    }

    // 3) Last decode covers remaining (T - i) output positions
    let remaining = T - i
    if remaining > 0 {
        let lastOut = model.callAsFunction(latents[0..., 0..., (T - kernel)..<T], fullAttention: false)
        let tail = remaining * SAMESDims.sinPerPos
        let total = lastOut.dim(2)
        pieces.append(lastOut[0..., 0..., (total - tail)..<total])
    }
    return MLX.concatenated(pieces, axis: -1)
}

extension SAMESDecoder: SA3AudioDecoder {}
