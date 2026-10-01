import Foundation
import MLX
import MLXNN
import MLXRandom
import AudioCommon

// MARK: - Generation parameters

public struct StableAudio3GenerationParams: Sendable {
    public var seconds: Float
    public var steps: Int
    public var cfgScale: Float
    public var apg: Float
    public var sigmaMax: Float
    public var seed: UInt64?

    public init(
        seconds: Float = 30.0,
        steps: Int = 8,
        cfgScale: Float = 1.0,
        apg: Float = 1.0,
        sigmaMax: Float = 1.0,
        seed: UInt64? = nil
    ) {
        self.seconds = seconds
        self.steps = steps
        self.cfgScale = cfgScale
        self.apg = apg
        self.sigmaMax = sigmaMax
        self.seed = seed
    }
}

// MARK: - Denoiser / decoder protocols (family-erased storage)

/// The DiT denoiser contract every family implements: velocity prediction
/// from (latents, sigma, cross-attn cond, global cond).
public protocol SA3Denoiser: AnyObject {
    func callAsFunction(_ x: MLXArray, t: MLXArray,
                        crossAttnCondRaw: MLXArray, globalCondRaw: MLXArray,
                        localAddCond: MLXArray?) -> MLXArray
}

/// The latent→patch decoder contract every SAME family implements.
/// `fullAttention` is meaningful for SAME-L only (SAME-S ignores it — its
/// attention window is whatever the caller reshaped to).
public protocol SA3AudioDecoder: AnyObject {
    func callAsFunction(_ latents: MLXArray, fullAttention: Bool) -> MLXArray
}

// MARK: - Main entry point

public final class StableAudio3MusicGen {
    public let variant: StableAudio3Variant
    public let t5: T5GemmaText
    public let dit: any SA3Denoiser            // DiT-Medium or DiT-Small
    public let decoder: any SA3AudioDecoder    // SAME-L or SAME-S
    public let padding: MLXArray              // [768]
    public let secondsEmbedder: SecondsTotalEmbedder

    private init(variant: StableAudio3Variant,
                 t5: T5GemmaText, dit: any SA3Denoiser, decoder: any SA3AudioDecoder,
                 padding: MLXArray, secondsEmbedder: SecondsTotalEmbedder) {
        self.variant = variant
        self.t5 = t5
        self.dit = dit
        self.decoder = decoder
        self.padding = padding
        self.secondsEmbedder = secondsEmbedder
    }

    // MARK: - Loading

    /// Download and load a published SA3 bundle. The default variant is
    /// `.mediumInt8` — corresponding to `aufklarer/Stable-Audio-3-DiT-Medium-MLX-8bit`.
    ///
    /// Family routing: `.medium*` → DiT-Medium + SAME-L;
    /// `.smallMusic*` / `.smallSFX*` → DiT-Small + SAME-S.
    public static func fromPretrained(
        variant: StableAudio3Variant = .mediumInt8,
        tLatHint: Int? = nil,
        localBundleOverride: URL? = nil,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws -> StableAudio3MusicGen {
        let paths = try await StableAudio3Downloader.ensureDownloaded(
            variant: variant,
            localBundleOverride: localBundleOverride,
            progressHandler: progressHandler)

        let tLat = tLatHint ?? computeTLat(seconds: 30.0)
        let ditModel: any SA3Denoiser
        let decoder: any SA3AudioDecoder
        let padding: MLXArray
        let secsEmb: SecondsTotalEmbedder
        switch variant.family {
        case .medium:
            let (model, pad, secs) = try sa3LoadDiTMedium(
                dir: paths.dit, tLat: tLat, bits: variant.bits)
            ditModel = model
            decoder = try sa3LoadSAMELDecoder(dir: paths.sameDecoder)
            padding = pad
            secsEmb = secs
        case .smallMusic, .smallSFX:
            let (model, pad, secs) = try sa3LoadDiTSmall(
                dir: paths.dit, tLat: tLat, bits: variant.bits)
            ditModel = model
            decoder = try sa3LoadSAMESDecoder(dir: paths.sameDecoder)
            padding = pad
            secsEmb = secs
        }
        let t5 = try sa3LoadT5Gemma(dir: paths.t5gemma)

        return StableAudio3MusicGen(
            variant: variant, t5: t5, dit: ditModel, decoder: decoder,
            padding: padding, secondsEmbedder: secsEmb)
    }

    /// Compute the latent timesteps required to render `seconds` of audio.
    /// `ceil(seconds * 44100 / 4096)`.
    public static func computeTLat(seconds: Float) -> Int {
        let samples = seconds * Float(SA3Audio.sampleRate)
        return max(1, Int((samples / Float(SA3Audio.samplesPerLatent)).rounded(.up)))
    }

    // MARK: - Generation

    /// Generate a stereo waveform from a text prompt. Returns interleaved
    /// stereo Float32 PCM at 44.1 kHz with exactly `floor(seconds * 44100)`
    /// frames per channel.
    public func generate(
        prompt: String,
        params: StableAudio3GenerationParams = StableAudio3GenerationParams()
    ) -> (left: [Float], right: [Float]) {
        if let seed = params.seed { MLXRandom.seed(seed) }
        let dtype: DType = .float16

        // 1) Text encoding
        let (embeds, mask) = t5.encode(prompt)
        let embedsF16 = embeds.asType(dtype)
        let embedsPadded = applyPromptPadding(
            embeds: embedsF16,
            mask: mask,
            paddingEmbedding: padding.asType(dtype))
        let secondsEmbed = secondsEmbedder(params.seconds).asType(dtype)   // [1,1,768]
        let crossAttn = MLX.concatenated([embedsPadded, secondsEmbed], axis: 1)  // [1,257,768]
        let globalCond = secondsEmbed[0..., 0, 0...]                              // [1,768]
        eval(crossAttn, globalCond)

        let nullCrossAttn: MLXArray? = params.cfgScale == 1.0
            ? nil
            : MLXArray.zeros(crossAttn.shape, dtype: dtype)

        // 2) Initial latent
        var tLat = Self.computeTLat(seconds: params.seconds)
        if variant.family != .medium {
            // SAME-S attention reshapes need T_lat × 17 ≡ 0 (mod 34) — i.e. an
            // even latent count. Round up one latent when needed; the output is
            // cropped to the requested length below anyway.
            if tLat % 2 != 0 { tLat += 1 }
        }
        let noise = MLXRandom.normal([1, variant.ioChannels, tLat], dtype: dtype)
        eval(noise)

        // 3) DiT sampling (rectified-flow pingpong)
        let sigmas = buildPingPongSchedule(steps: params.steps, sigmaMax: params.sigmaMax,
                                             useLogSNRShift: true)
        let cfg = params.cfgScale
        let apg = params.apg
        let modelFn: (MLXArray, MLXArray) -> MLXArray = { [unowned self] x, t in
            if cfg == 1.0 {
                return self.dit.callAsFunction(x, t: t, crossAttnCondRaw: crossAttn,
                                               globalCondRaw: globalCond, localAddCond: nil)
            }
            // Batched CFG: cat([x, x]) on the batch dim.
            let x2 = MLX.concatenated([x, x], axis: 0)
            let t2 = MLX.concatenated([t, t], axis: 0)
            let cross2 = MLX.concatenated([crossAttn, nullCrossAttn!], axis: 0)
            let global2 = MLX.concatenated([globalCond, globalCond], axis: 0)
            let vBatched = self.dit.callAsFunction(x2, t: t2, crossAttnCondRaw: cross2,
                                                   globalCondRaw: global2, localAddCond: nil)
            let halves = MLX.split(vBatched, parts: 2, axis: 0)
            let condV = halves[0], uncondV = halves[1]
            let sigma = t.reshaped([-1, 1, 1]).asType(.float32)
            let condD   = x.asType(.float32) - condV.asType(.float32)   * sigma
            let uncondD = x.asType(.float32) - uncondV.asType(.float32) * sigma
            let diff = condD - uncondD

            let cfgDiff: MLXArray
            if apg <= 0.0 {
                cfgDiff = diff
            } else {
                let norm = MLX.sqrt((condD * condD).sum(axes: [-2, -1], keepDims: true))
                let unit = condD / MLX.maximum(norm, MLXArray(Float(1e-8)))
                let parallel = (diff * unit).sum(axes: [-2, -1], keepDims: true) * unit
                let diffOrth = diff - parallel
                if apg >= 1.0 {
                    cfgDiff = diffOrth
                } else {
                    cfgDiff = MLXArray(apg) * diffOrth + MLXArray(1.0 - apg) * diff
                }
            }
            let cfgD = condD + MLXArray(cfg - 1.0) * cfgDiff
            let cfgV = (x.asType(.float32) - cfgD) / sigma
            return cfgV.asType(x.dtype)
        }

        let seed = params.seed ?? UInt64.random(in: 0..<UInt64(UInt32.max))
        let latents = sampleFlowPingPong(modelFn: modelFn, initial: noise, sigmas: sigmas,
                                          seed: seed &+ 1, onStep: nil)
        eval(latents)

        // 4) Decode latents → audio patches
        let latentsFP32 = latents.asType(.float32)
        let kernel = 128 + 2 * 8   // chunked decode defaults from upstream
        let patches: MLXArray
        if tLat > kernel {
            patches = variant.family == .medium
                ? sameLDecodeChunked(decoder, latents: latentsFP32, chunkSize: 128, overlap: 8)
                : sameSDecodeChunked(decoder, latents: latentsFP32, chunkSize: 128, overlap: 8)
        } else {
            patches = decoder.callAsFunction(latentsFP32, fullAttention: false)
        }
        eval(patches)

        // 5) Unpatch → audio + crop to requested length
        let audio = patchedDecode(patches, patchSize: SA3Audio.patchSize, channels: SA3Audio.channels)
        let audioF32 = audio.asType(.float32)
        eval(audioF32)
        let requestedSamples = Int((params.seconds * Float(SA3Audio.sampleRate)).rounded(.down))
        let trimmed = audioF32[0..., 0..., 0..<min(audioF32.dim(2), requestedSamples)]
        let left = trimmed[0, 0, 0...]
        let right = trimmed[0, 1, 0...]
        return (left: left.asArray(Float.self), right: right.asArray(Float.self))
    }
}
