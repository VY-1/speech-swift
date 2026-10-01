# Stable Audio 3 Inference

Stable Audio 3 is the default engine for `speech compose`. It generates
44.1 kHz stereo music/audio from a text prompt using MLX.

## CLI

```bash
speech compose "lofi house loop with warm bass" -o loop.wav
```

Useful options:

```bash
speech compose "cinematic trailer drums" \
  --engine sa3 \
  --sa3-variant medium-int8 \
  --seconds 15 \
  --sa3-steps 8 \
  --sa3-cfg 1.0 \
  --seed 42 \
  -o trailer.wav
```

| Flag | Meaning |
|---|---|
| `--engine sa3` | Select Stable Audio 3. This is the default. |
| `--sa3-variant` | `medium-int8` (default) → DiT-Medium + SAME-L; `small-music-int4` → DiT-Small + SAME-S (~50 M DiT, much lighter). |
| `--seconds` | Output duration in seconds. |
| `--sa3-steps` | Ping-pong sampler steps. Default: 8. |
| `--sa3-cfg` | Classifier-free guidance. `1.0` disables CFG. |
| `--sigma-max` | Initial noise level. Default: `1.0`. |
| `--sa3-bundle-dir` | Load a local exported bundle instead of downloading. |
| `--seed` | Reproducible generation seed. |

The command writes a stereo WAV at 44.1 kHz. Use `--engine magnet` when you
want the older MAGNeT path, which writes 32 kHz mono fixed-length clips.

## Swift

```swift
import StableAudio3MusicGen
import AudioCommon

let model = try await StableAudio3MusicGen.fromPretrained(
    variant: .mediumInt8
)

let (left, right) = model.generate(
    prompt: "ambient piano, soft tape hiss, slow evolving pads",
    params: StableAudio3GenerationParams(seconds: 20, seed: 7)
)

try WAVWriter.writeStereo(
    left: left,
    right: right,
    sampleRate: StableAudio3MusicGen.sampleRate,
    to: URL(fileURLWithPath: "ambient.wav")
)
```

## Current Limits

- Both published variants generate: `medium-int8` (DiT-Medium + SAME-L) and
  `small-music-int4` (DiT-Small + SAME-S). SAME-S rounds the requested latent
  count up to an even length — its internal attention reshape requires
  `T_lat × 17 % 34 == 0` — and the output is cropped to the requested length.
- Of the six variants the export pipeline can build, only two were ever
  published: `medium-int8` and `small-music-int4`. The four unpublished ones
  were removed from `StableAudio3Variant` because selecting them failed at
  download time; add a case back when its bundle is uploaded.
- Generation is local and model-heavy.
