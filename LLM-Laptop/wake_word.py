import sounddevice as sd
import sherpa_onnx
import numpy as np

MODEL_DIR = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"

SAMPLE_RATE = 16000
CHUNK_SIZE = 1600
MIC_DEVICE = 0

print("Loading MOMO wake-word detector...")

kws = sherpa_onnx.KeywordSpotter(
    tokens=f"{MODEL_DIR}/tokens.txt",
    encoder=f"{MODEL_DIR}/encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    decoder=f"{MODEL_DIR}/decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    joiner=f"{MODEL_DIR}/joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    keywords_file=f"{MODEL_DIR}/momo_keywords.txt",
    num_threads=2,
    sample_rate=SAMPLE_RATE,
    feature_dim=80,
    provider="cpu",
    keywords_score=1.0,
    keywords_threshold=0.25,
)

stream = kws.create_stream()

print("\nMOMO wake-word detector is ready!")
print("Say: Hey MOMO")
print("Press Ctrl+C to stop.\n")

try:
    with sd.InputStream(
        device=MIC_DEVICE,
        channels=1,
        dtype="float32",
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
    ) as mic:

        while True:
            audio, overflowed = mic.read(CHUNK_SIZE)

            if overflowed:
                print("Warning: microphone buffer overflow")

            audio = audio.reshape(-1)

            stream.accept_waveform(SAMPLE_RATE, audio)

            while kws.is_ready(stream):
                kws.decode_stream(stream)
                result = kws.get_result(stream)

                if result:
                    print(f"\n[WAKE WORD DETECTED] {result}")
                    print("MOMO is listening again...\n")
                    kws.reset_stream(stream)

except KeyboardInterrupt:
    print("\nWake-word detector stopped.")

except Exception as e:
    print(f"\nError: {e}")
    raise
