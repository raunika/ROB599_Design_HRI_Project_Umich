import json
import os
import subprocess
import urllib.request
import wave
import io
import warnings
import sys
import select

from collections import deque
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import webrtcvad
import sounddevice as sd
import sherpa_onnx
from kokoro_mlx import KokoroTTS


# ============================================================
# CONFIGURATION
# ============================================================

os.environ["PATH"] = (
    "/opt/homebrew/bin:/opt/homebrew/sbin:"
    + os.environ.get("PATH", "")
)

BASE_DIR = Path(__file__).resolve().parent

WHISPER_PYTHON = sys.executable
AUDIO_FILE = str(BASE_DIR / "momo_recording.wav")

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)

SILENCE_SECONDS = 1.0
MAX_RECORD_SECONDS = 30
VAD_AGGRESSIVENESS = 2

MODEL = "momo-q8"
OLLAMA_URL = "http://localhost:11434/api/chat"

VOICE = "af_heart"

# Wake-word configuration
WAKE_MODEL_DIR = (
    BASE_DIR /
    "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
)

WAKE_CHUNK_SIZE = 1600
WAKE_MIC_DEVICE = 0


# Suppress the specific PyTorch deprecation warning.
warnings.filterwarnings(
    "ignore",
    message=".*torch.jit.script.*deprecated.*",
    category=FutureWarning
)


# ============================================================
# GESTURE MAPPING
# ============================================================

GESTURES = {
    1: "Wave",
    2: "Blink",
    3: "Nod",
    4: "Head tilt left",
    5: "Head tilt right",
    6: "Both arms up",
    7: "Delayed two-arm raise",
    8: "Right arm raise",
    9: "Left arm raise",
    10: "Head shake",
}


# Conversation history.
# The system prompt is stored in the Ollama model.
messages = []


# ============================================================
# INTERACTION MODE
# ============================================================

def select_interaction_mode():

    print("\n===================================")
    print("       MOMO INTERACTION MODE")
    print("===================================")
    print("1. Hybrid mode  (Hey MOMO + Enter)")
    print("2. Manual mode  (Enter only)")
    print("===================================")

    while True:

        choice = input("\nSelect mode (1 or 2): ").strip()

        if choice in {"1", "2"}:
            return choice

        print("Invalid choice. Please enter 1 or 2.")


# ============================================================
# WAKE-WORD DETECTOR
# ============================================================

def create_wake_word_detector():

    print("\nLoading MOMO wake-word detector...")

    kws = sherpa_onnx.KeywordSpotter(

        tokens=str(WAKE_MODEL_DIR / "tokens.txt"),

        encoder=str(
            WAKE_MODEL_DIR /
            "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
        ),

        decoder=str(
            WAKE_MODEL_DIR /
            "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
        ),

        joiner=str(
            WAKE_MODEL_DIR /
            "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
        ),

        keywords_file=str(WAKE_MODEL_DIR / "momo_keywords.txt"),

        num_threads=2,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        provider="cpu",
        keywords_score=1.0,
        keywords_threshold=0.25,
    )

    print("Wake-word detector loaded.")

    return kws


def wait_for_hybrid_activation(kws):

    print("\nListening... Say 'Hey MOMO' or press Enter.")

    # A fresh stream is created for each activation cycle.
    # This allows the microphone to be released before recording.
    stream = kws.create_stream()

    with sd.InputStream(
        device=WAKE_MIC_DEVICE,
        channels=1,
        dtype="float32",
        samplerate=SAMPLE_RATE,
        blocksize=WAKE_CHUNK_SIZE,
    ) as mic:

        while True:

            # Check whether Enter was pressed.
            ready, _, _ = select.select([sys.stdin], [], [], 0)

            if ready:

                sys.stdin.readline()

                print("\n[ENTER PRESSED] MOMO is listening...")

                return

            # Read microphone audio for wake-word detection.
            audio, overflowed = mic.read(WAKE_CHUNK_SIZE)

            if overflowed:
                print("Warning: microphone buffer overflow")

            audio = audio.reshape(-1)

            stream.accept_waveform(SAMPLE_RATE, audio)

            while kws.is_ready(stream):

                kws.decode_stream(stream)

                result = kws.get_result(stream)

                if result:

                    print(f"\n[WAKE WORD DETECTED] {result}")
                    print("MOMO is listening...")

                    return


# ============================================================
# AUDIO RECORDING
# ============================================================

def record_audio():

    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

    pre_roll = deque(maxlen=10)
    frames = []

    started = False
    silence_frames = 0

    silence_limit = int(SILENCE_SECONDS * 1000 / FRAME_MS)
    max_frames = int(MAX_RECORD_SECONDS * 1000 / FRAME_MS)

    print("Speak now...")

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SAMPLES,
        channels=1,
        dtype="int16",
    ) as stream:

        for _ in range(max_frames):

            frame, overflowed = stream.read(FRAME_SAMPLES)

            is_speech = vad.is_speech(frame, SAMPLE_RATE)

            if not started:

                pre_roll.append(frame)

                if is_speech:
                    started = True
                    frames.extend(pre_roll)

            else:

                frames.append(frame)

                if is_speech:
                    silence_frames = 0
                else:
                    silence_frames += 1

                if silence_frames >= silence_limit:
                    break

    if not started:

        print("No speech detected. Try again.")

        with wave.open(AUDIO_FILE, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(b"")

        return False

    audio_bytes = b"".join(frames)

    with wave.open(AUDIO_FILE, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(audio_bytes)

    return True


# ============================================================
# SPEECH-TO-TEXT
# ============================================================

def transcribe_audio():

    result = subprocess.run(
        [
            WHISPER_PYTHON,
            "-c",
            (
                "import whisper, sys; "
                "model = whisper.load_model('base.en'); "
                "result = model.transcribe(sys.argv[1]); "
                "print(result['text'].strip())"
            ),
            AUDIO_FILE,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip() or "Whisper failed."
        )

    return result.stdout.strip()


# ============================================================
# GESTURE HANDLER
# ============================================================

def execute_gestures(gesture_ids):

    if not isinstance(gesture_ids, list):
        return []

    valid_gestures = []

    for gesture_id in gesture_ids:

        if (
            isinstance(gesture_id, int)
            and not isinstance(gesture_id, bool)
            and gesture_id in GESTURES
        ):
            valid_gestures.append(gesture_id)

    return valid_gestures


# ============================================================
# LLM RESPONSE
# ============================================================

def get_momo_response(student_text):

    global messages

    messages.append({
        "role": "user",
        "content": student_text
    })

    payload = json.dumps({

        "model": MODEL,
        "messages": messages,
        "think": False,
        "format": "json",
        "stream": False,

        "options": {
            "temperature": 0.4
        }

    }).encode("utf-8")

    request = urllib.request.Request(

        OLLAMA_URL,
        data=payload,

        headers={
            "Content-Type": "application/json"
        },

        method="POST"

    )

    with urllib.request.urlopen(request, timeout=180) as response:

        data = json.loads(
            response.read().decode("utf-8")
        )

    raw_answer = data["message"]["content"].strip()

    try:

        momo_output = json.loads(raw_answer)

        answer = momo_output["response"]
        gesture_ids = momo_output["gesture"]

        if not isinstance(answer, str):
            raise ValueError("Response field must be a string.")

        if not isinstance(gesture_ids, list):
            raise ValueError("Gesture field must be a list.")

        valid_gestures = execute_gestures(gesture_ids)

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:

        print(f"\nMOMO output error: {error}")
        print("Raw model output:", raw_answer)

        messages.pop()

        return None, []

    messages.append({
        "role": "assistant",
        "content": raw_answer
    })

    return answer.strip(), valid_gestures


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("\nStarting MOMO...")

    mode = select_interaction_mode()

    kws = None

    if mode == "1":
        kws = create_wake_word_detector()

    # Suppress initialization progress output from Kokoro.
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        tts = KokoroTTS.from_pretrained()

    print("\n===================================")
    print("           MOMO IS READY!")
    print("===================================")

    if mode == "1":
        print("Mode: Hybrid (Hey MOMO + Enter)")
    else:
        print("Mode: Manual (Enter only)")

    print("Say 'goodbye' to end the conversation.")

    try:

        while True:

            # Wait for the selected activation method.
            if mode == "1":

                wait_for_hybrid_activation(kws)

            else:

                input("\nPress Enter to speak...")

            # The wake-word microphone stream has closed here.
            # Speech recording can now safely use the microphone.
            recorded = record_audio()

            if not recorded:
                continue

            student_text = transcribe_audio()

            if not student_text:
                print("I didn't catch that. Try again.")
                continue

            print(f"\nYou: {student_text}")

            if student_text.lower().strip(" .!?") in {
                "quit",
                "exit",
                "goodbye"
            }:

                farewell = (
                    "Goodbye! Great learning with you. "
                    "See you next time!"
                )

                print(f"\nMOMO: {farewell}")

                tts.speak(
                    farewell,
                    voice=VOICE,
                    speed=1.0
                )

                break

            answer, gesture_ids = get_momo_response(student_text)

            if answer is None:
                print("MOMO couldn't generate a response. Please try again.")
                continue

            print(f"\nMOMO: {answer}")

            if gesture_ids:
                print(f"Gestures: {gesture_ids}")

            tts.speak(
                answer,
                voice=VOICE,
                speed=1.0
            )

    except KeyboardInterrupt:

        print("\nMOMO: See you next time!")

    except Exception as error:

        print(f"\nSomething went wrong: {error}")
        print("Check that Ollama is running and try again.")


if __name__ == "__main__":
    main()
