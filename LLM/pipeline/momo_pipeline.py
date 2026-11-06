import subprocess
import re
import sys
import os


# ============================================================
# WHISPER CONFIGURATION (Speech-to-Text)
# ============================================================

WHISPER = "/nvmeroot/momo/src/whisper.cpp/build/bin/main"
WHISPER_MODEL = "/nvmeroot/momo/src/whisper.cpp/models/ggml-tiny.en.bin"


# ============================================================
# QWEN CONFIGURATION (Language Model)
# ============================================================

LLAMA = "/nvmeroot/momo/src/llama.cpp-qwen-test/llama-cli"
LLM_MODEL = "/nvmeroot/momo/models/llm/qwen2.5-0.5b-instruct-q4_k_m.gguf"


# ============================================================
# PIPER CONFIGURATION (Text-to-Speech)
# ============================================================

PIPER_DIR = "/nvmeroot/momo/tts/piper/piper"
PIPER = os.path.join(PIPER_DIR, "piper")

# Selected voice: HFC Female Medium
PIPER_MODEL = os.path.join(
    PIPER_DIR,
    "en_US-hfc_female-medium.onnx"
)

TTS_OUTPUT = "/nvmeroot/momo/tts/momo_response.wav"


# ============================================================
# SPEECH-TO-TEXT
# ============================================================

def transcribe_audio(audio_path):

    print("\n[MOMO] Transcribing audio...")

    result = subprocess.run(
        [
            WHISPER,
            "-m", WHISPER_MODEL,
            "-f", audio_path
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )

    if result.returncode != 0:
        raise RuntimeError("Whisper failed:\n" + result.stdout)

    transcript_lines = []

    for line in result.stdout.splitlines():

        line = line.strip()

        if re.match(r"\[\d{2}:\d{2}:\d{2}\.\d{3} -->", line):

            text = re.sub(r"^\[.*?-->.*?\]\s*", "", line)

            if text:
                transcript_lines.append(text)

    transcript = " ".join(transcript_lines)

    if not transcript:
        raise RuntimeError(
            "No transcript found. Whisper output:\n" + result.stdout
        )

    print("[Student]:", transcript)

    return transcript


# ============================================================
# LANGUAGE MODEL: QWEN
# ============================================================

def generate_response(transcript):

    print("\n[MOMO] Thinking...")

    with open("/nvmeroot/momo/pipeline/momo_system_prompt.txt", "r") as f:
        system_prompt = f.read().strip()

    prompt = (
        "<|im_start|>system\n"
        + system_prompt
        + "<|im_end|>\n"
        "<|im_start|>user\n"
        + transcript
        + "\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    result = subprocess.run(
        [
            LLAMA,
            "-m", LLM_MODEL,
            "-ngl", "999",
            "-t", "4",
            "-n", "220",
            "--temp", "0.7",
            "--repeat-penalty", "1.1",
            "--repeat-last-n", "64",
            "-p", prompt
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )

    if result.returncode != 0:
        raise RuntimeError("Qwen failed:\n" + result.stdout)

    output = result.stdout

    # Extract the generated answer after the final assistant heading.
    if "assistant\n" in output:
        response = output.rsplit("assistant\n", 1)[-1]
    else:
        response = output.rsplit("MOMO:", 1)[-1]

    # Remove llama.cpp logs and generation markers.
    response = response.split("llama_print_timings:")[0]
    response = response.split("Log end")[0]

    response = response.replace("[BLANK_AUDIO]", "")
    response = response.replace("[end of text]", "")
    response = response.strip()

    if not response:
        raise RuntimeError(
            "Could not extract Qwen response. Raw output:\n" + output
        )

    print("\n[MOMO]:", response)

    return response


# ============================================================
# TEXT-TO-SPEECH: PIPER
# ============================================================

def synthesize_speech(text, output_path=TTS_OUTPUT):

    print("\n[MOMO] Generating speech...")

    # Piper needs its library directory in LD_LIBRARY_PATH.
    env = os.environ.copy()

    existing_library_path = env.get("LD_LIBRARY_PATH", "")

    env["LD_LIBRARY_PATH"] = (
        PIPER_DIR + ":" + existing_library_path
    )

    result = subprocess.run(
        [
            PIPER,
            "--model", PIPER_MODEL,
            "--output_file", output_path
        ],
        input=text,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        env=env
    )

    if result.returncode != 0:
        raise RuntimeError("Piper failed:\n" + result.stdout)

    print("[MOMO] Speech saved to:", output_path)

    return output_path


# ============================================================
# COMPLETE MOMO PIPELINE
# ============================================================

def run_pipeline(audio_path):

    # Step 1: Convert student's speech into text.
    transcript = transcribe_audio(audio_path)

    # Step 2: Generate MOMO's response using Qwen.
    response = generate_response(transcript)

    # Step 3: Convert MOMO's response into speech using Piper.
    audio_output = synthesize_speech(response)

    print("\n[MOMO] Pipeline completed successfully.")
    print("[MOMO] Audio output:", audio_output)

    return response


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print("Usage: python3 momo_pipeline.py path/to/audio.wav")
        sys.exit(1)

    run_pipeline(sys.argv[1])
