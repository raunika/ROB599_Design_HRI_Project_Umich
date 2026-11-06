# MOMO – LLM and Conversational Pipeline

## 1. System Architecture

The pipeline consists of four main stages:

1. **Speech-to-Text (STT):** Whisper.cpp converts the student's speech into text.
2. **Language Model (LLM):** Qwen2.5 0.5B Instruct generates a conversational response using llama.cpp.
3. **Text-to-Speech (TTS):** Piper converts the generated response into speech.
4. **Audio Playback:** The generated speech is played through the connected audio output device.

## 2. Hardware and Models

- **Hardware:** NVIDIA Jetson TX2 (4 GB RAM).
- **STT:** Whisper.cpp, using `ggml-tiny.en.bin`.
- **LLM:** Qwen2.5 0.5B Instruct, Q4_K_M GGUF quantization.
- **TTS:** Piper, using `en_US-hfc_female-medium.onnx` (HFC Female Medium voice; can be changed).

**Note:** Model files are not included in this repository and must be downloaded separately.

## 3. Repository Structure

```text
LLM/
├── README.md
├── requirements.txt
├── pipeline/
│   ├── momo_live.py
│   ├── momo_pipeline.py
│   └── momo_system_prompt.txt
└── src/
    ├── whisper.cpp/
    └── llama.cpp/
```

## 4. Getting Started

1. **Clone or download the repository** onto the Jetson TX2.

2. **Install the required dependencies:** Python 3, CMake, a C/C++ compiler, CUDA toolkit, and PulseAudio utilities (`parecord` and `paplay`). The native Whisper.cpp and llama.cpp components must be built for the Jetson environment.

3. **Download the required models:**
   - Whisper: `ggml-tiny.en.bin`
   - LLM: `qwen2.5-0.5b-instruct-q4_k_m.gguf`
   - Piper: `en_US-hfc_female-medium.onnx` and its corresponding JSON configuration.

4. **Configure the paths:** Update the executable and model paths in `pipeline/momo_pipeline.py` to match their locations on the Jetson. Also, configure the microphone source in `pipeline/momo_live.py`.

5. **Install Python dependencies:** The current scripts use Python standard-library modules only, so no additional pip packages are required.

6. **Run MOMO:**

   ```bash
   python3 pipeline/momo_live.py
   ```

## 5. Source Repositories and Modifications

### Whisper.cpp
- Upstream repository: https://github.com/ggml-org/whisper.cpp
- Commit: `1d716d6e`

### llama.cpp
- Upstream repository: https://github.com/ggml-org/llama.cpp
- Commit: `7a221b67`
- Includes Jetson-specific CUDA and ARM compatibility modifications to the Makefile and several GGML source files.

**Note:** The included source directories preserve the versions and modifications used in the working Jetson setup. Python scripts currently contain absolute paths under /nvmeroot/momo/

## Python Environment

- Python version: 3.6.9
- No third-party Python packages are required; the scripts use Python's standard library.
