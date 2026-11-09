# MOMO – LLM (Laptop)

Local LLM-based conversational subsystem for MOMO, an embodied social robot tutor.

This implementation runs locally on an Apple Silicon MacBook and supports two interaction modes:

1. **Hybrid Mode:** Activate MOMO using the wake phrase "Hey MOMO" or by pressing Enter.
2. **Manual Mode:** Press Enter to initiate each interaction.

## System Architecture

The conversational pipeline consists of:

1. **Wake-word Detection:** Sherpa-ONNX continuously listens for "Hey MOMO" in Hybrid Mode.
2. **Voice Activity Detection:** WebRTC VAD detects the beginning and end of speech.
3. **Speech-to-Text:** OpenAI Whisper converts the student's speech into text.
4. **Language Model:** Qwen 3.5 9B Q8 generates a response and associated gesture IDs through Ollama.
5. **Gesture Handler:** Processes the gesture IDs returned by the LLM.
6. **Text-to-Speech:** Kokoro generates MOMO's spoken response.

All inference runs locally. No cloud-based LLM API is required.

## Directory Structure

```text
LLM-Laptop/
│
├── momo_voice.py
├── momo_display.py
├── wake_word.py
├── Modelfile.qwen
├── requirements.txt
├── README.md
│
└── sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/
    ├── tokens.txt
    ├── bpe.model
    ├── momo_keywords.txt
    ├── encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx
    ├── decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx
    └── joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx
```

### File Descriptions

| File | Description |
|---|---|
| `momo_voice.py` | Main conversational pipeline integrating STT, LLM, gestures and TTS. |
| `momo_display.py` | On-screen UI (face, talk button, captions) for the same pipeline. |
| `wake_word.py` | Standalone wake-word detection script. |
| `Modelfile.qwen` | Ollama Modelfile used to configure MOMO's Qwen model. |
| `requirements.txt` | Python dependencies and pinned versions. |
| `tokens.txt` | Token vocabulary used by the wake-word model. |
| `bpe.model` | Byte Pair Encoding model. |
| `momo_keywords.txt` | Tokenized custom wake phrase. |
| `encoder...onnx` | INT8 encoder model. |
| `decoder...onnx` | INT8 decoder model. |
| `joiner...onnx` | INT8 joiner model. |

## Models and Versions

| Component | Model / Package | Version / Configuration |
|---|---|---|
| LLM | Qwen 3.5 9B | `qwen3.5:9b-q8_0` |
| Ollama model name | MOMO | `momo-q8` |
| Speech-to-Text | OpenAI Whisper | `base.en` |
| Whisper package | `openai-whisper` | `20250625` |
| Text-to-Speech | Kokoro MLX | `kokoro-mlx 0.1.4` |
| TTS voice | American Female Heart | `af_heart` |
| Wake-word detection | Sherpa-ONNX Zipformer GigaSpeech | `3.3M`, 2024-01-01 |
| Sherpa-ONNX | `sherpa-onnx` | `2.0.1` |
| Voice Activity Detection | WebRTC VAD | `2.0.10` |

## Requirements

- Apple Silicon Mac
- Python 3.12 (tested)
- Ollama installed and running
- Microphone and speakers
- Required Python packages listed in `requirements.txt`

## Installation

Clone the repository and navigate to this directory:

```bash
cd LLM-Laptop
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Ollama Setup

Install Ollama and download the base model:

```bash
ollama pull qwen3.5:9b-q8_0
```

Create the MOMO model using the supplied Modelfile:

```bash
ollama create momo-q8 -f Modelfile.qwen
```

Start Ollama if it is not already running:

```bash
ollama serve
```

## Running MOMO

Run the main conversational program:

```bash
python momo_voice.py
```

Choose an interaction mode at startup:

- Enter `1` for Hybrid Mode (Hey MOMO + Enter).
- Enter `2` for Manual Mode (Enter only).

Say "goodbye" to end the conversation.

### Running MOMO with the Display

`momo_display.py` runs the same conversation as `momo_voice.py` with MOMO's on-screen face, a tap-to-talk button and captions timed to MOMO's voice. It reuses the settings and functions in `momo_voice.py`, so nothing needs to be configured twice. Tkinter is required: python.org Python includes it; with Homebrew Python run `brew install python-tk@3.12`.

```bash
python momo_display.py              # fullscreen, "Hey MOMO" or tap (hybrid)
python momo_display.py --windowed   # 600x1024 window for testing
python momo_display.py --manual     # tap only, no wake word
python momo_display.py --demo       # UI only, fake replies, no models needed
```

Tap the button (or press Space) to talk; recording stops automatically when you pause. Toggle captions with the CC button (or `C`) and press Esc to quit. Saying "goodbye" ends the conversation and closes the window. Run either `momo_display.py` or `momo_voice.py`, not both, since they share the microphone.

### Standalone Wake-word Test

The wake-word detector can also be tested independently:

```bash
python wake_word.py
```

Say "Hey MOMO" to test detection.

## Notes

- The wake-word detector uses the INT8 ONNX models to reduce model size and computational requirements.
- The main program releases the wake-word microphone stream before starting speech recording.
- The Qwen model is managed separately by Ollama and is not included in this repository.
- Whisper model weights are downloaded by Whisper when first initialized.
- Kokoro model assets are managed by the Kokoro MLX package.
- The standalone wake-word model files are included in this repository.
- Physical robot gesture execution is not integrated yet; gesture IDs are currently generated and printed by the conversational pipeline.
