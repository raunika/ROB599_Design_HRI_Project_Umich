#!/usr/bin/env python3
"""
MOMO display for the laptop pipeline: portrait UI with YouTube-style captions.

Runs the same conversation as momo_voice.py and shows it on screen:
    "Hey MOMO" or tap -> WebRTC VAD recording -> Whisper base.en
    -> Ollama (momo-q8) -> Kokoro, with captions timed to MOMO's voice.

The screen shows:
    - a title that follows the lesson topic ("Let's learn!" until one starts)
    - a status icon: thinking dots, a voice wave while MOMO talks,
      a pulsing ring while listening (MOMO's eyes are physical, not on screen)
    - a ready-made lesson figure (momo_figures.py) when Qwen picks one

The topic and figure come from the "topic", "subtopic" and "figure" fields that
Modelfile.qwen asks Qwen to add to its JSON reply. momo_voice.py is unchanged:
this file imports its settings and functions, so it stays the one place to
configure the pipeline.

Setup: the same as momo_voice.py (see README.md). Tkinter must be available:
python.org Python includes it; with Homebrew run `brew install python-tk@3.12`.

Run (from LLM-Laptop/, with the venv active and Ollama running):
    python momo_display.py                 fullscreen, "Hey MOMO" or tap
    python momo_display.py --windowed      600x1024 window
    python momo_display.py --manual        tap only, no wake word
    python momo_display.py --demo          UI only, fake speech, no models

Controls:
    Talk button or Space   start listening (recording stops when you pause)
    CC button or C         captions on/off
    Esc                    quit
Say "goodbye" to end the conversation, as in momo_voice.py.
"""

import argparse
import io
import json
import math
import queue
import re
import threading
import time
import urllib.error
from contextlib import redirect_stderr, redirect_stdout

import tkinter as tk
import tkinter.font as tkfont

from momo_figures import FIGURES, draw_figure, figure_title

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_TITLE = "Let's learn!"
DEFAULT_SUBTITLE = "Ask MOMO a question to start"

CAPTION_MAX_LINES = 2  # YouTube shows at most two lines per cue
CAPTION_HOLD_MS = 1500  # how long the last cue stays after MOMO stops
ANIMATION_MS = 40  # status icon frame time (25 fps)
GOODBYE_WORDS = {"quit", "exit", "goodbye"}  # same as momo_voice.py
FAREWELL = "Goodbye! Great learning with you. See you next time!"

# Colors
BG = "#FFF6C7"  # light yellow
INK = "#3D3524"  # warm dark brown: titles, icons
INK_SOFT = "#8A7F62"
CAPTION_BOX = "#403E32"  # YouTube's 75%-black box blended over BG (Tk has no alpha)
CAPTION_TEXT = "#FFFFFF"
CC_RED = "#FF0000"
BUTTON_BUSY = "#E3D7A8"
BUTTON_LISTEN = "#D9473B"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean_for_speech(text):
    """Strip Markdown so captions match what Kokoro says."""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.*?)\*(?!\*)", r"\1", text)
    text = re.sub(r"`+", "", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-+*]\s+", "", text, flags=re.MULTILINE)
    return " ".join(text.split())


def parse_lesson(raw_reply):
    """Read topic, subtopic and figure from Qwen's JSON reply (all optional)."""
    try:
        data = json.loads(raw_reply)
    except (TypeError, ValueError):
        return "", "", None
    if not isinstance(data, dict):
        return "", "", None

    def short(value, limit):
        return " ".join(value.split())[:limit] if isinstance(value, str) else ""

    figure = data.get("figure")
    figure = (
        figure.strip()
        if isinstance(figure, str) and figure.strip() in FIGURES
        else None
    )
    return short(data.get("topic"), 40), short(data.get("subtopic"), 60), figure


def mix(color_a, color_b, amount):
    """Blend two #RRGGBB colors (Tk has no transparency). amount=1 gives color_b."""
    a = [int(color_a[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(color_b[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(
        "{:02X}".format(round(x + (y - x) * amount)) for x, y in zip(a, b)
    )


# ---------------------------------------------------------------------------
# Voice loop (background thread) using momo_voice.py
# ---------------------------------------------------------------------------
def wait_for_wake_or_tap(ui, mv, kws):
    """momo_voice.wait_for_hybrid_activation, with the talk button instead of Enter."""
    import sounddevice as sd

    stream = kws.create_stream()
    with sd.InputStream(
        device=mv.WAKE_MIC_DEVICE,
        channels=1,
        dtype="float32",
        samplerate=mv.SAMPLE_RATE,
        blocksize=mv.WAKE_CHUNK_SIZE,
    ) as mic:
        while not ui.tapped.is_set():
            audio, _ = mic.read(mv.WAKE_CHUNK_SIZE)
            stream.accept_waveform(mv.SAMPLE_RATE, audio.reshape(-1))
            while kws.is_ready(stream):
                kws.decode_stream(stream)
                result = kws.get_result(stream)
                if result:
                    print("\n[WAKE WORD DETECTED]", result)
                    ui.tapped.clear()
                    return
    ui.tapped.clear()
    print("\n[TAPPED] MOMO is listening...")
    # The microphone is released here, before record_audio() opens it again.


def speak(ui, mv, tts, text):
    """Same as tts.speak(), split in two so captions know the audio length."""
    import sounddevice as sd

    ui.events.put(("state", "preparing"))
    result = tts.generate(text, voice=mv.VOICE, speed=1.0)
    ui.events.put(("state", "speaking"))
    ui.events.put(("segment", text, result.duration))
    sd.play(result.audio, samplerate=result.sample_rate)
    sd.wait()


def voice_loop(ui, manual):
    post = ui.events.put

    arduino = None
    servos = None
    iterator = None

    # Initialize MOMO
    try:
        import momo_voice as mv

        kws = None if manual else mv.create_wake_word_detector()

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            tts = mv.KokoroTTS.from_pretrained()

    except Exception as error:
        print("\n[MOMO] Startup error:", repr(error))
        post(
            (
                "error",
                "MOMO couldn't start: {}".format(
                    str(error).splitlines()[0] if str(error) else error
                ),
            )
        )
        return

    # Connect Arduino
    try:
        arduino, servos, iterator = mv.connect_arduino()
        print("[MOMO] Arduino connected successfully.")

    except Exception as error:
        print("\n[ARDUINO ERROR]", repr(error))
        print("Continuing without physical gestures.")

    print(
        "\nMOMO IS READY! Mode:",
        "Manual (tap only)" if manual else "Hybrid (Hey MOMO + tap)",
    )

    # Main conversation loop
    while True:
        try:
            # (1) Wait for wake word or tap
            post(("state", "idle" if manual else "waiting"))

            if manual:
                ui.wait_for_tap()
            else:
                wait_for_wake_or_tap(ui, mv, kws)

            # (2) Record audio
            post(("state", "listening"))

            if not mv.record_audio():
                post(("hint", "I didn't hear anything. Let's try again."))
                continue

            # (3) Speech-to-text
            post(("state", "transcribing"))

            student_text = mv.transcribe_audio()

            if not student_text:
                post(("hint", "I didn't catch that. Let's try again."))
                continue

            print("\nYou:", student_text)

            # (4) Check for goodbye
            if student_text.lower().strip(" .!?") in GOODBYE_WORDS:
                print("\nMOMO:", FAREWELL)
                speak(ui, mv, tts, FAREWELL)
                post(("quit",))
                return

            # (5) Get response from Ollama
            post(("state", "thinking"))

            answer, gesture_ids = mv.get_momo_response(student_text)

            if answer is None:
                post(
                    ("hint", "MOMO couldn't come up with an answer. Please try again.")
                )
                continue

            answer = clean_for_speech(answer)

            topic, subtopic, figure = parse_lesson(mv.messages[-1]["content"])

            print("\nMOMO:", answer)

            if gesture_ids:
                print("Gestures:", gesture_ids)

            print(
                "Topic: {!r}  Subtopic: {!r}  Figure: {}".format(
                    topic, subtopic, figure
                )
            )

            post(("lesson", topic, subtopic, figure))

            # (6) Start Arduino gestures in a separate thread
            gesture_thread = None

            if arduino is not None and gesture_ids:
                gesture_thread = threading.Thread(
                    target=mv.execute_arduino_gestures,
                    args=(servos, gesture_ids),
                    daemon=True,
                )

                gesture_thread.start()

            # (7) MOMO speaks while gestures execute
            try:
                speak(ui, mv, tts, answer)

            finally:
                if gesture_thread is not None:
                    gesture_thread.join()

        except urllib.error.URLError as error:
            print("\n[MOMO] Ollama error:", error)
            post(("hint", "Can't reach Ollama. Check that it's running."))

        except Exception as error:
            print("\n[MOMO] Error:", repr(error))

            first_line = (
                str(error).splitlines()[0] if str(error) else type(error).__name__
            )

            post(("hint", "Something went wrong: {}".format(first_line)))


def demo_loop(ui, manual):
    """Fake conversation for testing the UI without models or audio."""
    post = ui.events.put
    time.sleep(1.0)
    turns = [
        (
            "Hi MOMO! Nice to meet you. What would you like to learn about today?",
            "",
            "",
            None,
        ),
        (
            "Good question. Think about the last time you ran up a flight of stairs. "
            "Your body heated up, but your temperature didn't keep climbing. "
            "What do you think your body did to stop it?",
            "Homeostasis",
            "Keeping body temperature steady",
            "set_point",
        ),
        (
            "Exactly, you started sweating. When sweat evaporates, it carries heat "
            "away from your skin, a bit like a radiator cooling an engine.",
            "Homeostasis",
            "",
            "cooling_down",
        ),
        (
            "Here's the whole loop. Receptors notice the change, the hypothalamus "
            "decides, and the sweat glands act to bring you back to 37 degrees.",
            "Negative Feedback",
            "Receptors, control centre, effectors",
            "feedback_loop",
        ),
    ]
    for reply, topic, subtopic, figure in turns:
        post(("state", "idle" if manual else "waiting"))
        ui.wait_for_tap()
        post(("state", "listening"))
        time.sleep(2.0)  # stands in for VAD stopping on a pause
        for state in ("transcribing", "thinking", "preparing"):
            post(("state", state))
            time.sleep(1.0)
        post(("lesson", topic, subtopic, figure))
        seconds = len(reply.split()) / 2.6
        post(("state", "speaking"))
        post(("segment", reply, seconds))
        time.sleep(seconds)
    post(("state", "idle" if manual else "waiting"))


# ---------------------------------------------------------------------------
# Display (Tkinter, main thread only)
# ---------------------------------------------------------------------------
def pick_family(root, *names):
    available = set(tkfont.families(root))
    for name in names:
        if name in available:
            return name
    return "TkDefaultFont"


def round_rect(canvas, x1, y1, x2, y2, r, **kwargs):
    points = [
        x1 + r,
        y1,
        x2 - r,
        y1,
        x2,
        y1,
        x2,
        y1 + r,
        x2,
        y2 - r,
        x2,
        y2,
        x2 - r,
        y2,
        x1 + r,
        y2,
        x1,
        y2,
        x1,
        y2 - r,
        x1,
        y1 + r,
        x1,
        y1,
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


class MomoDisplay:
    STATUS = {
        "loading": "Waking up MOMO…",
        "waiting": "Say “Hey MOMO” or tap the button.",
        "listening": "I'm listening. Just pause when you're done.",
        "transcribing": "Listening back…",
        "thinking": "Thinking…",
        "preparing": "Getting ready to answer…",
        "speaking": "",
    }
    BUTTON = {
        "loading": ("One moment…", BUTTON_BUSY),
        "idle": ("Tap to talk", INK),
        "waiting": ("Tap to talk", INK),
        "listening": ("Listening…", BUTTON_LISTEN),
        "transcribing": ("One moment…", BUTTON_BUSY),
        "thinking": ("One moment…", BUTTON_BUSY),
        "preparing": ("One moment…", BUTTON_BUSY),
        "speaking": ("MOMO is talking", BUTTON_BUSY),
        "error": ("Unavailable", BUTTON_BUSY),
    }
    # Which status icon each state shows (anything else: a quiet resting line)
    ICON = {
        "listening": "ring",
        "transcribing": "dots",
        "thinking": "dots",
        "preparing": "dots",
        "speaking": "wave",
    }
    TAPPABLE = ("idle", "waiting")
    LIGHT_TEXT = (INK, BUTTON_LISTEN)

    def __init__(self, root, windowed):
        self.root = root
        root.title("MOMO")
        root.configure(bg=BG)
        if windowed:
            self.W, self.H = 600, 1024
            root.geometry("{}x{}".format(self.W, self.H))
        else:
            root.attributes("-fullscreen", True)
            root.config(cursor="none")  # touchscreen: hide the mouse pointer
            self.W, self.H = root.winfo_screenwidth(), root.winfo_screenheight()
        # Designed at 600x1024; scale to fit so a landscape laptop screen works too
        s = self.s = min(self.W / 600.0, self.H / 1024.0)

        self.events = queue.Queue()
        self.tapped = threading.Event()
        self.state = "loading"
        self.hint = ""
        self.captions_on = True
        self.cue_lines = []
        self.cue_gen = 0
        self.topic = ""
        self.subtopic = ""
        self.figure = None
        self.t0 = time.time()

        self.family = pick_family(
            root, "Nunito", "Avenir Next", "Ubuntu", "Noto Sans", "DejaVu Sans"
        )
        caption_family = pick_family(
            root, "Roboto", "Helvetica Neue", "Noto Sans", "DejaVu Sans"
        )
        self.title_size = round(42 * s)
        self.f_title = tkfont.Font(
            root, family=self.family, size=-self.title_size, weight="bold"
        )
        self.f_sub = tkfont.Font(root, family=self.family, size=-round(20 * s))
        self.f_status = tkfont.Font(root, family=self.family, size=-round(22 * s))
        self.f_button = tkfont.Font(
            root, family=self.family, size=-round(28 * s), weight="bold"
        )
        self.f_cc = tkfont.Font(
            root, family=caption_family, size=-round(24 * s), weight="bold"
        )
        self.f_caption = tkfont.Font(root, family=caption_family, size=-round(30 * s))

        self.c = tk.Canvas(
            root, width=self.W, height=self.H, bg=BG, highlightthickness=0
        )
        self.c.pack(fill="both", expand=True)

        # Layout. Without a figure the icon sits where the eyes were; with one,
        # the figure fills the middle and a small icon + status sit under it.
        self.margin = 32 * s
        self.button_top = self.H - self.margin - 88 * s
        self.caption_bottom = self.button_top - 40 * s
        caption_top = self.caption_bottom - CAPTION_MAX_LINES * (
            self.f_caption.metrics("linespace") + 8 * s
        )
        self.header_width = (
            self.W - 2 * self.margin - 120 * s
        )  # stay clear of the CC button
        self.icon_y = self.H * 0.38
        self.status_y = self.icon_y + 120 * s
        self.fig_status_y = caption_top - 22 * s
        self.fig_icon_y = self.fig_status_y - 48 * s
        self.fig_top = 150 * s
        self.fig_bottom = self.fig_icon_y - 40 * s

        self.title_id = self.c.create_text(
            self.margin, self.margin - 4 * s, anchor="nw", font=self.f_title, fill=INK
        )
        self.sub_id = self.c.create_text(
            self.margin,
            self.margin + 52 * s,
            anchor="nw",
            font=self.f_sub,
            fill=INK_SOFT,
            width=self.header_width,
        )
        self.status_id = self.c.create_text(
            self.W / 2,
            self.status_y,
            text="",
            font=self.f_status,
            fill=INK_SOFT,
            width=self.W - 2 * self.margin,
            justify="center",
        )
        self._draw_header()
        self._draw_cc()
        self._set_state("loading")

        self.c.tag_bind("cc", "<Button-1>", lambda e: self.toggle_captions())
        self.c.tag_bind("talk", "<Button-1>", lambda e: self.tap_talk())
        root.bind("<space>", lambda e: self.tap_talk())
        root.bind("c", lambda e: self.toggle_captions())
        root.bind("<Escape>", lambda e: root.destroy())

        root.after(50, self._poll)
        root.after(ANIMATION_MS, self._animate)

    # ---- Talk button (shared with the voice thread) ---------------------
    def tap_talk(self):
        if self.state in self.TAPPABLE:
            self.tapped.set()

    def wait_for_tap(self, timeout=None):
        """Called from the voice thread. Blocks until the talk button is tapped."""
        self.tapped.wait(timeout)
        self.tapped.clear()

    # ---- Title and subtitle ---------------------------------------------
    def _draw_header(self):
        title = self.topic or DEFAULT_TITLE
        if not self.topic:
            subtitle = DEFAULT_SUBTITLE
        else:
            subtitle = self.subtopic or figure_title(self.figure)
        # Long topics shrink (down to 28 px) and only then get cut short
        size = self.title_size
        self.f_title.configure(size=-size)
        while self.f_title.measure(title) > self.header_width and size > round(
            28 * self.s
        ):
            size -= 1
            self.f_title.configure(size=-size)
        while self.f_title.measure(title) > self.header_width and len(title) > 4:
            title = title[:-2].rstrip() + "…"
        self.c.itemconfigure(self.title_id, text=title)
        self.c.itemconfigure(self.sub_id, text=subtitle)

    def set_lesson(self, topic, subtopic, figure):
        if topic:  # empty: keep the current topic
            if topic != self.topic:
                self.subtopic = ""
            self.topic = topic
            self.subtopic = subtopic
        if figure != self.figure:
            self.figure = figure
            self._draw_figure()
        self._draw_header()

    # ---- Lesson figure --------------------------------------------------
    def _draw_figure(self):
        self.c.delete("figure")
        if self.figure:
            m = self.margin
            draw_figure(
                self.c,
                self.figure,
                m,
                self.fig_top,
                self.W - 2 * m,
                self.fig_bottom - self.fig_top,
                self.family,
                tag="figure",
            )
            self.c.tag_raise("caption")
        self.c.coords(
            self.status_id,
            self.W / 2,
            self.fig_status_y if self.figure else self.status_y,
        )

    # ---- Status icon ----------------------------------------------------
    def _animate(self):
        self._draw_icon(time.time() - self.t0)
        self.root.after(ANIMATION_MS, self._animate)

    def _draw_icon(self, t):
        c, s = self.c, self.s
        c.delete("icon")
        kind = self.ICON.get(self.state, "rest")
        if self.figure:
            if kind == "rest":
                return  # the figure is enough on its own
            k, cy = 0.4 * s, self.fig_icon_y
        else:
            k, cy = s, self.icon_y
        cx = self.W / 2

        if kind == "rest":
            round_rect(
                c,
                cx - 100 * k,
                cy - 3 * k,
                cx + 100 * k,
                cy + 3 * k,
                3 * k,
                fill=BUTTON_BUSY,
                outline="",
                tags="icon",
            )

        elif kind == "ring":  # listening: a ring that breathes out
            p = (t % 1.6) / 1.6
            r = 60 * k * (1 + 0.7 * p)
            w = max(2, 6 * k)
            c.create_oval(
                cx - r,
                cy - r,
                cx + r,
                cy + r,
                outline=mix(BG, BUTTON_LISTEN, 0.7 * (1 - p)),
                width=w,
                tags="icon",
            )
            c.create_oval(
                cx - 60 * k,
                cy - 60 * k,
                cx + 60 * k,
                cy + 60 * k,
                outline=BUTTON_LISTEN,
                width=w,
                tags="icon",
            )
            c.create_oval(
                cx - 20 * k,
                cy - 20 * k,
                cx + 20 * k,
                cy + 20 * k,
                fill=BUTTON_LISTEN,
                outline="",
                tags="icon",
            )

        elif kind == "dots":  # thinking: dots appear one by one
            for i in range(3):
                p = ((t - 0.3 * i) % 1.8) / 1.8
                if p < 0.12:
                    a = 0.12 + 0.88 * p / 0.12
                elif p < 0.55:
                    a = 1.0
                elif p < 0.75:
                    a = 1.0 - 0.88 * (p - 0.55) / 0.2
                else:
                    a = 0.12
                x = cx + (i - 1) * 64 * k
                c.create_oval(
                    x - 17 * k,
                    cy - 17 * k,
                    x + 17 * k,
                    cy + 17 * k,
                    fill=mix(BG, INK, a),
                    outline="",
                    tags="icon",
                )

        elif kind == "wave":  # speaking: a voice-like wave
            width, height = 420 * k, 150 * k
            loud = 0.55 + 0.45 * abs(math.sin(t * 2.3) * math.sin(t * 0.9 + 1))
            for amp, freq, phase, alpha in (
                (0.45, 3.1, 3.1, 0.25),
                (0.7, 2.3, 1.7, 0.45),
                (1.0, 1.6, 0.0, 1.0),
            ):
                pts = []
                for i in range(61):
                    p = i / 60
                    y = cy + math.sin(
                        math.pi * p
                    ) * amp * loud * height * 0.42 * math.sin(
                        p * math.pi * 2 * freq + t * 6 + phase
                    )
                    pts += [cx - width / 2 + p * width, y]
                c.create_line(
                    pts,
                    fill=mix(BG, INK, alpha),
                    width=max(2, 6.5 * k),
                    capstyle="round",
                    joinstyle="round",
                    tags="icon",
                )

    # ---- CC toggle ------------------------------------------------------
    def _draw_cc(self):
        s, m = self.s, self.margin
        self.c.delete("cc")
        w, h = 88 * s, 60 * s
        x1, y1 = self.W - m - w, m
        # Larger invisible hit area so it's easy to tap with a finger
        self.c.create_rectangle(
            x1 - 16 * s, 0, self.W, y1 + h + 20 * s, fill=BG, outline="", tags="cc"
        )
        if self.captions_on:
            round_rect(
                self.c, x1, y1, x1 + w, y1 + h, 12 * s, fill=INK, outline="", tags="cc"
            )
            self.c.create_text(
                x1 + w / 2,
                y1 + h / 2 - 3 * s,
                text="CC",
                font=self.f_cc,
                fill=BG,
                tags="cc",
            )
            self.c.create_rectangle(
                x1 + 18 * s,
                y1 + h + 6 * s,
                x1 + w - 18 * s,
                y1 + h + 11 * s,
                fill=CC_RED,
                outline="",
                tags="cc",
            )
        else:
            round_rect(
                self.c,
                x1,
                y1,
                x1 + w,
                y1 + h,
                12 * s,
                fill=BG,
                outline=INK_SOFT,
                width=max(2, round(3 * s)),
                tags="cc",
            )
            self.c.create_text(
                x1 + w / 2,
                y1 + h / 2 - 3 * s,
                text="CC",
                font=self.f_cc,
                fill=INK_SOFT,
                tags="cc",
            )

    def toggle_captions(self):
        self.captions_on = not self.captions_on
        self._draw_cc()
        self._render_caption()

    # ---- Talk button and status -----------------------------------------
    def _draw_talk_button(self):
        s, m = self.s, self.margin
        self.c.delete("talk")
        label, fill = self.BUTTON.get(self.state, self.BUTTON["idle"])
        text_color = BG if fill in self.LIGHT_TEXT else INK_SOFT
        round_rect(
            self.c,
            m,
            self.button_top,
            self.W - m,
            self.H - m,
            44 * s,
            fill=fill,
            outline="",
            tags="talk",
        )
        self.c.create_text(
            self.W / 2,
            (self.button_top + self.H - m) / 2,
            text=label,
            font=self.f_button,
            fill=text_color,
            tags="talk",
        )

    def _set_state(self, state, message=None):
        self.state = state
        if state == "listening":
            self.hint = ""
        if message is None:
            if state in self.TAPPABLE and self.hint:
                message = self.hint
            else:
                message = self.STATUS.get(state, "")
        self.c.itemconfigure(self.status_id, text=message)
        self._draw_talk_button()

    # ---- Captions -------------------------------------------------------
    def _wrap(self, text):
        max_w = self.W - 2 * self.margin - 24 * self.s
        lines, line = [], ""
        for word in text.split():
            trial = (line + " " + word).strip()
            if line and self.f_caption.measure(trial) > max_w:
                lines.append(line)
                line = word
            else:
                line = trial
        if line:
            lines.append(line)
        return lines

    def show_segment(self, text, seconds):
        """Split speech into two-line cues, each shown for its share of the audio."""
        self.cue_gen += 1
        gen = self.cue_gen
        lines = self._wrap(text)
        cues = [
            lines[i : i + CAPTION_MAX_LINES]
            for i in range(0, len(lines), CAPTION_MAX_LINES)
        ]
        total = sum(len(" ".join(cue)) for cue in cues) or 1
        t = 0.0
        for cue in cues:
            self.root.after(int(t * 1000), self._set_cue, gen, cue)
            t += seconds * len(" ".join(cue)) / total
        # Clear after this segment unless a newer one has started
        self.root.after(int(seconds * 1000) + CAPTION_HOLD_MS, self._set_cue, gen, [])

    def _set_cue(self, gen, lines):
        if gen != self.cue_gen:
            return
        self.cue_lines = lines
        self._render_caption()

    def _render_caption(self):
        self.c.delete("caption")
        if not self.captions_on or not self.cue_lines:
            return
        s = self.s
        pad_x, pad_y = 12 * s, 4 * s
        line_h = self.f_caption.metrics("linespace") + 2 * pad_y
        for i, line in enumerate(reversed(self.cue_lines)):
            cy = self.caption_bottom - i * line_h - line_h / 2
            half_w = self.f_caption.measure(line) / 2 + pad_x
            self.c.create_rectangle(
                self.W / 2 - half_w,
                cy - line_h / 2,
                self.W / 2 + half_w,
                cy + line_h / 2,
                fill=CAPTION_BOX,
                outline="",
                tags="caption",
            )
            self.c.create_text(
                self.W / 2,
                cy,
                text=line,
                font=self.f_caption,
                fill=CAPTION_TEXT,
                tags="caption",
            )

    # ---- Events from the voice thread -----------------------------------
    def _poll(self):
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "state":
                    self._set_state(event[1])
                elif kind == "segment":
                    self.show_segment(event[1], event[2])
                elif kind == "lesson":
                    self.set_lesson(event[1], event[2], event[3])
                elif kind == "hint":
                    self.hint = event[1]
                elif kind == "error":
                    self._set_state("error", event[1])
                elif kind == "quit":
                    self.root.after(CAPTION_HOLD_MS, self.root.destroy)
        except queue.Empty:
            pass
        self.root.after(50, self._poll)


def main():
    parser = argparse.ArgumentParser(
        description="MOMO display with captions (laptop pipeline)"
    )
    parser.add_argument(
        "--windowed", action="store_true", help="600x1024 window instead of fullscreen"
    )
    parser.add_argument(
        "--manual", action="store_true", help="tap only, no 'Hey MOMO' wake word"
    )
    parser.add_argument(
        "--demo", action="store_true", help="fake speech, no models or audio"
    )
    args = parser.parse_args()

    root = tk.Tk()
    ui = MomoDisplay(root, windowed=args.windowed)
    worker = demo_loop if args.demo else voice_loop
    threading.Thread(target=worker, args=(ui, args.manual), daemon=True).start()
    root.mainloop()


if __name__ == "__main__":
    main()
