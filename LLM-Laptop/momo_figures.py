"""
Ready-made lesson figures for MOMO's screen.

Qwen picks a figure by ID (the "figure" field in its JSON reply, see
Modelfile.qwen) and momo_display.py draws it with draw_figure().

Every figure is a list of simple shapes laid out on a 536 x 560 design box,
so figures are easy to add or edit: copy one, change the coordinates and text,
add its ID to FIGURES and describe it in Modelfile.qwen's FIGURE list.

Shapes:
    ("rect", x1, y1, x2, y2, radius, fill, outline, width)
    ("oval", x1, y1, x2, y2, fill, outline, width)
    ("line", [(x, y), ...], color, width, arrow, dash)   arrow: None/"last"/"both"
    ("poly", [(x, y), ...], fill)
    ("text", x, y, text, size, bold, color, anchor)      anchor: Tk style, e.g. "center", "w"
"""
import math

DESIGN_W, DESIGN_H = 536, 560

INK = "#3D3524"
SOFT = "#6E6450"
FAINT = "#A99E80"
PAPER = "#FFFDF0"
TAN = "#E3D7A8"
SKIN = "#F3D9B8"
SKIN_DEEP = "#E9C49B"
HEAT = "#D9473B"
WATER = "#3F86C6"
MUSCLE = "#C9826B"


# ---------------------------------------------------------------------------
# Small helpers for building shapes
# ---------------------------------------------------------------------------
def curve(p0, c, p1, steps=24):
    """Quadratic Bezier from p0 to p1 bent toward control point c."""
    pts = []
    for i in range(steps + 1):
        t = i / steps
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * c[0] + t ** 2 * p1[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * c[1] + t ** 2 * p1[1]
        pts.append((x, y))
    return pts


def wavy(x, y1, y2, amp=7, waves=2.0, steps=28):
    """Vertical wavy path (a heat or vapour arrow) from y1 up/down to y2."""
    return [(x + amp * math.sin(i / steps * waves * 2 * math.pi), y1 + (y2 - y1) * i / steps)
            for i in range(steps + 1)]


def box(x1, y1, x2, y2, title, lines=(), fill=PAPER, outline=INK, title_size=21):
    """Rounded box with a bold title and up to two small lines, centred."""
    cx = (x1 + x2) / 2
    n = len(lines)
    top = (y1 + y2) / 2 - (13 * n)
    shapes = [("rect", x1, y1, x2, y2, 16, fill, outline, 3 if outline else 0),
              ("text", cx, top, title, title_size, True, INK, "center")]
    for i, line in enumerate(lines):
        shapes.append(("text", cx, top + 27 + 24 * i, line, 17, False, SOFT, "center"))
    return shapes


def arrow(points, color=INK, width=4, dash=None):
    return ("line", points, color, width, "last", dash)


def text(x, y, s, size=17, bold=False, color=SOFT, anchor="center"):
    return ("text", x, y, s, size, bold, color, anchor)


def skin_block(top=170, bottom=540):
    """Shared background for the skin cross-sections."""
    surface = [(0, top)] + [(x, top + 5 * math.sin(x / 536 * 6 * math.pi)) for x in range(0, 537, 8)] + [(536, top)]
    return [
        ("rect", 0, 0, 536, top + 10, 0, PAPER, None, 0),
        ("poly", surface + [(536, bottom), (0, bottom)], SKIN),
        ("rect", 0, top + 70, 536, bottom, 0, SKIN_DEEP, None, 0),
        text(14, top - 22, "outside air", 16, False, FAINT, "w"),
        text(14, top + 30, "skin", 16, False, SOFT, "w"),
    ]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_feedback_loop():
    s = []
    s += [("oval", 198, 210, 338, 350, PAPER, INK, 4),
          text(268, 270, "37 °C", 34, True, INK),
          text(268, 304, "set point", 18, False, SOFT)]
    s += box(148, 20, 388, 104, "Body heats up", ["running, a hot day"], fill=TAN, outline=None, title_size=22)
    s += box(340, 226, 530, 334, "Receptors", ["in the skin", "sense heat"])
    s += box(148, 456, 388, 544, "Hypothalamus", ["control centre"])
    s += box(6, 226, 196, 334, "Sweat glands", ["effectors:", "sweat cools skin"])
    s += [arrow(curve((392, 70), (470, 88), (452, 218))),
          arrow(curve((452, 342), (470, 470), (396, 494))),
          arrow(curve((140, 494), (66, 470), (84, 342))),
          arrow(curve((84, 218), (66, 90), (142, 70)), dash=(2, 10)),
          text(44, 140, "temp", 16, False, FAINT), text(44, 160, "falls", 16, False, FAINT)]
    return s


def fig_set_point():
    left, right, top, bottom = 84, 516, 40, 420

    def y_of(t):
        return bottom - (t - 35.0) / 4.0 * (bottom - top)

    s = [("rect", left, y_of(37.5), right, y_of(36.5), 0, TAN, None, 0),
         text(right - 8, y_of(37.5) + 16, "normal range", 15, False, SOFT, "e"),
         ("line", [(left, top - 10), (left, bottom), (right + 6, bottom)], INK, 3, None, None),
         ("line", [(left, y_of(37)), (right, y_of(37))], FAINT, 2, None, (6, 6))]
    for t in (35, 37, 39):
        s.append(text(left - 12, y_of(t), "{} °C".format(t), 16, t == 37, INK if t == 37 else SOFT, "e"))
    pts = []
    for i in range(121):
        p = i / 120
        t = 37 + 1.45 * math.exp(-((p - 0.27) / 0.085) ** 2) - 1.25 * math.exp(-((p - 0.68) / 0.085) ** 2)
        pts.append((left + p * (right - left), y_of(t)))
    s.append(("line", pts, INK, 5, None, None))
    px = left + 0.27 * (right - left)
    tx = left + 0.68 * (right - left)
    s += [text(px, y_of(38.45) - 22, "Too hot: sweat", 18, True, HEAT),
          text(tx, y_of(35.75) + 24, "Too cold: shiver", 18, True, WATER),
          text(right, bottom + 22, "time →", 16, False, SOFT, "e"),
          text(268, 492, "Your body pushes any change", 19, False, INK),
          text(268, 518, "back toward the set point.", 19, False, INK)]
    return s


def fig_cooling_down():
    top = 170
    s = skin_block(top)
    # sweat gland: coiled tube deep in the skin, duct up to a pore
    s.append(("line", [(140, top + 4), (146, 260), (136, 330), (146, 390)], WATER, 5, None, None))
    for i in range(4):
        s.append(("oval", 116 + i * 9, 384 + (i % 2) * 14, 160 + i * 9, 420 + (i % 2) * 14, "", WATER, 4))
    s.append(text(150, 470, "sweat gland", 17, True, INK))
    # drops on the surface, vapour and heat leaving
    s += [("oval", 126, top - 22, 152, top + 2, WATER, None, 0),
          ("oval", 164, top - 14, 182, top + 4, WATER, None, 0),
          arrow(wavy(140, top - 34, 46), HEAT, 4),
          arrow(wavy(176, top - 26, 64), HEAT, 4),
          text(196, 34, "Sweat evaporates", 19, True, INK, "w"),
          text(196, 58, "and carries heat away", 17, False, SOFT, "w")]
    # widened blood vessel
    s += [("rect", 250, 280, 536, 340, 30, HEAT, None, 0),
          text(390, 310, "wide blood vessel", 17, True, PAPER),
          arrow(wavy(330, 274, top + 8, 5, 1.5), HEAT, 3),
          arrow(wavy(440, 274, top + 8, 5, 1.5), HEAT, 3),
          text(390, 372, "Vessels widen, so more warm", 17, False, INK),
          text(390, 396, "blood flows near the skin", 17, False, INK),
          text(390, 420, "and loses heat to the air.", 17, False, INK)]
    return s


def fig_warming_up():
    top = 170
    s = skin_block(top, bottom=420)
    # raised hair with a goosebump
    s += [("oval", 96, top - 16, 152, top + 16, SKIN, None, 0),
          ("line", [(124, top + 50), (124, top - 6), (138, top - 96)], INK, 4, None, None),
          ("oval", 116, top + 44, 132, top + 60, SKIN_DEEP, INK, 2),
          text(160, top - 80, "Hairs stand up", 19, True, INK, "w"),
          text(160, top - 56, "(goosebumps) and trap air", 17, False, SOFT, "w")]
    # narrowed blood vessel, heat kept in
    s += [("rect", 250, 300, 536, 318, 9, HEAT, None, 0),
          text(394, 280, "Vessels narrow: less heat lost", 17, True, INK),
          text(394, 346, "warm blood stays deeper inside", 16, False, SOFT)]
    # shivering muscle
    s += [("rect", 0, 420, 536, 540, 0, MUSCLE, None, 0),
          text(14, 448, "muscle", 16, False, PAPER, "w")]
    zig = []
    for i in range(23):
        zig.append((40 + i * 20, 486 + (12 if i % 2 else -12)))
    s += [("line", zig, PAPER, 4, None, None),
          text(268, 528, "Muscles shiver: tiny fast twitches make heat", 17, True, PAPER)]
    return s


def fig_thermostat():
    rows = [("Sensor", "Thermometer", "Receptors in skin"),
            ("Controller", "Thermostat", "Hypothalamus"),
            ("Action", "Heater or A/C", "Shiver or sweat"),
            ("Target", "Set to 21 °C", "Set point 37 °C")]
    s = [text(206, 30, "House", 22, True, INK), text(420, 30, "Your body", 22, True, INK)]
    for i, (role, house, body) in enumerate(rows):
        y1 = 62 + i * 92
        y2 = y1 + 72
        mid = (y1 + y2) / 2
        s += [text(8, mid, role, 17, True, SOFT, "w"),
              ("rect", 120, y1, 292, y2, 14, TAN, None, 0),
              text(206, mid, house, 19, True, INK),
              ("rect", 334, y1, 530, y2, 14, PAPER, INK, 3),
              text(432, mid, body, 19, True, INK),
              ("line", [(296, mid), (330, mid)], FAINT, 3, "both", None)]
    s += [text(268, 460, "Both measure, compare to a setting,", 19, False, INK),
          text(268, 486, "then act to undo the change.", 19, False, INK),
          text(268, 520, "That loop is negative feedback.", 19, True, INK)]
    return s


def fig_control_parts():
    steps = [("Stimulus", "body temperature changes", TAN, None),
             ("Receptor", "detects it: skin & brain sensors", PAPER, INK),
             ("Control centre", "decides: hypothalamus", PAPER, INK),
             ("Effector", "acts: sweat glands, muscles, vessels", PAPER, INK),
             ("Response", "temperature returns to normal", TAN, None)]
    s = []
    h, gap, x1, x2 = 78, 26, 96, 520
    for i, (name, detail, fill, outline) in enumerate(steps):
        y1 = 10 + i * (h + gap)
        s += [("rect", x1, y1, x2, y1 + h, 16, fill, outline, 3 if outline else 0),
              text((x1 + x2) / 2, y1 + 28, name, 21, True, INK),
              text((x1 + x2) / 2, y1 + 54, detail, 17, False, SOFT)]
        if i < len(steps) - 1:
            s.append(arrow([((x1 + x2) / 2, y1 + h + 2), ((x1 + x2) / 2, y1 + h + gap - 2)], INK, 4))
    last = 10 + 4 * (h + gap) + h / 2
    s += [arrow([(x1 - 4, last), (40, last), (40, 10 + h / 2), (x1 - 6, 10 + h / 2)], FAINT, 3, (2, 9)),
          text(50, 10 + h / 2 - 22, "feedback", 16, False, FAINT, "center")]
    return s


# id -> (title shown under the topic, shape builder)
FIGURES = {
    "feedback_loop": ("Negative feedback loop", fig_feedback_loop),
    "set_point": ("Staying near 37 °C", fig_set_point),
    "cooling_down": ("How the body cools down", fig_cooling_down),
    "warming_up": ("How the body warms up", fig_warming_up),
    "thermostat": ("Like a home thermostat", fig_thermostat),
    "control_parts": ("Receptor, control centre, effector", fig_control_parts),
}


def figure_title(fig_id):
    return FIGURES[fig_id][0] if fig_id in FIGURES else ""


# ---------------------------------------------------------------------------
# Tk renderer
# ---------------------------------------------------------------------------
def draw_figure(canvas, fig_id, x, y, w, h, family, tag="figure"):
    """Draw figure fig_id fitted (aspect kept) and centred in the box x, y, w, h."""
    import tkinter.font as tkfont

    canvas.delete(tag)
    if fig_id not in FIGURES:
        return
    k = min(w / DESIGN_W, h / DESIGN_H)
    ox = x + (w - DESIGN_W * k) / 2
    oy = y + (h - DESIGN_H * k) / 2

    def P(px, py):
        return ox + px * k, oy + py * k

    fonts = {}

    def font(size, bold):
        key = (size, bold)
        if key not in fonts:
            fonts[key] = tkfont.Font(canvas, family=family, size=-max(8, round(size * k)),
                                     weight="bold" if bold else "normal")
        return fonts[key]

    for shape in FIGURES[fig_id][1]():
        kind = shape[0]
        if kind == "rect":
            _, x1, y1, x2, y2, r, fill, outline, width = shape
            (a, b), (c, d) = P(x1, y1), P(x2, y2)
            r = min(r * k, (c - a) / 2, (d - b) / 2)
            opts = dict(fill=fill or "", outline=outline or "", width=max(1, width * k) if outline else 0, tags=tag)
            if r > 0:
                pts = [a + r, b, c - r, b, c, b, c, b + r, c, d - r, c, d,
                       c - r, d, a + r, d, a, d, a, d - r, a, b + r, a, b]
                canvas.create_polygon(pts, smooth=True, **opts)
            else:
                canvas.create_rectangle(a, b, c, d, **opts)
        elif kind == "oval":
            _, x1, y1, x2, y2, fill, outline, width = shape
            (a, b), (c, d) = P(x1, y1), P(x2, y2)
            canvas.create_oval(a, b, c, d, fill=fill or "", outline=outline or "",
                               width=max(1, width * k) if outline else 0, tags=tag)
        elif kind == "poly":
            _, pts, fill = shape
            canvas.create_polygon([c for p in pts for c in P(*p)], fill=fill, outline="", tags=tag)
        elif kind == "line":
            _, pts, color, width, arrow_end, dash = shape
            opts = dict(fill=color, width=max(1, width * k), capstyle="round", joinstyle="round", tags=tag)
            if arrow_end:
                size = max(6, (10 + 2.2 * width) * k)
                opts.update(arrow=arrow_end, arrowshape=(size, size * 1.1, size * 0.45))
            if dash:
                opts["dash"] = tuple(max(1, round(v * k)) for v in dash)
            canvas.create_line([c for p in pts for c in P(*p)], **opts)
        elif kind == "text":
            _, tx, ty, s, size, bold, color, anchor = shape
            canvas.create_text(*P(tx, ty), text=s, font=font(size, bold), fill=color, anchor=anchor, tags=tag)
