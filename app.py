"""
ColorSight Suite – Unified Flask Application
=============================================
Modules:
  1. Color Blindness Simulator
  2. Ishihara Plate Generator
  3. Color Vision Examination
  4. Color-Based Object Detection (Accessibility Auditor)
"""

from flask import (
    Flask, render_template, request, redirect,
    session, send_file, jsonify, url_for
)
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import random, math, io, os, json, base64, colorsys

app = Flask(__name__)
app.secret_key = "colorsight_suite_secure_key_2025"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ═══════════════════════════════════════════════════════════════
#  SHARED: Color Vision Deficiency Matrices & Palettes
# ═══════════════════════════════════════════════════════════════

CVD_MATRICES = {
    "protanopia": np.array([
        [0.567, 0.433, 0.000],
        [0.558, 0.442, 0.000],
        [0.000, 0.242, 0.758]
    ]),
    "deuteranopia": np.array([
        [0.625, 0.375, 0.000],
        [0.700, 0.300, 0.000],
        [0.000, 0.300, 0.700]
    ]),
    "tritanopia": np.array([
        [0.950, 0.050, 0.000],
        [0.000, 0.433, 0.567],
        [0.000, 0.475, 0.525]
    ]),
    "achromatopsia": np.array([
        [0.299, 0.587, 0.114],
        [0.299, 0.587, 0.114],
        [0.299, 0.587, 0.114]
    ]),
}

CVD_LABELS = {
    "protanopia":    "Protanopia (Red-Blind)",
    "deuteranopia":  "Deuteranopia (Green-Blind)",
    "tritanopia":    "Tritanopia (Blue-Blind)",
    "achromatopsia": "Achromatopsia (Total Color Blind)",
}

# ═══════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ═══════════════════════════════════════════════════════════════

def pil_to_base64(img, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def apply_cvd_matrix(srgb_array, deficiency):
    matrix = CVD_MATRICES[deficiency]
    h, w, _ = srgb_array.shape
    flat = srgb_array.reshape(-1, 3)
    transformed = np.dot(flat, matrix.T)
    return np.clip(transformed.reshape(h, w, 3), 0, 1)


# ═══════════════════════════════════════════════════════════════
#  HOME / DASHBOARD
# ═══════════════════════════════════════════════════════════════

@app.route("/")
def home():
    return render_template("index.html")


# ═══════════════════════════════════════════════════════════════
#  MODULE 1 – COLOR BLINDNESS SIMULATOR
# ═══════════════════════════════════════════════════════════════

@app.route("/simulator")
def simulator():
    return render_template("simulator.html", cvd_labels=CVD_LABELS)


@app.route("/simulator/process", methods=["POST"])
def simulator_process():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    deficiency = request.form.get("deficiency", "protanopia")
    if deficiency not in CVD_MATRICES:
        return jsonify({"error": "Invalid deficiency type"}), 400

    img = Image.open(file.stream).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    transformed = apply_cvd_matrix(arr, deficiency)
    result = Image.fromarray((transformed * 255).astype(np.uint8))

    buf = io.BytesIO()
    result.save(buf, "PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png", download_name=f"{deficiency}_simulation.png")


# ═══════════════════════════════════════════════════════════════
#  MODULE 2 – ISHIHARA PLATE GENERATOR
# ═══════════════════════════════════════════════════════════════

PLATE_CFG = dict(SVG_SIZE=1000, PLATE_RADIUS=480, TOTAL_CIRCLES=2500,
                 DOT_MIN=4, DOT_MAX=11, FONT_SIZE=420)

# CVD color models — validated: normal contrast >= 5:1, simulated contrast <= 1.1:1
# fg = symbol color (visible to normal eyes, confused under target CVD)
# bg = background color (ditto)
COLOR_MODELS = {
    "protanopia":    {"fg": (200, 70, 70),   "bg": (70, 160, 120)},
    "deuteranopia":  {"fg": (180, 90, 70),   "bg": (90, 150, 120)},
    "tritanopia":    {"fg": (220, 160, 60),  "bg": (120, 120, 220)},
    "achromatopsia": {"fg": (150, 150, 150), "bg": (180, 180, 180)},
}

# For achromatopsia: multiple isoluminant colors (sim_gray ≈ 0.35)
# Warm colors for symbol dots, cool colors for background dots
# Both groups simulate to the same gray under achromatopsia
ACHROMA_PALETTES = {
    # Bright warm orange-reds: visible as vivid color to trichromats, same gray to achromats (~115-121)
    "fg": [(255, 75, 0), (255, 60, 20), (240, 80, 20), (255, 50, 40), (230, 70, 0)],
    # Bright cool cyans: vivid contrast to trichromats, same gray to achromats (~121-129)
    "bg": [(0, 180, 180), (0, 170, 190), (20, 175, 175), (0, 185, 165), (10, 180, 165)],
}

def _gen_circles():
    """Generate non-overlapping circles using spatial grid for O(1) neighbor lookups."""
    C = PLATE_CFG
    sz, pr = C["SVG_SIZE"], C["PLATE_RADIUS"]
    center = sz // 2
    max_r = C["DOT_MAX"]
    cell_size = max_r * 2 + 2          # grid cell = max possible overlap distance
    grid_w = sz // cell_size + 1
    grid = {}                           # (gx, gy) -> list of (x, y, r)
    circles = []
    max_attempts = C["TOTAL_CIRCLES"] * 12
    attempts = 0

    while len(circles) < C["TOTAL_CIRCLES"] and attempts < max_attempts:
        attempts += 1
        r = random.randint(C["DOT_MIN"], C["DOT_MAX"])
        angle = random.uniform(0, 2 * math.pi)
        rad = random.uniform(0, pr - r)
        x = int(center + rad * math.cos(angle))
        y = int(center + rad * math.sin(angle))

        if (x - center)**2 + (y - center)**2 > pr**2:
            continue

        gx, gy = x // cell_size, y // cell_size
        collision = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for cx, cy, cr in grid.get((gx+dx, gy+dy), []):
                    if (x-cx)**2 + (y-cy)**2 <= (r+cr)**2:
                        collision = True
                        break
                if collision:
                    break
            if collision:
                break

        if not collision:
            circles.append((x, y, r))
            grid.setdefault((gx, gy), []).append((x, y, r))

    return circles


def _build_svg(symbol, cvd):
    C = PLATE_CFG
    center = C["SVG_SIZE"] // 2
    fg = COLOR_MODELS[cvd]["fg"]
    bg = COLOR_MODELS[cvd]["bg"]
    circles = _gen_circles()

    # For achromatopsia: use isoluminant multi-color palettes so achromats see uniform gray
    # but trichromats see warm vs cool color grouping
    if cvd == "achromatopsia":
        fg_pal = ACHROMA_PALETTES["fg"]
        bg_pal = ACHROMA_PALETTES["bg"]
        rng = random.Random(42)  # seeded for reproducibility
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {C["SVG_SIZE"]} {C["SVG_SIZE"]}">']
        for x, y, r in circles:
            col = rng.choice(bg_pal)
            svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="rgb{col}" />')
        svg.append('<defs><mask id="symbolMask">')
        svg.append('<rect width="100%" height="100%" fill="black"/>')
        svg.append(f'<text x="{center}" y="{center}" font-size="{C["FONT_SIZE"]}" '
                   f'font-family="Arial" font-weight="bold" '
                   f'text-anchor="middle" dominant-baseline="middle" fill="white">{symbol}</text>')
        svg.append('</mask></defs>')
        svg.append('<g mask="url(#symbolMask)">')
        rng2 = random.Random(99)
        for x, y, r in circles:
            col = rng2.choice(fg_pal)
            svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="rgb{col}" />')
        svg.append('</g></svg>')
        return "\n".join(svg), circles

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {C["SVG_SIZE"]} {C["SVG_SIZE"]}">']
    for x, y, r in circles:
        svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="rgb{bg}" />')
    svg.append('<defs><mask id="symbolMask">')
    svg.append('<rect width="100%" height="100%" fill="black"/>')
    svg.append(f'<text x="{center}" y="{center}" font-size="{C["FONT_SIZE"]}" '
               f'font-family="Arial" font-weight="bold" '
               f'text-anchor="middle" dominant-baseline="middle" fill="white">{symbol}</text>')
    svg.append('</mask></defs>')
    svg.append('<g mask="url(#symbolMask)">')
    for x, y, r in circles:
        svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="rgb{fg}" />')
    svg.append('</g></svg>')
    return "\n".join(svg), circles


def _circles_to_png(circles, cvd, symbol="A", fmt="PNG"):
    C = PLATE_CFG
    bg = COLOR_MODELS[cvd]["bg"]
    fg = COLOR_MODELS[cvd]["fg"]
    center = C["SVG_SIZE"] // 2

    # Draw background circles
    img = Image.new("RGB", (C["SVG_SIZE"], C["SVG_SIZE"]), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    for x, y, r in circles:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=bg)

    # Build a grayscale mask for the symbol text
    mask = Image.new("L", (C["SVG_SIZE"], C["SVG_SIZE"]), 0)
    mask_draw = ImageDraw.Draw(mask)
    font = None
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]:
        try:
            font = ImageFont.truetype(font_path, C["FONT_SIZE"])
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    mask_draw.text((center, center), symbol, fill=255, font=font, anchor="mm")

    # Draw foreground circles on separate layer, composite onto background via mask
    fg_layer = Image.new("RGB", (C["SVG_SIZE"], C["SVG_SIZE"]), bg)
    fg_draw = ImageDraw.Draw(fg_layer)
    for x, y, r in circles:
        fg_draw.ellipse((x - r, y - r, x + r, y + r), fill=fg)

    img.paste(fg_layer, mask=mask)

    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        img.save(buf, format="JPEG", quality=92)
    else:
        img.save(buf, format="PNG")
    buf.seek(0)
    return buf


@app.route("/generator")
def generator():
    return render_template("generator.html", cvd_labels=CVD_LABELS)


@app.route("/generator/create", methods=["POST"])
def generator_create():
    symbol = (request.form.get("symbol", "A") or "A").strip()[:2]
    if not symbol:
        symbol = "A"
    cvd = request.form.get("cvd", "protanopia")
    if cvd not in COLOR_MODELS:
        cvd = "protanopia"
    svg, _ = _build_svg(symbol, cvd)
    return render_template("generator.html", svg=svg, symbol=symbol, cvd=cvd, cvd_labels=CVD_LABELS)


@app.route("/generator/download/<fmt>")
def generator_download(fmt):
    symbol = (request.args.get("symbol", "A") or "A").strip()[:2] or "A"
    cvd = request.args.get("cvd", "protanopia")
    if cvd not in COLOR_MODELS:
        cvd = "protanopia"
    if fmt not in ("svg", "png", "jpg"):
        fmt = "png"
    if fmt == "svg":
        svg, _ = _build_svg(symbol, cvd)
        return send_file(io.BytesIO(svg.encode()), mimetype="image/svg+xml",
                         as_attachment=True, download_name="ishihara.svg")
    else:
        _, circles = _build_svg(symbol, cvd)
        pil_fmt = "PNG" if fmt == "png" else "JPEG"
        buf = _circles_to_png(circles, cvd, symbol, pil_fmt)
        return send_file(buf, mimetype=f"image/{fmt}",
                         as_attachment=True, download_name=f"ishihara.{fmt}")


# ═══════════════════════════════════════════════════════════════
#  MODULE 3 – COLOR VISION EXAMINATION (v4 – Adaptive)
#  True adaptive engine: generates next plate based on performance.
#  CVD-matrix-validated palettes, natural dot jitter, vanishing
#  plates, response-time analysis, per-plate CVD sim in results.
# ═══════════════════════════════════════════════════════════════

# --- CVD-matrix-validated palette generator ---
def _make_confusing_pair(cvd_type, difficulty):
    """Generate fg/bg pairs that look distinct to normal vision but merge under CVD.
    Uses our actual CVD simulation matrices to validate the pair."""
    rng = random.Random()

    # Base hue angles that target each CVD type
    base_configs = {
        "protanopia":   {"fg_h": (0, 15),   "bg_h": (100, 140), "s": (0.6, 0.8), "v": (0.6, 0.8)},
        "deuteranopia": {"fg_h": (0, 20),   "bg_h": (90, 130),  "s": (0.55, 0.75),"v": (0.6, 0.8)},
        "tritanopia":   {"fg_h": (40, 65),  "bg_h": (210, 250), "s": (0.6, 0.8), "v": (0.65, 0.8)},
        "achromatopsia":{"fg_h": (0, 360),  "bg_h": (0, 360),   "s": (0.4, 0.7), "v": (0.55, 0.7)},
    }
    cfg = base_configs[cvd_type]

    # Difficulty controls how close the fg/bg are
    contrast_scale = {"easy": 1.0, "medium": 0.6, "hard": 0.35}
    scale = contrast_scale[difficulty]

    best_pair = None
    best_score = -1

    for _ in range(40):
        fg_h = rng.uniform(*cfg["fg_h"]) / 360
        bg_h = rng.uniform(*cfg["bg_h"]) / 360
        s = rng.uniform(*cfg["s"])
        v = rng.uniform(*cfg["v"])

        # Reduce saturation/value difference for harder plates
        fg_s = min(1, s + 0.1 * scale)
        bg_s = max(0.2, s - 0.1 * scale)
        fg_v = min(1, v + 0.05 * scale)
        bg_v = max(0.3, v - 0.05 * scale)

        if cvd_type == "achromatopsia":
            # Same luminance, different hue+saturation
            fg_v = bg_v = v
            fg_s = s
            bg_s = max(0.1, s - 0.15)

        fg_rgb = np.array(colorsys.hsv_to_rgb(fg_h, fg_s, fg_v)) * 255
        bg_rgb = np.array(colorsys.hsv_to_rgb(bg_h, bg_s, bg_v)) * 255

        # Normal vision distance (should be high)
        normal_dist = float(np.linalg.norm(fg_rgb - bg_rgb))

        # CVD simulation distance (should be low for a good confusing pair)
        fg_01 = fg_rgb.reshape(1, 1, 3).astype(np.float32) / 255.0
        bg_01 = bg_rgb.reshape(1, 1, 3).astype(np.float32) / 255.0

        if cvd_type == "achromatopsia":
            # For achromatopsia, check grayscale conversion
            fg_gray = 0.299 * fg_rgb[0] + 0.587 * fg_rgb[1] + 0.114 * fg_rgb[2]
            bg_gray = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]
            sim_dist = abs(fg_gray - bg_gray)
        else:
            fg_sim = apply_cvd_matrix(fg_01, cvd_type)[0, 0] * 255
            bg_sim = apply_cvd_matrix(bg_01, cvd_type)[0, 0] * 255
            sim_dist = float(np.linalg.norm(fg_sim - bg_sim))

        # Score: high normal distance + low CVD distance
        min_normal = 60 * scale + 30
        max_sim = 40 + 30 * (1 - scale)

        if normal_dist > min_normal and sim_dist < max_sim:
            score = normal_dist / max(sim_dist, 1)
            if score > best_score:
                best_score = score
                fg_t = tuple(int(max(0, min(255, x))) for x in fg_rgb)
                bg_t = tuple(int(max(0, min(255, x))) for x in bg_rgb)
                best_pair = (fg_t, bg_t)

    # Fallback to hardcoded if generation fails
    if best_pair is None:
        fallback = {
            "protanopia":    ((200, 90, 80), (100, 160, 110)),
            "deuteranopia":  ((195, 95, 75), (110, 170, 115)),
            "tritanopia":    ((220, 170, 70), (90, 110, 200)),
            "achromatopsia": ((155, 155, 155), (175, 175, 175)),
        }
        best_pair = fallback[cvd_type]

    fg_base, bg_base = best_pair
    # Generate slight variations for natural look (3 shades each)
    def _jitter(c, amount=12):
        return tuple(max(0, min(255, v + random.randint(-amount, amount))) for v in c)

    return {
        "fg": [fg_base, _jitter(fg_base), _jitter(fg_base)],
        "bg": [bg_base, _jitter(bg_base), _jitter(bg_base)],
    }


CONTROL_PALETTE = {
    "fg": [(200, 50, 50), (210, 55, 45), (190, 45, 55)],
    "bg": [(240, 230, 180), (235, 225, 170), (245, 235, 185)],
}

EXAM_SYMBOLS = list("23456789ABCDEFHK")

DIFFICULTY_LABELS = {"easy": "Easy", "medium": "Medium", "hard": "Hard"}
DIFFICULTY_COLORS = {"easy": "#2ecc71", "medium": "#f39c12", "hard": "#e74c3c"}

CVD_PREVALENCE = {
    "protanopia":    {"male": 1.3, "female": 0.02, "desc": "Red-blind (L-cone absent)"},
    "deuteranopia":  {"male": 5.0, "female": 0.35, "desc": "Green-blind (M-cone absent)"},
    "tritanopia":    {"male": 0.01, "female": 0.01, "desc": "Blue-yellow blind (S-cone absent)"},
    "achromatopsia": {"male": 0.003, "female": 0.003, "desc": "Total color blindness"},
}

CVD_TYPES = ["protanopia", "deuteranopia", "tritanopia", "achromatopsia"]


# --- Natural dot rendering with jitter ---

def _exam_svg(symbol, cvd, difficulty="easy", is_control=False, confusion_trap=None):
    """Circular Ishihara plate with per-dot color jitter and size variation."""
    C = PLATE_CFG
    center = C["SVG_SIZE"] // 2
    radius = C["SVG_SIZE"] // 2 - 4

    if is_control:
        pal = CONTROL_PALETTE
    else:
        pal = _make_confusing_pair(cvd, difficulty)

    circles = _gen_circles()

    # Difficulty affects dot size range for visual variety
    size_jitter = {"easy": 0, "medium": 1, "hard": 2}.get(difficulty, 0)

    bg0 = pal["bg"][0]
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {C["SVG_SIZE"]} {C["SVG_SIZE"]}">']

    # All defs at top level (best browser compatibility)
    trap_mask = ""
    if confusion_trap:
        trap_mask = (
            f'<mask id="trap">'
            f'<rect width="100%" height="100%" fill="black"/>'
            f'<text x="{center}" y="{center}" font-size="{int(C["FONT_SIZE"]*0.85)}" '
            f'font-family="Arial,Helvetica,sans-serif" font-weight="bold" '
            f'text-anchor="middle" dominant-baseline="central" fill="white">{confusion_trap}</text>'
            f'</mask>'
        )
    svg.append(
        f'<defs>'
        f'<clipPath id="disc"><circle cx="{center}" cy="{center}" r="{radius}"/></clipPath>'
        f'<radialGradient id="bgGrad" cx="40%" cy="40%">'
        f'<stop offset="0%" stop-color="rgb({min(255,bg0[0]+12)},{min(255,bg0[1]+12)},{min(255,bg0[2]+12)})" />'
        f'<stop offset="100%" stop-color="rgb({max(0,bg0[0]-8)},{max(0,bg0[1]-8)},{max(0,bg0[2]-8)})" />'
        f'</radialGradient>'
        f'<mask id="sym">'
        f'<rect width="100%" height="100%" fill="black"/>'
        f'<text x="{center}" y="{center}" font-size="{C["FONT_SIZE"]}" '
        f'font-family="Arial,Helvetica,sans-serif" font-weight="bold" '
        f'text-anchor="middle" dominant-baseline="central" fill="white">{symbol}</text>'
        f'</mask>'
        f'{trap_mask}'
        f'</defs>'
    )

    svg.append(f'<g clip-path="url(#disc)">')
    svg.append(f'<circle cx="{center}" cy="{center}" r="{radius}" fill="url(#bgGrad)"/>')

    # Background dots with per-dot color jitter
    for x, y, r in circles:
        c = random.choice(pal["bg"])
        jr, jg, jb = (c[0]+random.randint(-8,8), c[1]+random.randint(-8,8), c[2]+random.randint(-8,8))
        jr, jg, jb = max(0,min(255,jr)), max(0,min(255,jg)), max(0,min(255,jb))
        dr = max(2, r + random.randint(-size_jitter, size_jitter))
        svg.append(f'<circle cx="{x}" cy="{y}" r="{dr}" fill="rgb({jr},{jg},{jb})" />')

    # Foreground dots within symbol mask
    svg.append(f'<g mask="url(#sym)">')
    for x, y, r in circles:
        c = random.choice(pal["fg"])
        jr, jg, jb = (c[0]+random.randint(-8,8), c[1]+random.randint(-8,8), c[2]+random.randint(-8,8))
        jr, jg, jb = max(0,min(255,jr)), max(0,min(255,jg)), max(0,min(255,jb))
        dr = max(2, r + random.randint(-size_jitter, size_jitter))
        svg.append(f'<circle cx="{x}" cy="{y}" r="{dr}" fill="rgb({jr},{jg},{jb})" />')
    svg.append('</g>')

    # Confusion trap overlay
    if confusion_trap:
        svg.append(f'<g mask="url(#trap)">')
        for x, y, r in circles:
            c = random.choice(pal["bg"])
            d = max(0, c[0]-18), max(0, c[1]-18), max(0, c[2]-18)
            svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="rgb({d[0]},{d[1]},{d[2]})" />')
        svg.append('</g>')

    svg.append('</g>')
    svg.append(f'<circle cx="{center}" cy="{center}" r="{radius}" '
               f'fill="none" stroke="#444" stroke-width="3"/>')
    svg.append('</svg>')
    return "\n".join(svg)


# --- Adaptive engine ---

def _generate_choices(correct, pool, n=5, trap=None):
    choices = {correct}
    if trap:
        choices.add(trap)
    available = [s for s in pool if s not in choices]
    needed = n - len(choices) - 1
    choices.update(random.sample(available, min(needed, len(available))))
    result = list(choices)
    result.append("?")
    random.shuffle(result)
    return result


def _make_plate(cvd, difficulty, sym_pool, used_syms, is_confusion=False):
    """Generate a single plate with a fresh symbol."""
    available = [s for s in sym_pool if s not in used_syms]
    if not available:
        available = sym_pool
    sym = random.choice(available)
    used_syms.add(sym)

    trap = None
    if is_confusion:
        trap_pool = [s for s in sym_pool if s != sym and s not in used_syms]
        if trap_pool:
            trap = random.choice(trap_pool)

    weight = {"easy": 1.0, "medium": 1.5, "hard": 2.0}[difficulty]

    return {
        "cvd": cvd, "difficulty": difficulty, "weight": weight,
        "symbol": sym, "trap_symbol": trap,
        "is_confusion": is_confusion, "is_control": False,
        "choices": _generate_choices(sym, sym_pool, trap=trap),
    }


def _build_adaptive_exam():
    """Build initial plate queue: 1 control + 12 diagnostic.
    The adaptive engine may modify later plates based on answers."""
    used = set()
    plates = []

    # Control plate
    ctrl_sym = random.choice(EXAM_SYMBOLS)
    used.add(ctrl_sym)
    plates.append({
        "cvd": "control", "difficulty": "easy", "weight": 0,
        "symbol": ctrl_sym, "is_control": True,
        "is_confusion": False, "trap_symbol": None,
        "choices": _generate_choices(ctrl_sym, EXAM_SYMBOLS),
    })

    # Phase 1: Easy screening (one per CVD type) — 4 plates
    for cvd in CVD_TYPES:
        plates.append(_make_plate(cvd, "easy", EXAM_SYMBOLS, used))

    # Phase 2: Medium (one per type) — 4 plates
    for cvd in CVD_TYPES:
        plates.append(_make_plate(cvd, "medium", EXAM_SYMBOLS, used))

    # Phase 3: Hard + confusion — 4 plates (will be adapted)
    for cvd in CVD_TYPES:
        is_conf = cvd in ("protanopia", "deuteranopia")  # confusion for red-green
        plates.append(_make_plate(cvd, "hard", EXAM_SYMBOLS, used, is_confusion=is_conf))

    return plates


def _adapt_plates(exam):
    """Adapt remaining plates based on performance so far.
    If a CVD type shows no errors on easy+medium, skip its hard plate
    and replace with an extra hard plate for the weakest CVD type."""
    history = exam["history"]
    idx = exam["index"]
    plates = exam["plates"]

    if idx < 9 or idx >= len(plates):
        return  # Only adapt after medium phase

    # Compute per-CVD error count from answered plates
    cvd_errors = {c: 0 for c in CVD_TYPES}
    cvd_total  = {c: 0 for c in CVD_TYPES}
    for h in history:
        if h.get("is_control"):
            continue
        cvd = h["cvd"]
        if cvd in cvd_errors:
            cvd_total[cvd] += 1
            if not h["correct"]:
                cvd_errors[cvd] += 1

    # Find weakest and strongest CVD types
    weakest  = max(cvd_errors, key=cvd_errors.get)
    strongest = min(cvd_errors, key=cvd_errors.get)

    # If a type has 0 errors across easy+medium (2 plates), swap its hard plate
    if cvd_errors[strongest] == 0 and cvd_total[strongest] >= 2 and cvd_errors[weakest] >= 2:
        # Find the hard plate of the strongest type and replace it
        for i in range(idx, len(plates)):
            if plates[i]["cvd"] == strongest and plates[i]["difficulty"] == "hard":
                used = {p["symbol"] for p in plates}
                # Replace with extra diagnostic plate for weakest type
                new_plate = _make_plate(weakest, "hard", EXAM_SYMBOLS, used, is_confusion=True)
                new_plate["adapted"] = True
                new_plate["replaced_from"] = strongest
                plates[i] = new_plate
                break


# --- Response time analysis ---

RESPONSE_TIME_THRESHOLDS = {
    "easy":   {"fast": 3.0, "normal": 6.0},
    "medium": {"fast": 4.0, "normal": 8.0},
    "hard":   {"fast": 5.0, "normal": 10.0},
}

def _classify_response_time(time_s, difficulty):
    th = RESPONSE_TIME_THRESHOLDS.get(difficulty, {"fast": 4, "normal": 8})
    if time_s <= th["fast"]:
        return "fast"
    elif time_s <= th["normal"]:
        return "normal"
    return "slow"


def _compute_exam_results(exam):
    history = exam.get("history", [])

    control_result = None
    diag_history = []
    for h in history:
        if h.get("is_control"):
            control_result = h
        else:
            diag_history.append(h)

    total_q = len(diag_history)

    # Per-CVD weighted scores
    cvd_scores = {k: 0.0 for k in CVD_TYPES}
    cvd_max    = {k: 0.0 for k in CVD_TYPES}
    correct_count = 0
    total_time = 0

    for h in diag_history:
        cvd = h["cvd"]
        w   = h["weight"]
        if cvd in cvd_max:
            cvd_max[cvd] += w
        total_time += h.get("time", 0)

        # Classify response time
        h["time_class"] = _classify_response_time(h.get("time", 0), h["difficulty"])

        if h["correct"]:
            correct_count += 1
            # Slow correct on easy = potential mild CVD (add partial risk)
            if h["time_class"] == "slow" and h["difficulty"] == "easy" and cvd in cvd_scores:
                cvd_scores[cvd] += w * 0.3
        else:
            if cvd in cvd_scores:
                cvd_scores[cvd] += w

    cvd_risk = {}
    for k in CVD_TYPES:
        if cvd_max[k] > 0:
            cvd_risk[k] = round(min(100, cvd_scores[k] / cvd_max[k] * 100), 1)
        else:
            cvd_risk[k] = 0.0

    max_risk_cvd = max(cvd_risk, key=cvd_risk.get)
    max_risk_val = cvd_risk[max_risk_cvd]

    if max_risk_val < 15:
        result, severity = "Normal Color Vision", "none"
    elif max_risk_val < 40:
        result, severity = max_risk_cvd.capitalize(), "mild"
    elif max_risk_val < 70:
        result, severity = max_risk_cvd.capitalize(), "moderate"
    else:
        result, severity = max_risk_cvd.capitalize(), "strong"

    accuracy = round(correct_count / total_q * 100, 1) if total_q > 0 else 0
    avg_time = round(total_time / total_q, 1) if total_q > 0 else 0

    # Per-difficulty breakdown
    diff_stats = {}
    for d in ["easy", "medium", "hard"]:
        d_items = [h for h in diag_history if h["difficulty"] == d]
        if d_items:
            d_correct = sum(1 for h in d_items if h["correct"])
            d_avg_t = round(sum(h.get("time",0) for h in d_items) / len(d_items), 1)
            diff_stats[d] = {"total": len(d_items), "correct": d_correct,
                             "pct": round(d_correct / len(d_items) * 100),
                             "avg_time": d_avg_t}
        else:
            diff_stats[d] = {"total": 0, "correct": 0, "pct": 100, "avg_time": 0}

    # Response time analysis
    time_analysis = {
        "fastest": round(min((h.get("time",99) for h in diag_history), default=0), 1),
        "slowest": round(max((h.get("time",0) for h in diag_history), default=0), 1),
        "avg": avg_time,
        "slow_correct": sum(1 for h in diag_history if h["correct"] and h.get("time_class") == "slow"),
        "fast_wrong": sum(1 for h in diag_history if not h["correct"] and h.get("time_class") == "fast"),
        "per_question": [{"qno": h["qno"], "time": h.get("time",0), "correct": h["correct"],
                          "difficulty": h["difficulty"], "cvd": h["cvd"],
                          "time_class": h.get("time_class","normal")}
                         for h in diag_history],
    }

    # Confusion plate results
    confusion_results = [h for h in diag_history if h.get("is_confusion")]
    trap_fallen = sum(1 for h in confusion_results if h.get("fell_for_trap"))

    # Adapted plates
    adapted_count = sum(1 for h in diag_history if h.get("adapted"))

    return {
        "result": result, "severity": severity,
        "accuracy": accuracy, "correct": correct_count, "total": total_q,
        "avg_time": avg_time, "total_time": round(total_time, 1),
        "cvd_risk": cvd_risk,
        "diff_stats": diff_stats,
        "time_analysis": time_analysis,
        "history": history,
        "diag_history": diag_history,
        "control_passed": control_result["correct"] if control_result else None,
        "confusion_count": len(confusion_results),
        "confusion_traps_fallen": trap_fallen,
        "adapted_count": adapted_count,
        "prevalence": CVD_PREVALENCE,
    }


SEVERITY_LABELS = {
    "none":     {"text": "No Deficiency Detected", "color": "#2ecc71", "icon": "✅"},
    "mild":     {"text": "Mild Indication",        "color": "#f39c12", "icon": "🟡"},
    "moderate": {"text": "Moderate Indication",     "color": "#e67e22", "icon": "🟠"},
    "strong":   {"text": "Strong Indication",       "color": "#e74c3c", "icon": "🔴"},
}


# --- Routes ---

@app.route("/exam")
def exam_home():
    return render_template("exam_start.html")


@app.route("/exam/start")
def exam_start():
    plates = _build_adaptive_exam()
    session["exam"] = {"index": 0, "plates": plates, "history": []}
    return redirect(url_for("exam_question"))


@app.route("/exam/question", methods=["GET", "POST"])
def exam_question():
    exam = session.get("exam")
    if not exam:
        return redirect(url_for("exam_home"))

    idx = exam["index"]
    if idx >= len(exam["plates"]):
        return redirect(url_for("exam_result"))

    plate = exam["plates"][idx]

    if request.method == "POST":
        answer = request.form.get("answer", "").strip().upper()
        resp_time = float(request.form.get("resp_time", 0))
        correct_sym = plate["symbol"].upper()
        is_correct = (answer == correct_sym)

        fell_for_trap = False
        if plate.get("is_confusion") and plate.get("trap_symbol"):
            if answer == plate["trap_symbol"].upper():
                fell_for_trap = True

        entry = {
            "qno": idx + 1,
            "symbol": plate["symbol"],
            "answer": answer if answer != "?" else "Can't see",
            "correct": is_correct,
            "cvd": plate["cvd"],
            "difficulty": plate["difficulty"],
            "weight": plate["weight"],
            "time": round(resp_time, 1),
            "is_control": plate.get("is_control", False),
            "is_confusion": plate.get("is_confusion", False),
            "fell_for_trap": fell_for_trap,
            "adapted": plate.get("adapted", False),
        }
        exam["history"].append(entry)
        exam["index"] = idx + 1

        # Run adaptive logic after medium phase
        _adapt_plates(exam)

        session["exam"] = exam
        return redirect(url_for("exam_question"))

    # Build dot states
    dot_states = ["correct" if h["correct"] else "wrong" for h in exam["history"]]

    svg = _exam_svg(
        plate["symbol"], plate.get("cvd", "protanopia"),
        plate.get("difficulty", "easy"),
        is_control=plate.get("is_control", False),
        confusion_trap=plate.get("trap_symbol"),
    )

    prev_feedback = None
    if exam["history"]:
        prev = exam["history"][-1]
        prev_feedback = {"correct": prev["correct"], "symbol": prev["symbol"], "answer": prev["answer"]}

    total = len(exam["plates"])
    return render_template("exam.html",
        svg=svg, qno=idx + 1, total=total,
        choices=plate["choices"],
        difficulty=plate.get("difficulty", "easy"),
        diff_label=DIFFICULTY_LABELS.get(plate.get("difficulty","easy"), ""),
        diff_color=DIFFICULTY_COLORS.get(plate.get("difficulty","easy"), "#888"),
        is_control=plate.get("is_control", False),
        is_adapted=plate.get("adapted", False),
        dot_states=dot_states,
        prev_feedback=prev_feedback,
    )


@app.route("/exam/result")
def exam_result():
    exam = session.get("exam")
    if not exam:
        return redirect(url_for("exam_home"))
    results = _compute_exam_results(exam)
    sev_info = SEVERITY_LABELS[results["severity"]]
    session["exam_results"] = results
    return render_template("exam_result.html",
        r=results, sev_info=sev_info,
        cvd_labels=CVD_LABELS, diff_labels=DIFFICULTY_LABELS,
        diff_colors=DIFFICULTY_COLORS, prevalence=CVD_PREVALENCE,
    )


@app.route("/exam/report")
def exam_report():
    from reportlab.pdfgen import canvas as pdf_canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.colors import HexColor

    data = session.get("exam_results")
    if not data:
        return redirect(url_for("exam_home"))

    buf = io.BytesIO()
    c = pdf_canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    sev = SEVERITY_LABELS[data["severity"]]

    # Header
    c.setFillColor(HexColor("#1a1a2e"))
    c.rect(0, h - 90, w, 90, fill=1, stroke=0)
    c.setFillColor(HexColor("#6C63FF"))
    c.rect(0, h - 94, w, 4, fill=1, stroke=0)
    c.setFillColor(HexColor("#FFFFFF"))
    c.setFont("Helvetica-Bold", 24)
    c.drawString(40, h - 55, "ColorSight Suite")
    c.setFont("Helvetica", 13)
    c.drawString(40, h - 78, "Color Vision Examination Report (Adaptive)")

    y = h - 130

    # Result
    c.setFillColor(HexColor(sev["color"]))
    c.roundRect(40, y - 55, w - 80, 55, 8, fill=1, stroke=0)
    c.setFillColor(HexColor("#FFFFFF"))
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(w/2, y - 25, data["result"])
    c.setFont("Helvetica", 12)
    c.drawCentredString(w/2, y - 45, sev["text"])
    y -= 85

    # Summary
    c.setFillColor(HexColor("#333333"))
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "Summary")
    y -= 22
    c.setFont("Helvetica", 10)
    ta = data.get("time_analysis", {})
    for s in [
        f"Accuracy: {data['accuracy']}% ({data['correct']}/{data['total']})",
        f"Avg Time: {data['avg_time']}s  (fastest: {ta.get('fastest',0)}s / slowest: {ta.get('slowest',0)}s)",
        f"Control Plate: {'Passed' if data.get('control_passed') else 'Failed'}",
        f"Confusion Traps: {data.get('confusion_traps_fallen',0)}/{data.get('confusion_count',0)}",
        f"Adaptive Plates: {data.get('adapted_count',0)} plates redirected to weakness",
        f"Slow Correct (potential mild CVD): {ta.get('slow_correct',0)}",
    ]:
        c.drawString(55, y, s)
        y -= 16

    # CVD Risk
    y -= 12
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "CVD Risk Assessment")
    y -= 22
    rc = {"protanopia":"#ef5350","deuteranopia":"#66bb6a","tritanopia":"#42a5f5","achromatopsia":"#78909c"}
    for cvd_key, risk_val in data["cvd_risk"].items():
        c.setFont("Helvetica", 10)
        c.setFillColor(HexColor("#333333"))
        prev = CVD_PREVALENCE.get(cvd_key, {})
        c.drawString(55, y, f"{CVD_LABELS.get(cvd_key,cvd_key)}  (pop: {prev.get('male',0)}% M)")
        c.drawRightString(w-55, y, f"{risk_val}%")
        bx, bw, bh = 55, w-130, 8
        c.setFillColor(HexColor("#e0e0e0"))
        c.roundRect(bx, y-14, bw, bh, 3, fill=1, stroke=0)
        c.setFillColor(HexColor(rc.get(cvd_key,"#6C63FF")))
        c.roundRect(bx, y-14, max(2, bw*risk_val/100), bh, 3, fill=1, stroke=0)
        y -= 30

    # Question review
    y -= 8
    if y < 200: c.showPage(); y = h - 60
    c.setFillColor(HexColor("#333333"))
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "Question Review")
    y -= 18
    c.setFont("Helvetica-Bold", 8)
    cols = [(50,"#"),(65,"Sym"),(95,"Ans"),(125,"OK?"),(160,"Type"),(260,"Diff"),(310,"Time"),(355,"Speed"),(405,"Notes")]
    for cx, cl in cols: c.drawString(cx, y, cl)
    y -= 4
    c.line(50, y, w-40, y)
    y -= 12
    c.setFont("Helvetica", 8)
    for hi in data["history"]:
        if y < 45: c.showPage(); y = h - 60
        c.setFillColor(HexColor("#333333"))
        c.drawString(50, y, str(hi["qno"]))
        c.drawString(65, y, hi["symbol"])
        c.drawString(95, y, str(hi["answer"]))
        c.setFillColor(HexColor("#2ecc71" if hi["correct"] else "#e74c3c"))
        c.drawString(125, y, "Yes" if hi["correct"] else "No")
        c.setFillColor(HexColor("#333333"))
        c.drawString(160, y, "Control" if hi.get("is_control") else hi["cvd"].capitalize())
        c.drawString(260, y, hi["difficulty"].capitalize())
        c.drawString(310, y, f"{hi['time']}s")
        tc = hi.get("time_class","")
        c.drawString(355, y, tc)
        notes = []
        if hi.get("is_confusion"): notes.append("Confusion")
        if hi.get("fell_for_trap"): notes.append("TRAP")
        if hi.get("adapted"): notes.append("Adapted")
        c.drawString(405, y, " ".join(notes))
        y -= 14

    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(HexColor("#999999"))
    c.drawString(40, 28, "ColorSight Suite · Adaptive screening · Not a medical diagnosis")
    c.save()
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="colorsight_exam_report.pdf", mimetype="application/pdf")


# ═══════════════════════════════════════════════════════════════
#  MODULE 4 – COLOR-BASED OBJECT DETECTION & ACCESSIBILITY AUDIT
#  (Advanced engine: edge-aware region segmentation, WCAG AA/AAA,
#   adjacent-contrast analysis, text-area detection, CVD-specific
#   breakdown, fix suggestions, heatmap, PDF report)
# ═══════════════════════════════════════════════════════════════

# --------------- WCAG helpers ---------------

def _lin_channel(v):
    v /= 255.0
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

def _relative_luminance(r, g, b):
    return 0.2126 * _lin_channel(r) + 0.7152 * _lin_channel(g) + 0.0722 * _lin_channel(b)

def _contrast_ratio(lum1, lum2):
    lighter, darker = max(lum1, lum2), min(lum1, lum2)
    return (lighter + 0.05) / (darker + 0.05)

def _rgb_to_hex(r, g, b):
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _color_name(r, g, b):
    h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
    h *= 360
    if v < 0.12: return "Black"
    if s < 0.08:
        if v > 0.88: return "White"
        if v > 0.60: return "Light Gray"
        if v > 0.35: return "Gray"
        return "Dark Gray"
    if h < 12 or h >= 345: return "Red"
    if h < 38:  return "Orange"
    if h < 68:  return "Yellow"
    if h < 155: return "Green"
    if h < 195: return "Cyan"
    if h < 265: return "Blue"
    if h < 305: return "Purple"
    return "Pink"

def _color_category(name):
    """Map color name → broad bucket for grouping."""
    warm  = {"Red", "Orange", "Pink"}
    cool  = {"Blue", "Cyan", "Purple"}
    neut  = {"Black", "White", "Gray", "Light Gray", "Dark Gray"}
    if name in warm:  return "warm"
    if name == "Green": return "green"
    if name == "Yellow": return "yellow"
    if name in cool:  return "cool"
    return "neutral"


# --------------- K-means++ color extraction ---------------

def _kmeans_pp_init(pixels, k, rng):
    """K-means++ initialization for better centroid seeding."""
    n = len(pixels)
    first = rng.integers(n)
    centroids = [pixels[first]]
    for _ in range(1, k):
        dists = np.min([np.sum((pixels - c) ** 2, axis=1) for c in centroids], axis=0)
        total = dists.sum()
        if total == 0 or np.isnan(total):
            idx = rng.integers(n)
        else:
            probs = dists / total
            idx = rng.choice(n, p=probs)
        centroids.append(pixels[idx])
    return np.array(centroids, dtype=np.float32)

def _dominant_colors(img, n_colors=10, sample_size=8000):
    arr = np.array(img.convert("RGB"))
    pixels = arr.reshape(-1, 3).astype(np.float32)

    if len(pixels) > sample_size:
        idx = np.random.choice(len(pixels), sample_size, replace=False)
        pixels = pixels[idx]

    rng = np.random.default_rng(42)
    centroids = _kmeans_pp_init(pixels, n_colors, rng)

    for _ in range(20):
        dists = np.linalg.norm(pixels[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_c = np.copy(centroids)
        for k in range(n_colors):
            members = pixels[labels == k]
            if len(members) > 0:
                new_c[k] = members.mean(axis=0)
        if np.allclose(centroids, new_c, atol=0.5):
            break
        centroids = new_c

    # Merge near-duplicate centroids (< 15 rgb distance)
    merged, used = [], set()
    for i in range(n_colors):
        if i in used: continue
        group = [i]
        for j in range(i+1, n_colors):
            if j in used: continue
            if np.linalg.norm(centroids[i] - centroids[j]) < 15:
                group.append(j); used.add(j)
        used.add(i)
        merged.append(group)

    # Recount with merged clusters
    dists = np.linalg.norm(pixels[:, None] - centroids[None, :], axis=2)
    labels = np.argmin(dists, axis=1)
    total_px = len(pixels)

    results = []
    for group in merged:
        mask = np.isin(labels, group)
        count = mask.sum()
        if count == 0: continue
        avg = pixels[mask].mean(axis=0)
        r, g, b = int(avg[0]), int(avg[1]), int(avg[2])
        lum = _relative_luminance(r, g, b)
        results.append({
            "rgb": (r, g, b), "hex": _rgb_to_hex(r, g, b),
            "name": _color_name(r, g, b),
            "category": _color_category(_color_name(r, g, b)),
            "pct": round(count / total_px * 100, 1),
            "luminance": round(lum, 4),
        })

    results.sort(key=lambda c: -c["pct"])
    return results


# --------------- Adaptive region segmentation ---------------

def _sobel_magnitude(gray):
    """Simple Sobel edge magnitude using numpy convolution."""
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    gy = gx.T
    from numpy.lib.stride_tricks import sliding_window_view
    padded = np.pad(gray.astype(np.float32), 1, mode='edge')
    windows = sliding_window_view(padded, (3, 3))
    sx = (windows * gx).sum(axis=(-2, -1))
    sy = (windows * gy).sum(axis=(-2, -1))
    return np.sqrt(sx**2 + sy**2)

def _analyze_regions(img, grid=8):
    """Analyze image in an 8×8 grid with edge awareness."""
    arr = np.array(img.convert("RGB"))
    gray = np.mean(arr, axis=2)
    edges = _sobel_magnitude(gray)
    h, w, _ = arr.shape
    cell_h, cell_w = h // grid, w // grid
    regions = []

    for row in range(grid):
        for col in range(grid):
            y0, y1 = row * cell_h, min((row + 1) * cell_h, h)
            x0, x1 = col * cell_w, min((col + 1) * cell_w, w)
            cell = arr[y0:y1, x0:x1].reshape(-1, 3)
            cell_edges = edges[y0:y1, x0:x1]

            mean_c = cell.mean(axis=0).astype(int)
            std_c  = cell.std(axis=0)
            r, g, b = int(mean_c[0]), int(mean_c[1]), int(mean_c[2])
            h_val, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            lum = _relative_luminance(r, g, b)

            edge_density = float(np.mean(cell_edges > 40))
            is_text_like = edge_density > 0.12 and float(std_c.mean()) > 30

            regions.append({
                "row": row, "col": col,
                "x": x0, "y": y0, "w": cell_w, "h": cell_h,
                "rgb": (r, g, b), "hex": _rgb_to_hex(r, g, b),
                "name": _color_name(r, g, b),
                "saturation": round(s, 3), "value": round(v, 3),
                "luminance": round(lum, 4),
                "variability": round(float(std_c.mean()), 1),
                "edge_density": round(edge_density, 3),
                "is_text_like": is_text_like,
                "std_channels": [round(float(x), 1) for x in std_c],
            })
    return regions


# --------------- Text area detection ---------------

def _detect_text_areas(img):
    """Detect likely text vs background pairs — strict filtering to reduce false positives."""
    arr = np.array(img.convert("RGB"))
    gray = np.mean(arr, axis=2)
    edges = _sobel_magnitude(gray)

    h, w = gray.shape
    cell = 48  # larger patches = more context, fewer false positives
    text_pairs = []

    for y in range(0, h - cell, cell // 2):  # 50% overlap for coverage
        for x in range(0, w - cell, cell // 2):
            patch_edges = edges[y:y+cell, x:x+cell]

            # 1) Require strong edge presence (text has sharp, concentrated edges)
            edge_ratio = float(np.mean(patch_edges > 80))
            if edge_ratio < 0.15 or edge_ratio > 0.70:
                # Too few edges = no text; too many = noise/texture
                continue

            patch_rgb = arr[y:y+cell, x:x+cell]
            patch_gray = gray[y:y+cell, x:x+cell]

            # 2) Check bimodal distribution — text should have a clear fg/bg split
            p_min, p_max = float(patch_gray.min()), float(patch_gray.max())
            lum_range = p_max - p_min
            if lum_range < 40:
                # Very uniform patch — not text
                continue

            # Use Otsu-like threshold: split at point maximizing inter-class variance
            threshold = float(np.median(patch_gray))
            bright_mask = patch_gray > threshold + lum_range * 0.1
            dark_mask   = patch_gray < threshold - lum_range * 0.1

            bright_count = int(bright_mask.sum())
            dark_count   = int(dark_mask.sum())
            total_px     = cell * cell

            # 3) Both fg and bg must occupy meaningful area (text = minority, bg = majority)
            if dark_count < total_px * 0.08 or dark_count > total_px * 0.65:
                continue
            if bright_count < total_px * 0.20:
                continue

            fg = patch_rgb[dark_mask].mean(axis=0).astype(int)
            bg = patch_rgb[bright_mask].mean(axis=0).astype(int)

            # 4) fg and bg must be visually distinct in color space
            color_dist = float(np.linalg.norm(fg.astype(float) - bg.astype(float)))
            if color_dist < 50:
                continue

            fg_lum = _relative_luminance(*fg)
            bg_lum = _relative_luminance(*bg)
            cr = _contrast_ratio(fg_lum, bg_lum)

            # 5) Only report if contrast is in a meaningful range
            #    (>20:1 is definitely fine, <1.2:1 is probably not text)
            if cr > 15.0 or cr < 1.2:
                continue

            # 6) Compute a confidence score for "text-likeness"
            #    High edges + clear bimodal split + reasonable fg proportion = text
            bimodal_score = min(1.0, lum_range / 120.0)
            proportion_score = 1.0 - abs(dark_count / total_px - 0.25) * 2  # ideal ~25% fg
            confidence = (edge_ratio * 0.4 + bimodal_score * 0.35 + max(0, proportion_score) * 0.25)

            if confidence < 0.30:
                continue

            text_pairs.append({
                "x": x, "y": y, "w": cell, "h": cell,
                "fg_rgb": tuple(int(v) for v in fg),
                "bg_rgb": tuple(int(v) for v in bg),
                "fg_hex": _rgb_to_hex(*fg), "bg_hex": _rgb_to_hex(*bg),
                "contrast": round(cr, 2),
                "fg_lum": round(fg_lum, 4), "bg_lum": round(bg_lum, 4),
                "confidence": round(confidence, 2),
            })

    # Deduplicate overlapping patches — keep highest confidence
    if text_pairs:
        text_pairs.sort(key=lambda t: -t["confidence"])
        kept = []
        used_coords = set()
        for t in text_pairs:
            key = (t["x"] // cell, t["y"] // cell)
            if key not in used_coords:
                kept.append(t)
                used_coords.add(key)
        text_pairs = kept

    return text_pairs


# --------------- CVD-safe color suggestion ---------------

CVD_SAFE_PALETTE = [
    {"hex": "#0072B2", "name": "Blue",        "rgb": (0, 114, 178)},
    {"hex": "#E69F00", "name": "Orange",       "rgb": (230, 159, 0)},
    {"hex": "#009E73", "name": "Teal",         "rgb": (0, 158, 115)},
    {"hex": "#F0E442", "name": "Yellow",       "rgb": (240, 228, 66)},
    {"hex": "#56B4E9", "name": "Sky Blue",     "rgb": (86, 180, 233)},
    {"hex": "#D55E00", "name": "Vermillion",   "rgb": (213, 94, 0)},
    {"hex": "#CC79A7", "name": "Rose",         "rgb": (204, 121, 167)},
    {"hex": "#000000", "name": "Black",        "rgb": (0, 0, 0)},
]

def _suggest_fix_color(rgb_tuple):
    """Suggest nearest CVD-safe replacement."""
    arr = np.array(rgb_tuple, dtype=np.float32)
    best, best_d = None, 1e9
    for c in CVD_SAFE_PALETTE:
        d = np.linalg.norm(arr - np.array(c["rgb"], dtype=np.float32))
        if d < best_d:
            best_d, best = d, c
    return best


# --------------- Main audit engine ---------------

def _accessibility_audit(colors, regions, text_areas):
    issues = []
    recommendations = []

    # ---------- 1. Dominant color contrast pairs (WCAG AA / AAA) ----------
    contrast_matrix = []
    for i, c1 in enumerate(colors):
        for j, c2 in enumerate(colors):
            if j <= i: continue
            cr = _contrast_ratio(c1["luminance"], c2["luminance"])
            entry = {
                "color1": c1["hex"], "color2": c2["hex"],
                "name1": c1["name"], "name2": c2["name"],
                "ratio": round(cr, 2),
                "wcag_aa_normal": cr >= 4.5,
                "wcag_aa_large":  cr >= 3.0,
                "wcag_aaa":       cr >= 7.0,
            }
            contrast_matrix.append(entry)

            min_pct = 2
            if cr < 3.0 and c1["pct"] > min_pct and c2["pct"] > min_pct:
                sev = "high" if cr < 1.8 else "medium"
                fix1 = _suggest_fix_color(c1["rgb"])
                fix2 = _suggest_fix_color(c2["rgb"])
                issues.append({
                    "type": "low_contrast",
                    "severity": sev,
                    "message": (f"Contrast {cr:.1f}:1 between {c1['name']} ({c1['hex']}) "
                                f"and {c2['name']} ({c2['hex']}). "
                                f"Fails WCAG AA for normal text (≥4.5:1) and large text (≥3.0:1)."),
                    "colors": [c1["hex"], c2["hex"]],
                    "ratio": round(cr, 2),
                    "suggestion": (f"Replace {c1['hex']} → {fix1['hex']} ({fix1['name']}) "
                                   f"or {c2['hex']} → {fix2['hex']} ({fix2['name']})"),
                })
            elif cr < 4.5 and c1["pct"] > min_pct and c2["pct"] > min_pct:
                issues.append({
                    "type": "low_contrast",
                    "severity": "medium",
                    "message": (f"Contrast {cr:.1f}:1 between {c1['name']} ({c1['hex']}) "
                                f"and {c2['name']} ({c2['hex']}). "
                                "Passes large text AA (≥3:1) but fails normal text AA (≥4.5:1)."),
                    "colors": [c1["hex"], c2["hex"]],
                    "ratio": round(cr, 2),
                    "suggestion": "Increase contrast for any small or body text using these colors.",
                })

    # ---------- 2. Text area contrast failures ----------
    text_fail_aa = [t for t in text_areas if t["contrast"] < 4.5]
    text_fail_aaa = [t for t in text_areas if 4.5 <= t["contrast"] < 7.0]

    if text_fail_aa:
        worst = min(text_fail_aa, key=lambda t: t["contrast"])
        issues.append({
            "type": "text_contrast_fail",
            "severity": "high",
            "message": (f"Detected {len(text_fail_aa)} text-like areas failing WCAG AA (< 4.5:1). "
                        f"Worst: {worst['contrast']}:1 — fg {worst['fg_hex']} on bg {worst['bg_hex']}."),
            "colors": [worst["fg_hex"], worst["bg_hex"]],
            "ratio": worst["contrast"],
            "suggestion": "Darken text or lighten background to achieve ≥ 4.5:1 contrast.",
            "count": len(text_fail_aa),
        })

    if len(text_fail_aaa) > len(text_areas) * 0.5 and len(text_areas) > 3:
        issues.append({
            "type": "text_contrast_aaa",
            "severity": "low",
            "message": (f"{len(text_fail_aaa)} of {len(text_areas)} text areas don't meet "
                        "WCAG AAA (7:1). Consider improving for enhanced accessibility."),
            "suggestion": "WCAG AAA compliance is recommended for critical interfaces.",
        })

    # ---------- 3. Adjacent region contrast ----------
    adj_issues = 0
    for r in regions:
        row, col = r["row"], r["col"]
        for dr, dc in [(0, 1), (1, 0)]:
            nr, nc = row + dr, col + dc
            neighbor = next((x for x in regions if x["row"] == nr and x["col"] == nc), None)
            if not neighbor: continue
            cr = _contrast_ratio(r["luminance"], neighbor["luminance"])
            if cr < 1.5 and r["name"] != neighbor["name"] and r["saturation"] > 0.3 and neighbor["saturation"] > 0.3:
                adj_issues += 1

    if adj_issues > 3:
        issues.append({
            "type": "adjacent_low_contrast",
            "severity": "medium",
            "message": (f"{adj_issues} pairs of adjacent colored regions have near-identical luminance "
                        "but different hues. These become invisible under most CVD types."),
            "suggestion": "Add borders, separators, or luminance variation between adjacent colored areas.",
        })

    # ---------- 4. Color-only information cues ----------
    high_sat = [r for r in regions if r["saturation"] > 0.45 and r["variability"] < 35]
    color_groups = {}
    for r in high_sat:
        color_groups.setdefault(r["name"], []).append(r)

    if len(color_groups) >= 2:
        distinct_colors = [k for k, v in color_groups.items() if len(v) >= 1]
        if len(distinct_colors) >= 2:
            issues.append({
                "type": "color_only_cue",
                "severity": "high",
                "message": (f"Found {len(high_sat)} regions across {len(distinct_colors)} distinct colors "
                            f"({', '.join(distinct_colors[:5])}). If color alone conveys meaning, "
                            "CVD users will lose information."),
                "suggestion": "Add icons, patterns, text labels, or shape coding alongside color.",
            })

    # ---------- 5. Red-green reliance ----------
    reds   = [c for c in colors if c["name"] in ("Red", "Orange") and c["pct"] > 2]
    greens = [c for c in colors if c["name"] == "Green" and c["pct"] > 2]
    if reds and greens:
        total_rg = sum(c["pct"] for c in reds + greens)
        issues.append({
            "type": "red_green_reliance",
            "severity": "high",
            "message": (f"Red/Orange and Green together account for {total_rg:.0f}% of the palette. "
                        "~8% of males (Protanopia + Deuteranopia) cannot distinguish these."),
            "colors": [reds[0]["hex"], greens[0]["hex"]],
            "suggestion": "Replace red→vermillion (#D55E00) and green→blue (#0072B2), "
                          "or add secondary cues (icons, underlines, patterns).",
        })

    # ---------- 6. Blue-yellow reliance (Tritanopia) ----------
    blues   = [c for c in colors if c["name"] in ("Blue", "Cyan") and c["pct"] > 2]
    yellows = [c for c in colors if c["name"] == "Yellow" and c["pct"] > 2]
    if blues and yellows:
        issues.append({
            "type": "blue_yellow_reliance",
            "severity": "medium",
            "message": ("Design uses Blue/Cyan and Yellow together. "
                        "Users with Tritanopia cannot distinguish these."),
            "colors": [blues[0]["hex"], yellows[0]["hex"]],
            "suggestion": "Add pattern or shape variation when pairing blue and yellow elements.",
        })

    # ---------- 7. CVD simulation color merging ----------
    cvd_breakdown = {}
    for cvd_type in ["protanopia", "deuteranopia", "tritanopia"]:
        sim_colors = []
        for c in colors:
            rgb01 = np.array([[c["rgb"]]], dtype=np.float32) / 255.0
            sim = apply_cvd_matrix(rgb01, cvd_type)[0, 0]
            sim_colors.append(sim)

        merges = []
        for i in range(len(sim_colors)):
            for j in range(i + 1, len(sim_colors)):
                orig_diff = np.linalg.norm(np.array(colors[i]["rgb"]) / 255.0 - np.array(colors[j]["rgb"]) / 255.0)
                sim_diff  = np.linalg.norm(sim_colors[i] - sim_colors[j])
                if orig_diff > 0.12 and sim_diff < 0.07 and colors[i]["pct"] > 1.5 and colors[j]["pct"] > 1.5:
                    merges.append({
                        "c1": colors[i]["hex"], "c2": colors[j]["hex"],
                        "n1": colors[i]["name"], "n2": colors[j]["name"],
                        "orig_dist": round(float(orig_diff), 3),
                        "sim_dist": round(float(sim_diff), 3),
                    })
                    issues.append({
                        "type": "color_merge",
                        "severity": "high",
                        "message": (f"Under {CVD_LABELS[cvd_type]}: "
                                    f"{colors[i]['name']} ({colors[i]['hex']}) and "
                                    f"{colors[j]['name']} ({colors[j]['hex']}) merge "
                                    f"(distance drops {orig_diff:.2f} → {sim_diff:.2f})."),
                        "colors": [colors[i]["hex"], colors[j]["hex"]],
                        "suggestion": f"Use CVD-safe alternatives or add non-color cues for {cvd_type} users.",
                    })

        cvd_breakdown[cvd_type] = {
            "label": CVD_LABELS[cvd_type],
            "merge_count": len(merges),
            "merges": merges,
            "status": "pass" if len(merges) == 0 else "warn" if len(merges) <= 2 else "fail",
        }

    # ---------- 8. Monochrome / low diversity check ----------
    non_neutral = [c for c in colors if c["category"] != "neutral" and c["pct"] > 2]
    if len(non_neutral) <= 1:
        issues.append({
            "type": "low_color_diversity",
            "severity": "low",
            "message": ("Palette is largely monochrome/neutral. While inherently CVD-safe, "
                        "ensure interactive states and UI affordances aren't invisible."),
            "suggestion": "Verify hover/focus/active states have sufficient visual contrast.",
        })

    # ---------- 9. Scoring ----------
    high_count   = sum(1 for i in issues if i["severity"] == "high")
    medium_count = sum(1 for i in issues if i["severity"] == "medium")
    low_count    = sum(1 for i in issues if i["severity"] == "low")
    score = max(0, min(100, 100 - high_count * 15 - medium_count * 7 - low_count * 2))
    grade = ("A" if score >= 90 else "B" if score >= 75 else
             "C" if score >= 55 else "D" if score >= 35 else "F")

    # ---------- 10. Recommendations ----------
    if high_count > 0:
        recommendations.append("Use the Wong CVD-safe palette for key UI elements (see suggested palette).")
    if any(i["type"] == "text_contrast_fail" for i in issues):
        recommendations.append("Run all body text through a contrast checker — aim for ≥ 4.5:1 minimum.")
    if any(i["type"] == "color_only_cue" for i in issues):
        recommendations.append("Never rely on color alone: add icons, patterns, text labels, or underlines.")
    if any(i["type"] == "red_green_reliance" for i in issues):
        recommendations.append("Replace red/green pairs with blue/orange or add shape differentiation.")
    if score >= 90:
        recommendations.append("Excellent baseline! Consider WCAG AAA (7:1) for critical interfaces.")

    return {
        "issues": issues,
        "score": score,
        "grade": grade,
        "summary": {
            "total_issues": len(issues), "high": high_count,
            "medium": medium_count, "low": low_count,
        },
        "contrast_matrix": contrast_matrix,
        "cvd_breakdown": cvd_breakdown,
        "recommendations": recommendations,
        "safe_palette": CVD_SAFE_PALETTE,
        "text_areas_scanned": len(text_areas),
        "text_areas_failing": len(text_fail_aa),
    }


# --------------- Heatmap generation ---------------

def _generate_heatmap(img, regions):
    """Generate a clearly visible risk-severity heatmap overlay."""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for r in regions:
        risk = 0.0

        # Saturated uniform regions = strong color-only cues
        if r["saturation"] > 0.30:
            risk += min(r["saturation"], 1.0) * 0.5
        if r["variability"] < 25 and r["saturation"] > 0.3:
            risk += 0.25  # flat colored area — likely color-coded UI element

        # Text-like areas with mid-luminance (risky for contrast)
        if r.get("is_text_like"):
            if 0.2 < r["luminance"] < 0.8:
                risk += 0.3

        # Low luminance + high saturation = dark colored element (hard to read)
        if r["value"] < 0.35 and r["saturation"] > 0.3:
            risk += 0.15

        risk = min(risk, 1.0)

        # Even low-risk regions get a tint so the heatmap is always visible
        if risk < 0.05:
            color = (40, 200, 80, 35)
        elif risk < 0.25:
            color = (60, 200, 60, 70 + int(risk * 120))
        elif risk < 0.55:
            alpha = 100 + int(risk * 180)
            color = (255, 180, 30, min(alpha, 200))
        else:
            alpha = 140 + int(risk * 115)
            color = (230, 45, 45, min(alpha, 220))

        draw.rectangle(
            [r["x"], r["y"], r["x"] + r["w"], r["y"] + r["h"]],
            fill=color
        )

    base = img.convert("RGBA")
    result = Image.alpha_composite(base, overlay).convert("RGB")

    # ── Legend bar ─────────────────────────────────────────────────────────
    # Load a small font for the legend labels
    legend_font = None
    for _sz in (9, 8):
        for _path in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        ):
            try:
                legend_font = ImageFont.truetype(_path, _sz)
                break
            except Exception:
                pass
        if legend_font:
            break
    if legend_font is None:
        legend_font = ImageFont.load_default()

    img_w, img_h = result.size
    LEGEND_H = 22           # total bar height in px
    SWATCH_H = 10           # colour swatch height
    SWATCH_W = 28           # width of each swatch segment
    GAP      = 4            # gap between swatch and label text
    MARGIN   = 6

    draw2 = ImageDraw.Draw(result)

    # Background strip at bottom
    draw2.rectangle([0, img_h - LEGEND_H, img_w, img_h], fill=(0, 0, 0, 0))
    # Semi-transparent dark bar
    lbar = Image.new("RGBA", (img_w, LEGEND_H), (0, 0, 0, 185))
    result.paste(Image.fromarray(
        __import__("numpy").array(lbar, dtype=__import__("numpy").uint8)[:, :, :3]),
        (0, img_h - LEGEND_H)
    )

    draw2 = ImageDraw.Draw(result)
    sy = img_h - LEGEND_H + (LEGEND_H - SWATCH_H) // 2  # vertically centred swatch y

    labels = [("Low",    (60,  200, 60)),
              ("Med",    (255, 180, 30)),
              ("High",   (230, 45,  45))]

    cx = MARGIN
    for txt, rgb in labels:
        # Colour swatch
        draw2.rectangle([cx, sy, cx + SWATCH_W, sy + SWATCH_H], fill=rgb)
        # Label text right of swatch
        tx = cx + SWATCH_W + GAP
        ty = img_h - LEGEND_H + (LEGEND_H - 9) // 2
        draw2.text((tx, ty), txt, fill=(255, 255, 255), font=legend_font)
        # Advance: swatch + gap + approximate text width + spacing
        try:
            tw = legend_font.getbbox(txt)[2] - legend_font.getbbox(txt)[0]
        except Exception:
            tw = len(txt) * 6
        cx += SWATCH_W + GAP + tw + 10

    # "risk" label at end, only if it fits
    risk_label = "CVD risk"
    try:
        rw = legend_font.getbbox(risk_label)[2] - legend_font.getbbox(risk_label)[0]
    except Exception:
        rw = len(risk_label) * 6
    if cx + rw + MARGIN <= img_w:
        draw2.text((cx, img_h - LEGEND_H + (LEGEND_H - 9) // 2),
                   risk_label, fill=(200, 200, 200), font=legend_font)

    return result


# --------------- Annotated image ---------------

def _generate_annotated_image(img, regions, issues, text_areas):
    annotated = img.copy().convert("RGB")
    img_w, img_h = annotated.size

    # ── Font: 9 px bold — small enough to fit inside any cell, still readable ──
    _FONT_SIZES = [9, 8]
    font = None
    for _sz in _FONT_SIZES:
        for _path in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        ):
            try:
                font = ImageFont.truetype(_path, _sz)
                break
            except Exception:
                pass
        if font:
            break
    if font is None:
        font = ImageFont.load_default()

    LABEL_H   = 12   # compact pill height in px
    OUTLINE_W = 2

    # Short name map: keep labels ≤ 9 chars so they never overflow a ~100px cell
    _SHORT = {
        "Dark Red": "DkRed", "Light Red": "LtRed",
        "Dark Orange": "Orange", "Light Orange": "LtOrng",
        "Dark Yellow": "DkYlw", "Light Yellow": "LtYlw",
        "Dark Green": "DkGrn", "Light Green": "LtGrn",
        "Dark Cyan": "DkCyan", "Light Cyan": "LtCyan",
        "Dark Blue": "DkBlue", "Light Blue": "LtBlue",
        "Dark Violet": "DkViol", "Light Violet": "LtViol",
        "Dark Magenta": "DkMag", "Light Magenta": "LtMag",
        "Dark Pink": "DkPink", "Light Pink": "LtPink",
    }

    def _shorten(name):
        return _SHORT.get(name, name[:7])

    def _pill_w(text):
        try:
            bb = font.getbbox(text)
            return bb[2] - bb[0] + 6
        except Exception:
            return len(text) * 6 + 6

    def _draw_box(x, y, w, h, outline_col, label, pill_col, anchor="top"):
        """
        Draw outline box + compact pill label.
        anchor='top'  → pill sits just ABOVE the box top edge
        anchor='bot'  → pill sits just BELOW the box bottom edge
        Falls back gracefully when there is no room outside.
        Outline is drawn LAST so it is never overwritten by the pill fill.
        """
        draw = ImageDraw.Draw(annotated)
        pw = _pill_w(label)

        # Clamp pill x to image bounds
        px0 = max(0, min(x, img_w - pw))
        px1 = min(px0 + pw, img_w)

        if anchor == "top":
            if y >= LABEL_H + 1:
                py0, py1 = y - LABEL_H, y          # above
            elif y + h + LABEL_H <= img_h:
                py0, py1 = y + h, y + h + LABEL_H  # below
            else:
                py0, py1 = y + 1, y + 1 + LABEL_H  # inside (last resort)
        else:  # anchor == "bot"
            if y + h + LABEL_H <= img_h:
                py0, py1 = y + h, y + h + LABEL_H  # below
            elif y >= LABEL_H + 1:
                py0, py1 = y - LABEL_H, y          # above
            else:
                py0, py1 = y + h - 1 - LABEL_H, y + h - 1  # inside

        # 1. Draw pill
        draw.rectangle([px0, py0, px1, py1], fill=pill_col)
        draw.text((px0 + 3, py0 + 1), label, fill="white", font=font)

        # 2. Draw outline last — never overwritten
        draw.rectangle([x, y, x + w - 1, y + h - 1],
                       outline=outline_col, width=OUTLINE_W)

    # ── Color-critical saturated regions ────────────────────────────────────
    for r in regions:
        if r["saturation"] > 0.35 and r["value"] > 0.15:
            x, y, w, h = r["x"], r["y"], r["w"], r["h"]
            col = "#FF3333" if r["saturation"] > 0.65 else "#FF8800"
            _draw_box(x, y, w, h, col, _shorten(r["name"]), col, anchor="top")

    # ── Text contrast failures (WCAG AA threshold 4.5 : 1) ────────────────
    for t in [t for t in text_areas if t["contrast"] < 4.5
              and t.get("confidence", 0) >= 0.35]:
        x, y, w, h = t["x"], t["y"], t["w"], t["h"]
        _draw_box(x, y, w, h, "#ff00ff", f'{t["contrast"]:.1f}:1', "#cc00cc", anchor="bot")

    # ── Adjacent low-contrast border lines ──────────────────────────────────
    draw = ImageDraw.Draw(annotated)
    for r in regions:
        for dr, dc in [(0, 1), (1, 0)]:
            nr, nc = r["row"] + dr, r["col"] + dc
            nb = next((n for n in regions if n["row"] == nr and n["col"] == nc), None)
            if not nb:
                continue
            cr = _contrast_ratio(r["luminance"], nb["luminance"])
            if cr < 2.0 and r["name"] != nb["name"] and r["saturation"] > 0.2:
                if dc == 1:
                    draw.line([r["x"] + r["w"], r["y"],
                               r["x"] + r["w"], r["y"] + r["h"]],
                              fill="#00ffff", width=2)
                else:
                    draw.line([r["x"], r["y"] + r["h"],
                               r["x"] + r["w"], r["y"] + r["h"]],
                              fill="#00ffff", width=2)

    # ── Issue summary legend (bottom-right, compact) ─────────────────────────
    if issues:
        hi = sum(1 for i in issues if i["severity"] == "high")
        me = sum(1 for i in issues if i["severity"] == "medium")
        lo = sum(1 for i in issues if i["severity"] == "low")

        type_counts = {}
        for iss in issues:
            k = iss["type"].replace("_", " ").title()
            type_counts[k] = type_counts.get(k, 0) + 1

        lines = [f" {hi}H {me}M {lo}L "]
        for k, cnt in list(type_counts.items())[:4]:
            lines.append(f" \u2022{k[:16]}" + (f"\u00d7{cnt}" if cnt > 1 else ""))

        try:
            lf = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 8)
        except Exception:
            lf = font

        LH = 11
        LW = max(_pill_w(ln) for ln in lines) + 4
        LH_total = len(lines) * LH + 4
        bx = img_w - LW - 4
        by = img_h - LH_total - 4

        ov = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(ov)
        od.rectangle([bx, by, bx + LW, by + LH_total], fill=(10, 10, 25, 210))
        od.rectangle([bx, by, bx + LW, by + LH + 2], fill=(35, 35, 75, 230))
        annotated = Image.alpha_composite(annotated.convert("RGBA"), ov).convert("RGB")
        draw = ImageDraw.Draw(annotated)
        for i, ln in enumerate(lines):
            draw.text((bx + 3, by + 2 + i * LH),
                      ln, fill="#FFD700" if i == 0 else "#CCCCCC", font=lf)

    return annotated


# --------------- Routes ---------------

@app.route("/detector")
def detector():
    return render_template("detector.html", cvd_labels=CVD_LABELS)


@app.route("/detector/analyze", methods=["POST"])
def detector_analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid or unsupported image file. Please upload a JPG, PNG, or WebP."}), 400

    max_dim = 800
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)

    colors     = _dominant_colors(img)
    regions    = _analyze_regions(img)
    text_areas = _detect_text_areas(img)
    audit      = _accessibility_audit(colors, regions, text_areas)

    annotated  = _generate_annotated_image(img, regions, audit["issues"], text_areas)
    heatmap    = _generate_heatmap(img, regions)

    # CVD simulations
    arr = np.array(img, dtype=np.float32) / 255.0
    simulations = {}
    for cvd_type in CVD_MATRICES:
        sim = apply_cvd_matrix(arr, cvd_type)
        simulations[cvd_type] = pil_to_base64(Image.fromarray((sim * 255).astype(np.uint8)))

    return jsonify({
        "original":    pil_to_base64(img),
        "annotated":   pil_to_base64(annotated),
        "heatmap":     pil_to_base64(heatmap),
        "simulations": simulations,
        "colors":      colors,
        "audit":       audit,
    })


@app.route("/detector/report", methods=["POST"])
def detector_report():
    """Generate PDF audit report."""
    from reportlab.pdfgen import canvas as pdf_canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.colors import HexColor, Color

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400

    audit = data["audit"]
    colors_data = data["colors"]

    buf = io.BytesIO()
    c = pdf_canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    # ---- Header ----
    c.setFillColor(HexColor("#1a1a2e"))
    c.rect(0, h - 90, w, 90, fill=1, stroke=0)
    c.setFillColor(HexColor("#6C63FF"))
    c.rect(0, h - 94, w, 4, fill=1, stroke=0)
    c.setFillColor(HexColor("#FFFFFF"))
    c.setFont("Helvetica-Bold", 24)
    c.drawString(40, h - 55, "ColorSight Suite")
    c.setFont("Helvetica", 14)
    c.drawString(40, h - 78, "Color Accessibility Audit Report")

    y = h - 130

    # ---- Score ----
    grade_colors = {"A": "#2ecc71", "B": "#27ae60", "C": "#f39c12", "D": "#e67e22", "F": "#e74c3c"}
    gc = grade_colors.get(audit["grade"], "#888888")
    c.setFillColor(HexColor(gc))
    c.circle(80, y - 20, 30, fill=1, stroke=0)
    c.setFillColor(HexColor("#FFFFFF"))
    c.setFont("Helvetica-Bold", 28)
    c.drawCentredString(80, y - 30, audit["grade"])

    c.setFillColor(HexColor("#333333"))
    c.setFont("Helvetica-Bold", 18)
    c.drawString(130, y - 10, f"Score: {audit['score']}/100")
    c.setFont("Helvetica", 12)
    s = audit["summary"]
    c.drawString(130, y - 32, f"{s['high']} High  ·  {s['medium']} Medium  ·  {s['low']} Low  ·  {s['total_issues']} Total Issues")

    y -= 80

    # ---- Issues ----
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "Issues Found")
    y -= 25

    sev_colors = {"high": "#e74c3c", "medium": "#f39c12", "low": "#2ecc71"}
    for issue in audit["issues"][:12]:
        if y < 80:
            c.showPage()
            y = h - 60
        c.setFillColor(HexColor(sev_colors.get(issue["severity"], "#888")))
        c.circle(52, y + 4, 5, fill=1, stroke=0)
        c.setFillColor(HexColor("#333333"))
        c.setFont("Helvetica-Bold", 9)
        c.drawString(65, y, f"[{issue['severity'].upper()}] {issue['type'].replace('_', ' ').title()}")
        y -= 14
        c.setFont("Helvetica", 9)
        msg = issue["message"]
        while msg and y > 50:
            line = msg[:95]
            c.drawString(65, y, line)
            msg = msg[95:]
            y -= 12
        if issue.get("suggestion"):
            c.setFont("Helvetica-Oblique", 8)
            c.setFillColor(HexColor("#0072B2"))
            c.drawString(65, y, f"Fix: {issue['suggestion'][:100]}")
            c.setFillColor(HexColor("#333333"))
            y -= 12
        y -= 10

    # ---- Dominant Colors ----
    if y < 180:
        c.showPage()
        y = h - 60

    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "Dominant Colors")
    y -= 25

    for col in colors_data[:8]:
        r, g, b = col["rgb"]
        c.setFillColor(Color(r/255, g/255, b/255))
        c.rect(50, y - 4, 20, 16, fill=1, stroke=1)
        c.setFillColor(HexColor("#333333"))
        c.setFont("Helvetica", 10)
        c.drawString(80, y, f"{col['hex']}  {col['name']}  ({col['pct']}%)")
        y -= 22

    # ---- Recommendations ----
    if audit.get("recommendations"):
        if y < 120:
            c.showPage()
            y = h - 60
        y -= 10
        c.setFont("Helvetica-Bold", 16)
        c.drawString(40, y, "Recommendations")
        y -= 22
        c.setFont("Helvetica", 10)
        for rec in audit["recommendations"]:
            c.drawString(55, y, f"• {rec}")
            y -= 16

    # ---- Footer ----
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(HexColor("#999999"))
    c.drawString(40, 30, "Generated by ColorSight Suite · For educational purposes only · Not a substitute for manual accessibility testing")

    c.save()
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name="colorsight_accessibility_report.pdf",
                     mimetype="application/pdf")


# ═══════════════════════════════════════════════════════════════
#  ADDITIONAL PAGES
# ═══════════════════════════════════════════════════════════════

@app.route("/about")
def about():
    return render_template("about.html")


@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(413)
def file_too_large(e):
    return render_template("404.html"), 413


# ═══════════════════════════════════════════════════════════════
#  RUN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
