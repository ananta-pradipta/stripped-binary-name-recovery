#!/usr/bin/env python3
"""Generate FuncR Architecture Diagrams PowerPoint presentation."""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import copy

# ── Constants ──────────────────────────────────────────────────────────
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

# Color palette
C_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
C_BLACK = RGBColor(0x00, 0x00, 0x00)
C_DARK = RGBColor(0x2D, 0x2D, 0x2D)
C_GRAY_BG = RGBColor(0xF5, 0xF5, 0xF5)
C_LIGHT_GRAY = RGBColor(0xE0, 0xE0, 0xE0)
C_MED_GRAY = RGBColor(0x99, 0x99, 0x99)

# Themed colors
C_BLUE = RGBColor(0x1A, 0x73, 0xE8)        # Pipeline / titles
C_GREEN = RGBColor(0x0F, 0x9D, 0x58)       # Block encoder
C_GREEN_LIGHT = RGBColor(0xE8, 0xF5, 0xE9)
C_PURPLE = RGBColor(0x7B, 0x1F, 0xA2)      # Graph encoder
C_PURPLE_LIGHT = RGBColor(0xF3, 0xE5, 0xF5)
C_ORANGE = RGBColor(0xF5, 0x7C, 0x00)      # External encoder
C_ORANGE_LIGHT = RGBColor(0xFF, 0xF3, 0xE0)
C_TEAL = RGBColor(0x00, 0x89, 0x7B)        # Gated fusion
C_TEAL_LIGHT = RGBColor(0xE0, 0xF2, 0xF1)
C_RED = RGBColor(0xD9, 0x3B, 0x25)         # Decoder
C_RED_LIGHT = RGBColor(0xFB, 0xE9, 0xE7)
C_INDIGO = RGBColor(0x30, 0x3F, 0x9F)      # k-NN
C_INDIGO_LIGHT = RGBColor(0xE8, 0xEA, 0xF6)
C_AMBER = RGBColor(0xFF, 0x8F, 0x00)       # Pretraining
C_AMBER_LIGHT = RGBColor(0xFF, 0xF8, 0xE1)
C_CYAN = RGBColor(0x00, 0x97, 0xA7)        # Evaluation
C_CYAN_LIGHT = RGBColor(0xE0, 0xF7, 0xFA)
C_BROWN = RGBColor(0x5D, 0x40, 0x37)       # Preprocessing
C_BROWN_LIGHT = RGBColor(0xEF, 0xEB, 0xE9)

FONT = "Calibri"


def set_font(run, size=10, bold=False, italic=False, color=C_DARK, name=FONT):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = name


def add_text_box(slide, left, top, width, height, text, font_size=10,
                 bold=False, color=C_DARK, align=PP_ALIGN.LEFT, italic=False,
                 anchor=MSO_ANCHOR.TOP):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    set_font(run, font_size, bold, italic, color)
    return txBox


def add_rounded_box(slide, left, top, width, height, fill_color, border_color=None,
                    text="", font_size=10, bold=False, font_color=C_DARK,
                    align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, line_width=Pt(1.5)):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = line_width
    else:
        shape.line.fill.background()
    # Rounded corner adjustment
    if hasattr(shape, 'adjustments') and len(shape.adjustments) > 0:
        shape.adjustments[0] = 0.1
    if text:
        tf = shape.text_frame
        tf.word_wrap = True
        tf.auto_size = None
        p = tf.paragraphs[0]
        p.alignment = align
        p.space_before = Pt(0)
        p.space_after = Pt(0)
        run = p.add_run()
        run.text = text
        set_font(run, font_size, bold, color=font_color)
    return shape


def add_multiline_box(slide, left, top, width, height, lines, fill_color,
                      border_color=None, default_size=9, line_width=Pt(1.5)):
    """Add rounded box with multiple formatted lines.
    lines: list of (text, size, bold, color) tuples
    """
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = line_width
    else:
        shape.line.fill.background()
    if hasattr(shape, 'adjustments') and len(shape.adjustments) > 0:
        shape.adjustments[0] = 0.08

    tf = shape.text_frame
    tf.word_wrap = True
    tf.auto_size = None

    for i, line_info in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.space_before = Pt(1)
        p.space_after = Pt(1)
        p.alignment = PP_ALIGN.CENTER

        if isinstance(line_info, str):
            run = p.add_run()
            run.text = line_info
            set_font(run, default_size)
        else:
            text, size, bold, color = line_info
            run = p.add_run()
            run.text = text
            set_font(run, size, bold, color=color)
    return shape


def add_arrow(slide, start_left, start_top, end_left, end_top, color=C_MED_GRAY,
              width=Pt(2), dashed=False):
    """Add a connector arrow from (start) to (end)."""
    connector = slide.shapes.add_connector(
        1,  # straight connector
        start_left, start_top, end_left, end_top
    )
    connector.line.color.rgb = color
    connector.line.width = width
    if dashed:
        connector.line.dash_style = 4  # dash
    # Add arrowhead at end
    connector.end_x = end_left
    connector.end_y = end_top
    return connector


def add_arrow_shape(slide, left, top, width, height, color=C_MED_GRAY, direction='right'):
    """Add arrow shape."""
    if direction == 'right':
        shape_type = MSO_SHAPE.RIGHT_ARROW
    elif direction == 'down':
        shape_type = MSO_SHAPE.DOWN_ARROW
    else:
        shape_type = MSO_SHAPE.RIGHT_ARROW
    shape = slide.shapes.add_shape(shape_type, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


def add_slide_number(slide, num):
    add_text_box(slide, Inches(12.5), Inches(7.1), Inches(0.7), Inches(0.3),
                 str(num), font_size=9, color=C_MED_GRAY, align=PP_ALIGN.RIGHT)


def add_slide_title(slide, title, color=C_BLUE):
    add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.5),
                 title, font_size=22, bold=True, color=color)


# ══════════════════════════════════════════════════════════════════════
# SLIDE 1: High-Level Pipeline Overview
# ══════════════════════════════════════════════════════════════════════
def build_slide1(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    add_slide_title(slide, "FuncRecover: High-Level Pipeline Overview")

    # ── Phase 1: Acquisition & Preprocessing ──
    y = Inches(0.85)
    add_rounded_box(slide, Inches(0.3), y, Inches(12.7), Inches(1.75),
                    C_BROWN_LIGHT, C_BROWN, line_width=Pt(2))
    add_text_box(slide, Inches(0.5), y + Inches(0.05), Inches(5), Inches(0.35),
                 "1  Acquisition, Preprocessing & CFG Extraction",
                 font_size=13, bold=True, color=C_BROWN)

    # Flow boxes
    boxes_1 = [
        ("Source\nPackages\n(40 pkgs)", C_BROWN_LIGHT, C_BROWN),
        ("Compile\n(O0, O2)\n363 binaries", C_BROWN_LIGHT, C_BROWN),
        ("BAP 2.5\nIR Lifting\n(.bir files)", C_BROWN_LIGHT, C_BROWN),
        ("V3 Tokenization\n1,510 types\n(semantic)", C_BROWN_LIGHT, C_BROWN),
        ("CFG + Labels\n+ External Calls\n87,724 functions", C_BROWN_LIGHT, C_BROWN),
    ]
    bw = Inches(2.1)
    gap = Inches(0.35)
    x_start = Inches(0.6)
    by = y + Inches(0.45)
    for i, (txt, fill, border) in enumerate(boxes_1):
        bx = x_start + i * (bw + gap)
        add_rounded_box(slide, bx, by, bw, Inches(1.1), fill, border, txt,
                        font_size=10, bold=False, font_color=C_DARK)
        if i < len(boxes_1) - 1:
            add_arrow_shape(slide, bx + bw + Inches(0.05), by + Inches(0.4),
                           Inches(0.25), Inches(0.3), C_BROWN)

    # ── Phase 2: Encoder + Multi-Context Fusion ──
    y2 = Inches(2.8)
    add_rounded_box(slide, Inches(0.3), y2, Inches(12.7), Inches(2.4),
                    C_TEAL_LIGHT, C_TEAL, line_width=Pt(2))
    add_text_box(slide, Inches(0.5), y2 + Inches(0.05), Inches(8), Inches(0.35),
                 "2  Encoder + Multi-Context Fusion (25M params)",
                 font_size=13, bold=True, color=C_TEAL)

    # Stage 1
    s1x = Inches(0.6)
    s1y = y2 + Inches(0.5)
    add_rounded_box(slide, s1x, s1y, Inches(2.8), Inches(0.85),
                    C_GREEN_LIGHT, C_GREEN,
                    "Stage 1: Block Encoder\nTransformer 4L/8H/256d\nb_i per block",
                    font_size=9, bold=False, font_color=C_DARK)
    add_text_box(slide, s1x + Inches(0.2), s1y - Inches(0.02), Inches(1.5), Inches(0.25),
                 "STAGE 1", font_size=8, bold=True, color=C_GREEN)

    # Arrow 1→2
    add_arrow_shape(slide, s1x + Inches(2.85), s1y + Inches(0.3),
                   Inches(0.3), Inches(0.25), C_MED_GRAY)

    # Stage 2
    s2x = Inches(3.8)
    add_rounded_box(slide, s2x, s1y, Inches(2.8), Inches(0.85),
                    C_PURPLE_LIGHT, C_PURPLE,
                    "Stage 2: Graph Encoder\nGAT 3L/8H + Attn Pooling\nf \u2208 R^1024",
                    font_size=9, bold=False, font_color=C_DARK)
    add_text_box(slide, s2x + Inches(0.2), s1y - Inches(0.02), Inches(1.5), Inches(0.25),
                 "STAGE 2", font_size=8, bold=True, color=C_PURPLE)

    # Arrow 2→fusion
    add_arrow_shape(slide, s2x + Inches(2.85), s1y + Inches(0.3),
                   Inches(0.3), Inches(0.25), C_MED_GRAY)

    # Cascaded Gated Fusion
    fx = Inches(7.0)
    add_rounded_box(slide, fx, s1y - Inches(0.15), Inches(5.7), Inches(1.1),
                    C_TEAL_LIGHT, C_TEAL, line_width=Pt(2))
    add_text_box(slide, fx + Inches(0.1), s1y - Inches(0.12), Inches(4), Inches(0.25),
                 "Cascaded Gated Fusion", font_size=10, bold=True, color=C_TEAL)

    gate_y = s1y + Inches(0.2)
    gate_w = Inches(1.55)
    add_rounded_box(slide, fx + Inches(0.15), gate_y, gate_w, Inches(0.6),
                    C_ORANGE_LIGHT, C_ORANGE,
                    "Gate 1\nExt Calls", font_size=8, font_color=C_DARK)
    add_arrow_shape(slide, fx + Inches(1.75), gate_y + Inches(0.2),
                   Inches(0.2), Inches(0.2), C_MED_GRAY)
    add_rounded_box(slide, fx + Inches(2.0), gate_y, gate_w, Inches(0.6),
                    C_ORANGE_LIGHT, C_ORANGE,
                    "Gate 2\nCallees", font_size=8, font_color=C_DARK)
    add_arrow_shape(slide, fx + Inches(3.6), gate_y + Inches(0.2),
                   Inches(0.2), Inches(0.2), C_MED_GRAY)
    add_rounded_box(slide, fx + Inches(3.85), gate_y, gate_w, Inches(0.6),
                    C_ORANGE_LIGHT, C_ORANGE,
                    "Gate 3\nCallers", font_size=8, font_color=C_DARK)

    # Bypass arrow (dashed)
    bypass_y = s1y + Inches(0.95)
    add_text_box(slide, fx + Inches(0.5), bypass_y + Inches(0.05), Inches(4.5), Inches(0.25),
                 "--- conditional bypass: if no context, z = f (skip gate) --->",
                 font_size=7, italic=True, color=C_TEAL)

    # Output z
    add_arrow_shape(slide, fx + Inches(5.45), gate_y + Inches(0.2),
                   Inches(0.2), Inches(0.2), C_TEAL)
    add_text_box(slide, fx + Inches(4.9), gate_y + Inches(0.55), Inches(0.8), Inches(0.25),
                 "z \u2208 R^1024", font_size=8, bold=True, color=C_TEAL)

    # ── Phase 3: Inference & Evaluation ──
    y3 = Inches(5.35)
    add_rounded_box(slide, Inches(0.3), y3, Inches(12.7), Inches(1.85),
                    C_RED_LIGHT, C_RED, line_width=Pt(2))
    add_text_box(slide, Inches(0.5), y3 + Inches(0.05), Inches(5), Inches(0.35),
                 "3  Inference & Evaluation",
                 font_size=13, bold=True, color=C_RED)

    # Decoder branch
    dec_x = Inches(0.8)
    dec_y = y3 + Inches(0.5)
    add_text_box(slide, dec_x - Inches(0.1), dec_y - Inches(0.05), Inches(0.8), Inches(0.3),
                 "z \u2192", font_size=11, bold=True, color=C_RED)
    add_rounded_box(slide, dec_x + Inches(0.5), dec_y, Inches(2.5), Inches(0.65),
                    C_RED_LIGHT, C_RED,
                    "GRU Decoder\nBeam Search k=5\n\u2192 Function Name",
                    font_size=9, font_color=C_DARK)

    # k-NN branch
    add_text_box(slide, dec_x - Inches(0.1), dec_y + Inches(0.7), Inches(0.8), Inches(0.3),
                 "z \u2192", font_size=11, bold=True, color=C_INDIGO)
    add_rounded_box(slide, dec_x + Inches(0.5), dec_y + Inches(0.7), Inches(2.5), Inches(0.55),
                    C_INDIGO_LIGHT, C_INDIGO,
                    "k-NN Retrieval\nk=1, cosine sim \u2192 Name",
                    font_size=9, font_color=C_DARK)

    # Metrics
    metrics_x = Inches(4.5)
    add_rounded_box(slide, metrics_x, dec_y, Inches(4.0), Inches(1.25),
                    C_CYAN_LIGHT, C_CYAN,
                    "Evaluation Metrics\n\nSub-token F1  |  Exact Match\nEdit Similarity  |  N-gram Similarity",
                    font_size=10, font_color=C_DARK)

    # Best results
    res_x = Inches(9.0)
    add_multiline_box(slide, res_x, dec_y, Inches(3.8), Inches(1.25),
                      [("Best Results (Exp 36 + k-NN)", 10, True, C_BLUE),
                       ("Test:  EM=76.3%  F1=0.804", 9, False, C_DARK),
                       ("Demo: EM=55.1%  F1=0.625", 9, False, C_DARK),
                       ("Diffutils: 69.9% EM (unseen)", 9, False, C_DARK),
                       ("87,724 training functions", 8, False, C_MED_GRAY)],
                      C_GRAY_BG, C_BLUE)

    add_slide_number(slide, 1)


# ══════════════════════════════════════════════════════════════════════
# SLIDE 2: Block Encoder (Stage 1)
# ══════════════════════════════════════════════════════════════════════
def build_slide2(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_title(slide, "Stage 1: Block Encoder", C_GREEN)

    cx = Inches(3.5)  # center column
    cw = Inches(6.0)

    # Input example
    y = Inches(1.0)
    add_rounded_box(slide, Inches(0.5), y, Inches(12.3), Inches(1.1),
                    C_GREEN_LIGHT, C_GREEN, line_width=Pt(1.5))
    add_text_box(slide, Inches(0.7), y + Inches(0.05), Inches(3), Inches(0.3),
                 "Input: V3 instruction-type tokens (one basic block)",
                 font_size=11, bold=True, color=C_GREEN)
    # Token examples
    tokens = ["CALL_malloc", "MEM_READ_32", "ARITH_ADD", "STACK_STORE_64", "RETURN"]
    tx = Inches(0.8)
    for tok in tokens:
        tw = Inches(1.9) if "STACK" in tok else Inches(1.7)
        add_rounded_box(slide, tx, y + Inches(0.45), tw, Inches(0.5),
                        C_WHITE, C_GREEN, tok, font_size=9, font_color=C_GREEN)
        tx += tw + Inches(0.15)

    # Down arrow
    add_arrow_shape(slide, cx + Inches(2.5), y + Inches(1.15), Inches(0.35), Inches(0.4),
                   C_GREEN, 'down')

    # Embedding layer
    y2 = Inches(2.55)
    add_rounded_box(slide, cx, y2, cw, Inches(0.65),
                    C_GREEN_LIGHT, C_GREEN,
                    "Token Embedding: 2,279 x 256d  +  Sinusoidal Positional Encoding",
                    font_size=11, font_color=C_DARK)

    add_arrow_shape(slide, cx + Inches(2.5), y2 + Inches(0.7), Inches(0.35), Inches(0.4),
                   C_GREEN, 'down')

    # Transformer
    y3 = Inches(3.7)
    add_rounded_box(slide, cx, y3, cw, Inches(0.85),
                    C_GREEN_LIGHT, C_GREEN,
                    "Transformer Encoder\n4 Layers  |  8 Attention Heads  |  FFN dim=512  |  Dropout=0.15",
                    font_size=11, font_color=C_DARK)

    add_arrow_shape(slide, cx + Inches(2.5), y3 + Inches(0.9), Inches(0.35), Inches(0.4),
                   C_GREEN, 'down')

    # Mean pooling
    y4 = Inches(5.0)
    add_rounded_box(slide, cx, y4, cw, Inches(0.55),
                    C_GREEN_LIGHT, C_GREEN,
                    "Mean Pooling over all tokens in block",
                    font_size=11, font_color=C_DARK)

    add_arrow_shape(slide, cx + Inches(2.5), y4 + Inches(0.6), Inches(0.35), Inches(0.4),
                   C_GREEN, 'down')

    # Output
    y5 = Inches(6.1)
    add_rounded_box(slide, cx + Inches(1.5), y5, Inches(3.0), Inches(0.6),
                    C_GREEN, None,
                    "b_i \u2208 R^512", font_size=14, bold=True, font_color=C_WHITE)

    # Note
    add_text_box(slide, Inches(0.5), Inches(6.85), Inches(12), Inches(0.4),
                 "One embedding per basic block, regardless of token count. "
                 "Pretrained via MLM (mask 15% of tokens, predict them).",
                 font_size=10, italic=True, color=C_MED_GRAY, align=PP_ALIGN.CENTER)

    add_slide_number(slide, 2)


# ══════════════════════════════════════════════════════════════════════
# SLIDE 3: Graph Encoder (Stage 2)
# ══════════════════════════════════════════════════════════════════════
def build_slide3(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_title(slide, "Stage 2: Graph Encoder (GAT over CFG)", C_PURPLE)

    # Left: CFG diagram
    add_text_box(slide, Inches(0.5), Inches(0.9), Inches(3.5), Inches(0.35),
                 "Control Flow Graph (CFG)", font_size=12, bold=True, color=C_PURPLE)

    # CFG nodes
    nodes = {
        'b1': (Inches(2.0), Inches(1.5)),
        'b2': (Inches(1.0), Inches(2.7)),
        'b3': (Inches(3.0), Inches(2.7)),
        'b4': (Inches(1.0), Inches(3.9)),
        'b5': (Inches(3.0), Inches(3.9)),
    }
    node_labels = {
        'b1': 'b1: COMPARE',
        'b2': 'b2: ARITH_ADD',
        'b3': 'b3: CALL_malloc',
        'b4': 'b4: MEM_WRITE',
        'b5': 'b5: RETURN',
    }
    nw, nh = Inches(1.5), Inches(0.5)
    for name, (nx, ny) in nodes.items():
        add_rounded_box(slide, nx, ny, nw, nh, C_PURPLE_LIGHT, C_PURPLE,
                        node_labels[name], font_size=8, font_color=C_PURPLE)

    # Edges (approximate with lines)
    edges = [('b1', 'b2'), ('b1', 'b3'), ('b2', 'b4'), ('b3', 'b4'), ('b3', 'b5')]
    for src, dst in edges:
        sx, sy = nodes[src]
        dx, dy = nodes[dst]
        add_arrow(slide,
                  sx + nw // 2, sy + nh,
                  dx + nw // 2, dy,
                  C_PURPLE, Pt(1.5))

    # Center: GAT layers
    gat_x = Inches(4.8)
    add_text_box(slide, gat_x, Inches(0.9), Inches(4), Inches(0.35),
                 "Graph Attention Network", font_size=12, bold=True, color=C_PURPLE)

    gat_y = Inches(1.5)
    for i, label in enumerate(["GAT Layer 1 (8 heads)", "GAT Layer 2 (8 heads)", "GAT Layer 3 (8 heads)"]):
        add_rounded_box(slide, gat_x, gat_y + i * Inches(0.9),
                        Inches(3.5), Inches(0.6),
                        C_PURPLE_LIGHT, C_PURPLE, label,
                        font_size=10, font_color=C_DARK)
        if i < 2:
            add_arrow_shape(slide, gat_x + Inches(1.5),
                           gat_y + Inches(0.65) + i * Inches(0.9),
                           Inches(0.3), Inches(0.2), C_PURPLE, 'down')

    # Arrow to attention pooling
    add_arrow_shape(slide, gat_x + Inches(1.5), gat_y + Inches(2.85),
                   Inches(0.3), Inches(0.3), C_PURPLE, 'down')

    # Attention Pooling
    ap_y = Inches(4.65)
    add_rounded_box(slide, gat_x - Inches(0.2), ap_y, Inches(3.9), Inches(0.85),
                    C_PURPLE_LIGHT, C_PURPLE, line_width=Pt(2))
    add_text_box(slide, gat_x, ap_y + Inches(0.05), Inches(3.5), Inches(0.25),
                 "Attention Pooling", font_size=11, bold=True, color=C_PURPLE,
                 align=PP_ALIGN.CENTER)
    add_text_box(slide, gat_x, ap_y + Inches(0.3), Inches(3.5), Inches(0.5),
                 "s_i = MLP(b'_i),  \u03b1_i = softmax(s_i)\nf = \u03a3 \u03b1_i \u00b7 b'_i",
                 font_size=9, color=C_DARK, align=PP_ALIGN.CENTER)

    # Right: Attention weights
    aw_x = Inches(9.2)
    add_text_box(slide, aw_x, Inches(0.9), Inches(3.8), Inches(0.35),
                 "Learned Attention Weights", font_size=12, bold=True, color=C_PURPLE)

    weights = [
        ("b'3 (CALL_malloc)", "\u03b1 = 0.42", C_PURPLE),
        ("b'5 (RETURN)", "\u03b1 = 0.28", C_PURPLE),
        ("b'1 (COMPARE)", "\u03b1 = 0.15", C_MED_GRAY),
        ("b'4 (MEM_WRITE)", "\u03b1 = 0.09", C_MED_GRAY),
        ("b'2 (ARITH_ADD)", "\u03b1 = 0.06", C_MED_GRAY),
    ]
    for i, (node, weight, col) in enumerate(weights):
        wy = Inches(1.5) + i * Inches(0.65)
        # Bar chart visual
        alpha_val = float(weight.split("= ")[1])
        bar_w = Inches(alpha_val * 5.0)
        add_rounded_box(slide, aw_x, wy, bar_w, Inches(0.45),
                        C_PURPLE_LIGHT if col == C_PURPLE else C_GRAY_BG,
                        col, f"{node}  {weight}",
                        font_size=8, font_color=col)

    # Output
    add_arrow_shape(slide, gat_x + Inches(1.5), ap_y + Inches(0.9),
                   Inches(0.3), Inches(0.3), C_PURPLE, 'down')

    out_y = Inches(5.95)
    add_rounded_box(slide, gat_x + Inches(0.5), out_y, Inches(2.5), Inches(0.55),
                    C_PURPLE, None,
                    "f \u2208 R^1024", font_size=14, bold=True, font_color=C_WHITE)

    # Note
    add_text_box(slide, Inches(0.5), Inches(6.7), Inches(12), Inches(0.5),
                 "GAT learns which blocks are most informative for naming. "
                 "Blocks with distinctive calls (malloc, printf) receive higher attention.",
                 font_size=10, italic=True, color=C_MED_GRAY, align=PP_ALIGN.CENTER)

    add_slide_number(slide, 3)


# ══════════════════════════════════════════════════════════════════════
# SLIDE 4: Multi-Context Gated Fusion (Key Innovation)
# ══════════════════════════════════════════════════════════════════════
def build_slide4(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_title(slide, "Multi-Context Gated Fusion (Key Innovation)", C_TEAL)

    # Input from Stage 2
    input_y = Inches(0.85)
    add_rounded_box(slide, Inches(0.5), input_y, Inches(2.2), Inches(0.5),
                    C_PURPLE, None,
                    "From Stage 2: f \u2208 R^1024",
                    font_size=10, bold=True, font_color=C_WHITE)

    # ── STAGE A: External Calls ──
    sa_y = Inches(1.65)
    add_rounded_box(slide, Inches(0.3), sa_y, Inches(12.7), Inches(1.55),
                    C_ORANGE_LIGHT, C_ORANGE, line_width=Pt(2))
    add_text_box(slide, Inches(0.5), sa_y + Inches(0.03), Inches(4), Inches(0.3),
                 "STAGE A: External Call Fusion", font_size=11, bold=True, color=C_ORANGE)

    sa_inner_y = sa_y + Inches(0.35)
    add_rounded_box(slide, Inches(0.6), sa_inner_y, Inches(2.5), Inches(0.55),
                    C_WHITE, C_ORANGE,
                    "PLT calls: malloc, printf,\nsocket, bind, ...",
                    font_size=8, font_color=C_DARK)
    add_arrow_shape(slide, Inches(3.15), sa_inner_y + Inches(0.15),
                   Inches(0.25), Inches(0.2), C_ORANGE)
    add_rounded_box(slide, Inches(3.5), sa_inner_y, Inches(2.8), Inches(0.55),
                    C_WHITE, C_ORANGE,
                    "Embed(656, 256) \u2192 Bi-GRU\n\u2192 c_ext \u2208 R^1024",
                    font_size=8, font_color=C_DARK)
    add_arrow_shape(slide, Inches(6.35), sa_inner_y + Inches(0.15),
                   Inches(0.25), Inches(0.2), C_ORANGE)
    add_rounded_box(slide, Inches(6.7), sa_inner_y, Inches(3.3), Inches(0.55),
                    C_ORANGE_LIGHT, C_ORANGE,
                    "g = \u03c3(W[f; c_ext])\nz = g\u2299f + (1-g)\u2299c_ext",
                    font_size=9, bold=False, font_color=C_DARK)

    # Bypass
    add_rounded_box(slide, Inches(10.3), sa_inner_y, Inches(2.5), Inches(0.55),
                    C_GRAY_BG, C_MED_GRAY,
                    "BYPASS: no ext calls\n\u2192 z = f",
                    font_size=8, font_color=C_MED_GRAY)

    # Down arrow
    add_arrow_shape(slide, Inches(6.0), sa_y + Inches(1.55), Inches(0.3), Inches(0.25),
                   C_TEAL, 'down')

    # ── STAGE B: Callee Context ──
    sb_y = Inches(3.5)
    add_rounded_box(slide, Inches(0.3), sb_y, Inches(12.7), Inches(1.45),
                    C_TEAL_LIGHT, C_TEAL, line_width=Pt(2))
    add_text_box(slide, Inches(0.5), sb_y + Inches(0.03), Inches(4), Inches(0.3),
                 "STAGE B: Callee Context Fusion", font_size=11, bold=True, color=C_TEAL)

    sb_inner_y = sb_y + Inches(0.35)
    add_rounded_box(slide, Inches(0.6), sb_inner_y, Inches(2.5), Inches(0.55),
                    C_WHITE, C_TEAL,
                    "Token signatures of top-5\ninternal callees (10 tok each)",
                    font_size=8, font_color=C_DARK)
    add_arrow_shape(slide, Inches(3.15), sb_inner_y + Inches(0.15),
                   Inches(0.25), Inches(0.2), C_TEAL)
    add_rounded_box(slide, Inches(3.5), sb_inner_y, Inches(2.8), Inches(0.55),
                    C_WHITE, C_TEAL,
                    "Embed(2279, 128) \u2192 Bi-GRU\n\u2192 c_callee \u2208 R^1024",
                    font_size=8, font_color=C_DARK)
    add_arrow_shape(slide, Inches(6.35), sb_inner_y + Inches(0.15),
                   Inches(0.25), Inches(0.2), C_TEAL)
    add_rounded_box(slide, Inches(6.7), sb_inner_y, Inches(3.3), Inches(0.55),
                    C_TEAL_LIGHT, C_TEAL,
                    "g = \u03c3(W[z; c_callee])\nz = g\u2299z + (1-g)\u2299c_callee",
                    font_size=9, font_color=C_DARK)
    add_rounded_box(slide, Inches(10.3), sb_inner_y, Inches(2.5), Inches(0.55),
                    C_GRAY_BG, C_MED_GRAY,
                    "BYPASS: no callees\n\u2192 z unchanged",
                    font_size=8, font_color=C_MED_GRAY)

    # Down arrow
    add_arrow_shape(slide, Inches(6.0), sb_y + Inches(1.45), Inches(0.3), Inches(0.25),
                   C_TEAL, 'down')

    # ── STAGE C: Caller Context ──
    sc_y = Inches(5.25)
    add_rounded_box(slide, Inches(0.3), sc_y, Inches(12.7), Inches(1.45),
                    C_TEAL_LIGHT, C_TEAL, line_width=Pt(2))
    add_text_box(slide, Inches(0.5), sc_y + Inches(0.03), Inches(4), Inches(0.3),
                 "STAGE C: Caller Context Fusion", font_size=11, bold=True, color=C_TEAL)

    sc_inner_y = sc_y + Inches(0.35)
    add_rounded_box(slide, Inches(0.6), sc_inner_y, Inches(2.5), Inches(0.55),
                    C_WHITE, C_TEAL,
                    "Token signatures of top-5\ncallers (10 tok each)",
                    font_size=8, font_color=C_DARK)
    add_arrow_shape(slide, Inches(3.15), sc_inner_y + Inches(0.15),
                   Inches(0.25), Inches(0.2), C_TEAL)
    add_rounded_box(slide, Inches(3.5), sc_inner_y, Inches(2.8), Inches(0.55),
                    C_WHITE, C_TEAL,
                    "Same arch as callee encoder\n\u2192 c_caller \u2208 R^1024",
                    font_size=8, font_color=C_DARK)
    add_arrow_shape(slide, Inches(6.35), sc_inner_y + Inches(0.15),
                   Inches(0.25), Inches(0.2), C_TEAL)
    add_rounded_box(slide, Inches(6.7), sc_inner_y, Inches(3.3), Inches(0.55),
                    C_TEAL_LIGHT, C_TEAL,
                    "g = \u03c3(W[z; c_caller])\nz = g\u2299z + (1-g)\u2299c_caller",
                    font_size=9, font_color=C_DARK)
    add_rounded_box(slide, Inches(10.3), sc_inner_y, Inches(2.5), Inches(0.55),
                    C_GRAY_BG, C_MED_GRAY,
                    "BYPASS: no callers\n\u2192 z unchanged",
                    font_size=8, font_color=C_MED_GRAY)

    # Output
    out_y = Inches(6.85)
    add_rounded_box(slide, Inches(4.5), out_y, Inches(2.5), Inches(0.5),
                    C_TEAL, None,
                    "z \u2208 R^1024 \u2192 Decoder / k-NN",
                    font_size=11, bold=True, font_color=C_WHITE)

    # Critical note
    add_rounded_box(slide, Inches(7.5), out_y - Inches(0.05), Inches(5.3), Inches(0.6),
                    RGBColor(0xFF, 0xEB, 0xEE), C_RED,
                    "Conditional bypass is CRITICAL:\nWithout it, 47% of predictions collapse to a single name",
                    font_size=9, bold=True, font_color=C_RED, line_width=Pt(2))

    add_slide_number(slide, 4)


# ══════════════════════════════════════════════════════════════════════
# SLIDE 5: Decoder + k-NN Retrieval
# ══════════════════════════════════════════════════════════════════════
def build_slide5(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_title(slide, "Inference: GRU Decoder + k-NN Retrieval", C_RED)

    # Input z
    add_rounded_box(slide, Inches(5.5), Inches(0.85), Inches(2.5), Inches(0.5),
                    C_TEAL, None,
                    "z \u2208 R^1024", font_size=12, bold=True, font_color=C_WHITE)

    # ── Left branch: GRU Decoder ──
    left_x = Inches(0.4)
    add_rounded_box(slide, left_x, Inches(1.6), Inches(6.0), Inches(5.0),
                    C_RED_LIGHT, C_RED, line_width=Pt(2))
    add_text_box(slide, left_x + Inches(0.2), Inches(1.65), Inches(4), Inches(0.35),
                 "GRU Decoder (Autoregressive)", font_size=13, bold=True, color=C_RED)

    # Init
    dy = Inches(2.15)
    add_rounded_box(slide, left_x + Inches(0.3), dy, Inches(5.3), Inches(0.5),
                    C_WHITE, C_RED,
                    "h_0 = W_init \u00b7 z   (initialize hidden state from encoder)",
                    font_size=9, font_color=C_DARK)

    # Autoregressive example
    dy2 = Inches(2.85)
    add_text_box(slide, left_x + Inches(0.3), dy2, Inches(5), Inches(0.3),
                 "Autoregressive generation example:", font_size=10, bold=True, color=C_RED)

    seq_y = Inches(3.25)
    toks = [("<SOS>", C_MED_GRAY), ("x", C_RED), ("malloc", C_RED), ("<EOS>", C_MED_GRAY)]
    tx = left_x + Inches(0.4)
    for i, (tok, col) in enumerate(toks):
        tw = Inches(1.0)
        add_rounded_box(slide, tx, seq_y, tw, Inches(0.45),
                        C_WHITE, col, tok, font_size=10, bold=True, font_color=col)
        if i < len(toks) - 1:
            add_arrow_shape(slide, tx + tw + Inches(0.02), seq_y + Inches(0.12),
                           Inches(0.2), Inches(0.18), C_RED)
        tx += tw + Inches(0.25)

    add_text_box(slide, left_x + Inches(0.4), seq_y + Inches(0.55), Inches(5), Inches(0.3),
                 '\u2192  produces "xmalloc"', font_size=10, bold=True, color=C_RED)

    # Training details
    dy3 = Inches(4.15)
    add_multiline_box(slide, left_x + Inches(0.3), dy3, Inches(5.3), Inches(1.1),
                      [("Training", 10, True, C_RED),
                       ("Teacher forcing: 1.0 \u2192 0.3 (scheduled)", 9, False, C_DARK),
                       ("Votes tokenizer: 2,642 sub-tokens", 9, False, C_DARK),
                       ("Label smoothing: 0.1", 9, False, C_DARK)],
                      C_WHITE, C_RED)

    # Inference details
    dy4 = Inches(5.4)
    add_multiline_box(slide, left_x + Inches(0.3), dy4, Inches(5.3), Inches(0.85),
                      [("Inference", 10, True, C_RED),
                       ("Beam search k=5 + repetition penalty", 9, False, C_DARK),
                       ("Length-normalized scoring", 9, False, C_DARK)],
                      C_WHITE, C_RED)

    # ── Right branch: k-NN ──
    right_x = Inches(6.85)
    add_rounded_box(slide, right_x, Inches(1.6), Inches(6.1), Inches(5.0),
                    C_INDIGO_LIGHT, C_INDIGO, line_width=Pt(2))
    add_text_box(slide, right_x + Inches(0.2), Inches(1.65), Inches(4), Inches(0.35),
                 "k-NN Retrieval (NEW)", font_size=13, bold=True, color=C_INDIGO)

    ky = Inches(2.15)
    add_rounded_box(slide, right_x + Inches(0.3), ky, Inches(5.4), Inches(0.5),
                    C_WHITE, C_INDIGO,
                    "Build FAISS index: 87,724 training embeddings",
                    font_size=10, font_color=C_DARK)

    add_arrow_shape(slide, right_x + Inches(2.5), ky + Inches(0.55),
                   Inches(0.3), Inches(0.25), C_INDIGO, 'down')

    ky2 = Inches(2.95)
    add_rounded_box(slide, right_x + Inches(0.3), ky2, Inches(5.4), Inches(0.5),
                    C_WHITE, C_INDIGO,
                    "Query: cosine similarity \u2192 nearest neighbor's name",
                    font_size=10, font_color=C_DARK)

    add_arrow_shape(slide, right_x + Inches(2.5), ky2 + Inches(0.55),
                   Inches(0.3), Inches(0.25), C_INDIGO, 'down')

    ky3 = Inches(3.75)
    add_multiline_box(slide, right_x + Inches(0.3), ky3, Inches(5.4), Inches(1.5),
                      [("Key Findings", 11, True, C_INDIGO),
                       ("k=1 is optimal (k>1 majority vote adds noise)", 9, False, C_DARK),
                       ("87% of predictions come from k-NN", 9, False, C_DARK),
                       ("at threshold t = -0.02", 9, False, C_DARK),
                       ("", 6, False, C_DARK),
                       ("k-NN outperforms decoder on both test & demo", 9, True, C_INDIGO)],
                      C_WHITE, C_INDIGO)

    # Results comparison
    ky4 = Inches(5.5)
    add_multiline_box(slide, right_x + Inches(0.3), ky4, Inches(5.4), Inches(1.0),
                      [("Improvement over Decoder-Only", 10, True, C_INDIGO),
                       ("Test EM: 74.3% \u2192 76.3% (+2.0pp)", 9, False, C_DARK),
                       ("Demo EM: 48.5% \u2192 55.1% (+6.6pp)", 9, False, C_DARK),
                       ("No retraining required!", 9, True, C_INDIGO)],
                      C_WHITE, C_INDIGO)

    # Bottom note
    add_text_box(slide, Inches(0.5), Inches(6.85), Inches(12), Inches(0.4),
                 "The k-NN result confirms our model is a pure recognizer: "
                 "retrieval outperforms generation because correct predictions are always names seen in training.",
                 font_size=10, italic=True, color=C_MED_GRAY, align=PP_ALIGN.CENTER)

    add_slide_number(slide, 5)


# ══════════════════════════════════════════════════════════════════════
# SLIDE 6: Self-Supervised Pretraining
# ══════════════════════════════════════════════════════════════════════
def build_slide6(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_title(slide, "Self-Supervised Pretraining", C_AMBER)

    # ── Objective 1: MLM ──
    mlm_x = Inches(0.4)
    mlm_y = Inches(1.0)
    add_rounded_box(slide, mlm_x, mlm_y, Inches(6.1), Inches(3.7),
                    C_AMBER_LIGHT, C_AMBER, line_width=Pt(2))
    add_text_box(slide, mlm_x + Inches(0.2), mlm_y + Inches(0.05), Inches(5), Inches(0.35),
                 "Objective 1: Masked Language Model (MLM)", font_size=12, bold=True, color=C_AMBER)

    # MLM flow
    my = mlm_y + Inches(0.5)
    add_rounded_box(slide, mlm_x + Inches(0.3), my, Inches(5.4), Inches(0.6),
                    C_WHITE, C_AMBER,
                    "Input: [CALL_malloc, [MASK], ARITH_ADD, [MASK], RETURN]\nMask 15% of instruction tokens randomly",
                    font_size=9, font_color=C_DARK)

    add_arrow_shape(slide, mlm_x + Inches(2.5), my + Inches(0.65),
                   Inches(0.3), Inches(0.25), C_AMBER, 'down')

    my2 = my + Inches(1.0)
    add_rounded_box(slide, mlm_x + Inches(0.3), my2, Inches(5.4), Inches(0.5),
                    C_GREEN_LIGHT, C_GREEN,
                    "Block Encoder + Graph Encoder \u2192 contextualized embeddings",
                    font_size=9, font_color=C_DARK)

    add_arrow_shape(slide, mlm_x + Inches(2.5), my2 + Inches(0.55),
                   Inches(0.3), Inches(0.25), C_AMBER, 'down')

    my3 = my2 + Inches(0.9)
    add_rounded_box(slide, mlm_x + Inches(0.3), my3, Inches(5.4), Inches(0.5),
                    C_WHITE, C_AMBER,
                    "Predict masked tokens: [MASK] \u2192 MEM_READ_32, [MASK] \u2192 STACK_STORE_64",
                    font_size=9, font_color=C_DARK)

    add_text_box(slide, mlm_x + Inches(0.3), my3 + Inches(0.55), Inches(5.4), Inches(0.35),
                 "Learns token-level representations within CFG context",
                 font_size=9, italic=True, color=C_AMBER, align=PP_ALIGN.CENTER)

    # ── Objective 2: Contrastive ──
    cl_x = Inches(6.85)
    add_rounded_box(slide, cl_x, mlm_y, Inches(6.1), Inches(3.7),
                    C_AMBER_LIGHT, C_AMBER, line_width=Pt(2))
    add_text_box(slide, cl_x + Inches(0.2), mlm_y + Inches(0.05), Inches(5), Inches(0.35),
                 "Objective 2: Contrastive Learning (NT-Xent)", font_size=12, bold=True, color=C_AMBER)

    cy = mlm_y + Inches(0.5)
    # Positive pair
    add_rounded_box(slide, cl_x + Inches(0.3), cy, Inches(2.5), Inches(0.8),
                    C_GREEN_LIGHT, C_GREEN,
                    "Same function\ncompiled at O0\nf_O0", font_size=9, font_color=C_GREEN)
    add_rounded_box(slide, cl_x + Inches(3.3), cy, Inches(2.5), Inches(0.8),
                    C_GREEN_LIGHT, C_GREEN,
                    "Same function\ncompiled at O2\nf_O2", font_size=9, font_color=C_GREEN)

    add_text_box(slide, cl_x + Inches(2.0), cy + Inches(0.25), Inches(2.0), Inches(0.3),
                 "\u2190 pull together \u2192", font_size=9, bold=True, color=C_GREEN,
                 align=PP_ALIGN.CENTER)

    # Negative pairs
    cy2 = cy + Inches(1.0)
    add_rounded_box(slide, cl_x + Inches(0.3), cy2, Inches(2.5), Inches(0.6),
                    C_RED_LIGHT, C_RED,
                    "Different function\nf_other", font_size=9, font_color=C_RED)
    add_text_box(slide, cl_x + Inches(2.0), cy2 + Inches(0.15), Inches(2.0), Inches(0.3),
                 "\u2190 push apart \u2192", font_size=9, bold=True, color=C_RED,
                 align=PP_ALIGN.CENTER)
    add_rounded_box(slide, cl_x + Inches(3.3), cy2, Inches(2.5), Inches(0.6),
                    C_RED_LIGHT, C_RED,
                    "Different function\nf_other'", font_size=9, font_color=C_RED)

    cy3 = cy2 + Inches(0.75)
    add_text_box(slide, cl_x + Inches(0.3), cy3, Inches(5.4), Inches(0.7),
                 "Learns optimization-level invariance:\nO0 and O2 of same function map to similar embeddings",
                 font_size=9, italic=True, color=C_AMBER, align=PP_ALIGN.CENTER)

    # ── Transfer section ──
    tr_y = Inches(5.0)
    add_rounded_box(slide, Inches(0.4), tr_y, Inches(12.5), Inches(1.8),
                    C_GRAY_BG, C_AMBER, line_width=Pt(2))
    add_text_box(slide, Inches(0.6), tr_y + Inches(0.05), Inches(5), Inches(0.35),
                 "Pretraining \u2192 Fine-tuning Transfer", font_size=12, bold=True, color=C_AMBER)

    # Transfer details
    add_multiline_box(slide, Inches(0.7), tr_y + Inches(0.45), Inches(5.5), Inches(1.1),
                      [("Embedding Transfer", 10, True, C_AMBER),
                       ("84.4% of token embeddings copied by matching token names", 9, False, C_DARK),
                       ("(V3 vocab expanded 1,510 \u2192 2,279 between pretrain and fine-tune)", 8, False, C_MED_GRAY),
                       ("Block encoder + graph encoder weights fully transferred", 9, False, C_DARK)],
                      C_WHITE, C_AMBER)

    add_multiline_box(slide, Inches(6.6), tr_y + Inches(0.45), Inches(6.0), Inches(1.1),
                      [("Pretraining Setup", 10, True, C_AMBER),
                       ("10 epochs on A100 40GB", 9, False, C_DARK),
                       ("Joint MLM + contrastive loss", 9, False, C_DARK),
                       ("O0/O2 pair sampling via ContrastiveBatchSampler", 9, False, C_DARK),
                       ("Result: +3.4pp demo EM, +0.024 Val F1 over random init", 9, True, C_AMBER)],
                      C_WHITE, C_AMBER)

    add_slide_number(slide, 6)


# ══════════════════════════════════════════════════════════════════════
# SLIDE 7: Evaluation & Ablation
# ══════════════════════════════════════════════════════════════════════
def build_slide7(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_title(slide, "Evaluation & Ablation Study", C_CYAN)

    # ── Left: Ablation table ──
    tbl_x = Inches(0.3)
    tbl_y = Inches(0.95)
    add_text_box(slide, tbl_x, tbl_y, Inches(4), Inches(0.35),
                 "Ablation Study (Incremental)", font_size=12, bold=True, color=C_CYAN)

    # Table
    rows = 7
    cols = 5
    tw = Inches(7.5)
    th = Inches(3.2)
    table_shape = slide.shapes.add_table(rows, cols, tbl_x, tbl_y + Inches(0.4), tw, th)
    table = table_shape.table

    # Column widths
    col_widths = [Inches(0.4), Inches(2.6), Inches(1.0), Inches(1.0), Inches(1.0)]
    for i, w in enumerate(col_widths):
        table.columns[i].width = w

    # Header
    headers = ["#", "Model Configuration", "Params", "Test F1", "Demo EM"]
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = h
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            for run in p.runs:
                set_font(run, 9, True, color=C_WHITE)
        cell.fill.solid()
        cell.fill.fore_color.rgb = C_CYAN

    # Data rows
    data = [
        ("2", "GAT + GRU Decoder (baseline)", "4.4M", "0.606", "24.1%"),
        ("3", "+ External Call Encoder", "6.1M", "0.683", "19.8%"),
        ("4", "+ Callee/Caller Context", "8.0M", "0.781", "32.6%"),
        ("5", "+ Pretrain + Scale (25M)", "25M", "0.795", "48.5%"),
        ("5+", "+ k-NN Hybrid Retrieval", "25M", "0.804", "55.1%"),
        ("", "\u0394 (full pipeline vs baseline)", "", "+0.198", "+31.0pp"),
    ]
    for r, row_data in enumerate(data):
        row_idx = r + 1
        for c, val in enumerate(row_data):
            cell = table.cell(row_idx, c)
            cell.text = val
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER
                for run in p.runs:
                    if row_idx == len(data):
                        set_font(run, 9, True, color=C_CYAN)
                    elif row_idx == 5:  # k-NN row highlight
                        set_font(run, 9, True, color=C_INDIGO)
                    elif row_idx == 2 and c == 4:  # Demo EM drop
                        set_font(run, 9, True, color=C_RED)
                    else:
                        set_font(run, 9, False, color=C_DARK)
            # Alternating row colors
            if row_idx == len(data):
                cell.fill.solid()
                cell.fill.fore_color.rgb = C_CYAN_LIGHT
            elif row_idx == 5:
                cell.fill.solid()
                cell.fill.fore_color.rgb = C_INDIGO_LIGHT
            elif row_idx % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = C_GRAY_BG
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = C_WHITE

    # ── Ext-call paradox callout ──
    para_y = tbl_y + Inches(3.8)
    add_rounded_box(slide, tbl_x, para_y, Inches(7.5), Inches(0.75),
                    RGBColor(0xFF, 0xEB, 0xEE), C_RED, line_width=Pt(2))
    add_text_box(slide, tbl_x + Inches(0.15), para_y + Inches(0.05), Inches(7.2), Inches(0.65),
                 "Ext-Call Paradox: Model 3 demo EM DROPS -4.3pp despite +0.077 test F1 gain.\n"
                 "External calls alone cause overfitting to library patterns. Callee/caller context resolves this (+12.8pp).",
                 font_size=9, bold=True, color=C_RED)

    # ── Right: Metrics & Pipeline ──
    right_x = Inches(8.2)
    add_text_box(slide, right_x, tbl_y, Inches(4.5), Inches(0.35),
                 "Evaluation Metrics", font_size=12, bold=True, color=C_CYAN)

    metrics = [
        ("Sub-token F1", "Precision/recall on predicted\nsub-tokens vs ground truth"),
        ("Exact Match (EM)", "Fraction of functions where\nprediction = ground truth exactly"),
        ("Edit Similarity", "1 - (edit_distance / max_len)\nbetween prediction and truth"),
        ("N-gram Similarity", "Overlap of character n-grams\nbetween prediction and truth"),
    ]
    my = tbl_y + Inches(0.45)
    for name, desc in metrics:
        add_rounded_box(slide, right_x, my, Inches(4.8), Inches(0.7),
                        C_CYAN_LIGHT, C_CYAN)
        add_text_box(slide, right_x + Inches(0.1), my + Inches(0.02), Inches(4.6), Inches(0.25),
                     name, font_size=10, bold=True, color=C_CYAN)
        add_text_box(slide, right_x + Inches(0.1), my + Inches(0.25), Inches(4.6), Inches(0.4),
                     desc, font_size=8, color=C_DARK)
        my += Inches(0.78)

    # Per-package breakdown
    pkg_y = my + Inches(0.15)
    add_text_box(slide, right_x, pkg_y, Inches(4.5), Inches(0.3),
                 "Per-Package Demo EM (Top 5)", font_size=10, bold=True, color=C_CYAN)

    pkgs = [
        ("texinfo", "91.6%", 0.916),
        ("diffutils", "68.5%", 0.685),
        ("acct", "65.2%", 0.652),
        ("direvent", "60.1%", 0.601),
        ("rush", "58.4%", 0.584),
    ]
    for i, (name, pct, val) in enumerate(pkgs):
        py = pkg_y + Inches(0.35) + i * Inches(0.3)
        bar_w = Inches(val * 3.0)
        add_rounded_box(slide, right_x, py, bar_w, Inches(0.25),
                        C_CYAN_LIGHT, C_CYAN, line_width=Pt(0.5))
        add_text_box(slide, right_x + Inches(0.05), py - Inches(0.02), Inches(2.5), Inches(0.25),
                     f"{name}: {pct}", font_size=8, bold=True, color=C_CYAN)

    # Key insight
    ki_y = Inches(6.5)
    add_rounded_box(slide, Inches(0.3), ki_y, Inches(12.7), Inches(0.65),
                    C_CYAN_LIGHT, C_CYAN, line_width=Pt(2))
    add_text_box(slide, Inches(0.5), ki_y + Inches(0.05), Inches(12), Inches(0.55),
                 "Key Insight: The model is a pure recognizer \u2014 0/1,873 unseen function names correctly predicted. "
                 "All correct predictions are names seen in training (shared gnulib functions). "
                 "k-NN retrieval confirms and exploits this property for +6.6pp demo EM.",
                 font_size=10, bold=False, color=C_CYAN)

    add_slide_number(slide, 7)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    build_slide1(prs)
    build_slide2(prs)
    build_slide3(prs)
    build_slide4(prs)
    build_slide5(prs)
    build_slide6(prs)
    build_slide7(prs)

    out_path = "/home/apradipta/cs785-project/paper/FuncR_Architecture_Diagrams.pptx"
    prs.save(out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
