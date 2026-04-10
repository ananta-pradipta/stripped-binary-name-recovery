#!/usr/bin/env python3
"""
Generate FuncR paper figures as a PowerPoint file (4 slides).
Each slide is one figure, meant to be exported as an individual PDF
and included in the LaTeX paper via \\includegraphics.

Slide dimensions: 7.0 x 4.0 in (max needed; Figures 1,2 use full width,
Figures 3,4 use left ~3.35 in).
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor
from pptx.oxml.ns import qn
from lxml import etree

# ── Colors ─────────────────────────────────────────────────────────────
BLUE         = RGBColor(0x2563, 0xEB, 0xFF)[:3] if False else RGBColor(0x25, 0x63, 0xEB)
BLUE_FILL    = RGBColor(0xDB, 0xEA, 0xFE)
GREEN        = RGBColor(0x16, 0xA3, 0x4A)
GREEN_FILL   = RGBColor(0xDC, 0xFC, 0xE7)
ORANGE       = RGBColor(0xEA, 0x58, 0x0C)
ORANGE_FILL  = RGBColor(0xFF, 0xED, 0xD5)
RED          = RGBColor(0xDC, 0x26, 0x26)
RED_FILL     = RGBColor(0xFE, 0xE2, 0xE2)
PURPLE       = RGBColor(0x7C, 0x3A, 0xED)
PURPLE_FILL  = RGBColor(0xF3, 0xE8, 0xFF)
GRAY         = RGBColor(0x6B, 0x72, 0x80)
GRAY_FILL    = RGBColor(0xF3, 0xF4, 0xF6)
GRAY_BG      = RGBColor(0xF9, 0xFA, 0xFB)
WHITE        = RGBColor(0xFF, 0xFF, 0xFF)
BLACK        = RGBColor(0x00, 0x00, 0x00)
DARK         = RGBColor(0x1F, 0x29, 0x37)
BLUE_BG      = RGBColor(0xEF, 0xF6, 0xFF)
GREEN_BG     = RGBColor(0xEC, 0xFD, 0xF5)

FONT = "Calibri"


# ── Utility ────────────────────────────────────────────────────────────
def _box(slide, left, top, w, h, fill, border, text="", fsz=7, bold=False, fc=DARK,
         align=PP_ALIGN.CENTER, bw=Pt(0.75)):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = fill
    s.line.color.rgb = border; s.line.width = bw
    s.adjustments[0] = 0.08
    tf = s.text_frame; tf.word_wrap = True; tf.auto_size = None
    tf.margin_left = tf.margin_right = Pt(2)
    tf.margin_top = tf.margin_bottom = Pt(1)
    if text:
        for i, ln in enumerate(text.split("\n")):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align; p.space_before = Pt(0); p.space_after = Pt(0)
            r = p.add_run(); r.text = ln
            r.font.name = FONT; r.font.size = Pt(fsz); r.font.bold = bold; r.font.color.rgb = fc
    return s

def _txt(slide, left, top, w, h, text, fsz=7, bold=False, color=DARK, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(left, top, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Pt(1)
    tf.margin_top = tf.margin_bottom = Pt(0)
    for i, ln in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.space_before = Pt(0); p.space_after = Pt(0)
        r = p.add_run(); r.text = ln
        r.font.name = FONT; r.font.size = Pt(fsz); r.font.bold = bold; r.font.color.rgb = color
    return tb

def _arrow(slide, x1, y1, x2, y2, color=GRAY, w=Pt(0.75)):
    c = slide.shapes.add_connector(1, x1, y1, x2, y2)
    c.line.color.rgb = color; c.line.width = w
    c.begin_x = x1; c.begin_y = y1; c.end_x = x2; c.end_y = y2
    ln = c.line._ln
    te = ln.find(qn('a:tailEnd'))
    if te is None: te = etree.SubElement(ln, qn('a:tailEnd'))
    te.set('type', 'triangle'); te.set('w', 'sm'); te.set('len', 'sm')
    return c

def _dash_arrow(slide, x1, y1, x2, y2, color=RED, w=Pt(0.75)):
    c = _arrow(slide, x1, y1, x2, y2, color=color, w=w)
    ln = c.line._ln
    pd = ln.find(qn('a:prstDash'))
    if pd is None: pd = etree.SubElement(ln, qn('a:prstDash'))
    pd.set('val', 'dash')
    return c

def _phase(slide, l, t, w, h, fill, label, lc=DARK):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = fill
    s.line.color.rgb = RGBColor(0x9C, 0xA3, 0xAF); s.line.width = Pt(0.5)
    ln = s.line._ln
    pd = ln.find(qn('a:prstDash'))
    if pd is None: pd = etree.SubElement(ln, qn('a:prstDash'))
    pd.set('val', 'dash')
    s.adjustments[0] = 0.02
    _txt(slide, l + Pt(4), t + Pt(2), w - Pt(8), Pt(10), label, fsz=5, bold=True, color=lc)
    return s

def R(s): return s.left + s.width      # right
def B(s): return s.top + s.height      # bottom
def CX(s): return s.left + s.width//2  # center x
def CY(s): return s.top + s.height//2  # center y


# ═══════════════════════════════════════════════════════════════════════
#  SLIDE 1 — Figure 1: End-to-End Pipeline (7.0 x 3.5)
# ═══════════════════════════════════════════════════════════════════════
def slide1(prs):
    sl = prs.slides.add_slide(prs.slide_layouts[6])

    # Phase backgrounds (all within top 3.5 in)
    _phase(sl, Inches(0.04), Inches(0.22), Inches(2.10), Inches(2.95),
           GRAY_BG, "Phase 1: Preprocessing", GRAY)
    _phase(sl, Inches(2.22), Inches(0.22), Inches(3.05), Inches(2.95),
           BLUE_BG, "Phase 2: Encoder + Multi-Context Fusion", BLUE)
    _phase(sl, Inches(5.35), Inches(0.22), Inches(1.55), Inches(2.95),
           GREEN_BG, "Phase 3: Inference & Evaluation", GREEN)

    # ── Phase 1 ──
    bw, bh = Inches(0.37), Inches(0.42)
    g = Inches(0.04)
    y1 = Inches(1.40)
    x = Inches(0.10)
    labels1 = ["Source\nPackages", "Compile\n(O0, O2)", "BAP 2.5\nIR Lifting",
                "V3 Tokenize\n1,510 types", "CFG +\nLabels\nnm matching"]
    p1 = []
    for lb in labels1:
        b = _box(sl, x, y1, bw, bh, GRAY_FILL, GRAY, lb, fsz=4.5)
        p1.append(b); x += bw + g
    for i in range(len(p1)-1):
        _arrow(sl, R(p1[i]), CY(p1[i]), p1[i+1].left, CY(p1[i+1]), GRAY, Pt(0.6))

    # ── Phase 2 ──
    bw2, bh2 = Inches(0.58), Inches(0.52)
    x2 = Inches(2.32)
    y2 = Inches(1.35)

    be = _box(sl, x2, y2, bw2, bh2, BLUE_FILL, BLUE, "Block Encoder\nTransformer\n4L, 8H, 256d", fsz=4.5)
    ga = _box(sl, x2+bw2+Inches(0.07), y2, bw2, bh2, BLUE_FILL, BLUE,
              "Graph Encoder\nGAT\n3L, 8H, 1024d", fsz=4.5)

    _arrow(sl, R(p1[-1]), CY(p1[-1]), be.left, CY(be), GRAY, Pt(0.6))
    _txt(sl, R(p1[-1])-Pt(2), CY(p1[-1])-Pt(12), Inches(0.28), Pt(10),
         "tokens\n+edges", fsz=3.5, color=GRAY)

    _arrow(sl, R(be), CY(be), ga.left, CY(ga), BLUE, Pt(0.6))
    _txt(sl, R(be)+Pt(1), CY(be)-Pt(9), Inches(0.15), Pt(8), "b_i", fsz=5, bold=True, color=BLUE)

    # Gates
    gw, gh = Inches(0.36), Inches(0.32)
    gg = Inches(0.05)
    gx = R(ga) + Inches(0.10)
    gy = y2 + Inches(0.10)
    g1 = _box(sl, gx, gy, gw, gh, GREEN_FILL, GREEN, "Gate 1", fsz=4.5, bold=True)
    g2 = _box(sl, gx+gw+gg, gy, gw, gh, GREEN_FILL, GREEN, "Gate 2", fsz=4.5, bold=True)
    g3 = _box(sl, gx+2*(gw+gg), gy, gw, gh, GREEN_FILL, GREEN, "Gate 3", fsz=4.5, bold=True)

    _arrow(sl, R(ga), CY(ga), g1.left, CY(g1), GREEN, Pt(0.6))
    _txt(sl, R(ga)+Pt(1), CY(ga)-Pt(9), Inches(0.12), Pt(8), "f", fsz=5, bold=True, color=BLUE)
    _arrow(sl, R(g1), CY(g1), g2.left, CY(g2), GREEN, Pt(0.6))
    _arrow(sl, R(g2), CY(g2), g3.left, CY(g3), GREEN, Pt(0.6))

    # Context encoders above gates
    cw, ch = Inches(0.48), Inches(0.36)
    cy_c = gy - Inches(0.48)
    ee = _box(sl, CX(g1)-cw//2, cy_c, cw, ch, ORANGE_FILL, ORANGE, "Ext Call\nBiGRU", fsz=4)
    ce = _box(sl, CX(g2)-cw//2, cy_c, cw, ch, ORANGE_FILL, ORANGE, "Callee\nBiGRU", fsz=4)
    cr = _box(sl, CX(g3)-cw//2, cy_c, cw, ch, ORANGE_FILL, ORANGE, "Caller\nBiGRU", fsz=4)

    _arrow(sl, CX(ee), B(ee), CX(g1), g1.top, ORANGE, Pt(0.6))
    _arrow(sl, CX(ce), B(ce), CX(g2), g2.top, ORANGE, Pt(0.6))
    _arrow(sl, CX(cr), B(cr), CX(g3), g3.top, ORANGE, Pt(0.6))

    # Bypass
    _dash_arrow(sl, CX(ga), B(ga), CX(g3), B(g3), GRAY, Pt(0.5))
    _txt(sl, CX(ga)+Inches(0.10), B(ga)+Pt(1), Inches(0.35), Pt(7),
         "bypass", fsz=3.5, color=GRAY)

    # ── Phase 3 ──
    px = Inches(5.45)
    dec = _box(sl, px, Inches(1.05), Inches(0.68), Inches(0.38),
               RED_FILL, RED, "GRU Decoder\nbeam k=5", fsz=4.5)
    kn = _box(sl, px, Inches(1.55), Inches(0.68), Inches(0.38),
              RED_FILL, RED, "k-NN Retrieval\nk=1, cosine", fsz=4.5)
    pn = _box(sl, px+Inches(0.78), Inches(1.22), Inches(0.52), Inches(0.35),
              GRAY_FILL, GRAY, "Predicted\nName", fsz=4.5)
    mt = _box(sl, px+Inches(0.78), Inches(1.70), Inches(0.52), Inches(0.40),
              GRAY_FILL, GRAY, "Metrics\nF1, EM\nEdSim, NgSim", fsz=4)

    _arrow(sl, R(g3), CY(g3), dec.left, CY(dec), GREEN, Pt(0.6))
    _arrow(sl, R(g3), CY(g3), kn.left, CY(kn), GREEN, Pt(0.6))
    _txt(sl, R(g3)+Pt(1), CY(g3)-Pt(9), Inches(0.12), Pt(8), "z", fsz=5, bold=True, color=GREEN)

    _arrow(sl, R(dec), CY(dec), pn.left, CY(pn), GRAY, Pt(0.6))
    _arrow(sl, R(kn), CY(kn), pn.left, B(pn)-Pt(5), GRAY, Pt(0.6))
    _arrow(sl, CX(pn), B(pn), CX(mt), mt.top, GRAY, Pt(0.6))


# ═══════════════════════════════════════════════════════════════════════
#  SLIDE 2 — Figure 2: Detailed Architecture (7.0 x 4.0)
# ═══════════════════════════════════════════════════════════════════════
def slide2(prs):
    sl = prs.slides.add_slide(prs.slide_layouts[6])

    # ── Top: Block Encoder chain ──
    _txt(sl, Inches(0.65), Inches(0.04), Inches(2.5), Pt(11),
         "Block Encoder (per basic block)", fsz=6, bold=True, color=BLUE)

    bh = Inches(0.42)
    y = Inches(0.22)
    x = Inches(0.06)
    tok = _box(sl, x, y, Inches(0.60), bh, GRAY_FILL, GRAY,
               "Instruction\nTokens\n(max 20)", fsz=4.5)
    x = R(tok)+Inches(0.06)
    emb = _box(sl, x, y, Inches(0.70), bh, BLUE_FILL, BLUE,
               "Token Embedding\n2,279 x 256", fsz=4.5)
    x = R(emb)+Inches(0.06)
    pe = _box(sl, x, y, Inches(0.55), bh, BLUE_FILL, BLUE,
              "+ Positional\nEncoding", fsz=4.5)
    x = R(pe)+Inches(0.06)
    txf = _box(sl, x, y, Inches(0.88), bh, BLUE_FILL, BLUE,
               "Transformer Encoder\n4 layers, 8 heads, ff=512", fsz=4.5)
    x = R(txf)+Inches(0.06)
    pool = _box(sl, x, y, Inches(0.52), bh, BLUE_FILL, BLUE,
                "Mean Pool\n-> 512d", fsz=4.5)

    for a, b in [(tok,emb),(emb,pe),(pe,txf),(txf,pool)]:
        _arrow(sl, R(a), CY(a), b.left, CY(b), BLUE, Pt(0.6))

    # Graph encoder
    _txt(sl, R(pool)+Inches(0.15), Inches(0.04), Inches(1.8), Pt(11),
         "Graph Encoder (over CFG)", fsz=6, bold=True, color=BLUE)

    gat = _box(sl, R(pool)+Inches(0.15), y, Inches(0.78), bh, BLUE_FILL, BLUE,
               "GAT\n3L, 8H\n512d -> 1024d", fsz=4.5)
    ap = _box(sl, R(gat)+Inches(0.06), y, Inches(0.62), bh, BLUE_FILL, BLUE,
              "Attention\nPooling\n-> 1024d", fsz=4.5)

    _arrow(sl, R(pool), CY(pool), gat.left, CY(gat), BLUE, Pt(0.6))
    _txt(sl, R(pool)+Pt(2), CY(pool)-Pt(9), Inches(0.30), Pt(8),
         "b_i (512d)", fsz=4, bold=True, color=BLUE)
    _arrow(sl, R(gat), CY(gat), ap.left, CY(ap), BLUE, Pt(0.6))

    # CFG edges
    cfg = _box(sl, CX(gat)-Inches(0.22), y-Inches(0.22), Inches(0.44), Inches(0.17),
               GRAY_FILL, GRAY, "CFG Edges", fsz=4)
    _arrow(sl, CX(cfg), B(cfg), CX(gat), gat.top, GRAY, Pt(0.5))

    # ── Fusion section label ──
    _txt(sl, Inches(1.7), Inches(0.78), Inches(3.5), Pt(11),
         "Cascaded Gated Fusion (conditional bypass per stage)", fsz=6, bold=True, color=GREEN)

    # Three gates in a row (right to left: g1, g2, g3)
    gw = Inches(1.80)
    gh = Inches(0.50)
    yg = Inches(0.98)
    gg = Inches(0.15)

    g1 = _box(sl, Inches(4.95), yg, gw, gh, GREEN_FILL, GREEN,
              "Gate 1: External Calls\ng = s(W[f;c_ext])\nz = g*f + (1-g)*c_ext", fsz=4.5)
    g2 = _box(sl, Inches(4.95)-gw-gg, yg, gw, gh, GREEN_FILL, GREEN,
              "Gate 2: Callee Context\ng = s(W[z;c_callee])\nz = g*z + (1-g)*c_callee", fsz=4.5)
    g3 = _box(sl, Inches(4.95)-2*(gw+gg), yg, gw, gh, GREEN_FILL, GREEN,
              "Gate 3: Caller Context\ng = s(W[z;c_caller])\nz = g*z + (1-g)*c_caller", fsz=4.5)

    # Bold first line of each gate
    for g in [g1, g2, g3]:
        g.text_frame.paragraphs[0].runs[0].font.bold = True

    # f -> g1
    _arrow(sl, CX(ap), B(ap), CX(g1), g1.top, BLUE, Pt(0.6))
    _txt(sl, CX(ap)+Pt(3), B(ap), Inches(0.35), Pt(8), "f (1024d)", fsz=4, bold=True, color=BLUE)

    # g1->g2->g3
    _arrow(sl, g1.left, CY(g1), R(g2), CY(g2), GREEN, Pt(0.6))
    _arrow(sl, g2.left, CY(g2), R(g3), CY(g3), GREEN, Pt(0.6))

    # Bypass
    _dash_arrow(sl, ap.left, B(ap), g3.left+Pt(5), g3.top, GRAY, Pt(0.5))
    _txt(sl, g3.left-Inches(0.25), g3.top-Pt(10), Inches(0.45), Pt(8),
         "bypass (no context)", fsz=3.5, color=GRAY)

    # Context encoders below
    cw = Inches(1.55)
    ch = Inches(0.36)
    yc = B(g1)+Inches(0.10)

    ext = _box(sl, CX(g1)-cw//2, yc, cw, ch, ORANGE_FILL, ORANGE,
               "Ext Call Encoder\nEmb(656,256) -> BiGRU -> 1024d", fsz=4)
    cle = _box(sl, CX(g2)-cw//2, yc, cw, ch, ORANGE_FILL, ORANGE,
               "Callee Encoder\nEmb(2279,128) -> BiGRU -> 1024d", fsz=4)
    clr = _box(sl, CX(g3)-cw//2, yc, cw, ch, ORANGE_FILL, ORANGE,
               "Caller Encoder\nEmb(2279,128) -> BiGRU -> 1024d", fsz=4)

    for enc, gate in [(ext,g1),(cle,g2),(clr,g3)]:
        _arrow(sl, CX(enc), enc.top, CX(gate), B(gate), ORANGE, Pt(0.6))

    # Data inputs
    dw, dh = Inches(1.20), Inches(0.28)
    yd = B(ext)+Inches(0.06)
    ed = _box(sl, CX(ext)-dw//2, yd, dw, dh, GRAY_FILL, GRAY,
              "PLT names\nmalloc, printf, ...", fsz=4)
    cd = _box(sl, CX(cle)-dw//2, yd, dw, dh, GRAY_FILL, GRAY,
              "Callee token sigs\ntop 5 x 10 tokens", fsz=4)
    rd = _box(sl, CX(clr)-dw//2, yd, dw, dh, GRAY_FILL, GRAY,
              "Caller token sigs\ntop 5 x 10 tokens", fsz=4)

    for d, e in [(ed,ext),(cd,cle),(rd,clr)]:
        _arrow(sl, CX(d), d.top, CX(e), B(e), GRAY, Pt(0.4))

    # Output heads
    ydec = Inches(2.72)
    dec = _box(sl, Inches(0.06), ydec, Inches(1.35), Inches(0.48),
               RED_FILL, RED, "GRU Decoder\n1024 hidden, 1 layer\nVotes vocab 2,642, beam k=5", fsz=4.5)
    knn = _box(sl, Inches(1.50), ydec, Inches(1.25), Inches(0.48),
               RED_FILL, RED, "k-NN Retrieval\ncosine sim, k=1\n87K training index", fsz=4.5)
    out = _box(sl, Inches(0.50), ydec+Inches(0.58), Inches(0.85), Inches(0.28),
               GRAY_FILL, GRAY, "Function Name", fsz=5, bold=True)

    # z arrow
    _arrow(sl, g3.left, B(g3), R(dec), dec.top, GREEN, Pt(0.6))
    _arrow(sl, g3.left, B(g3), knn.left, knn.top, GREEN, Pt(0.6))
    _txt(sl, g3.left-Inches(0.12), B(g3)-Pt(2), Inches(0.15), Pt(8),
         "z", fsz=5, bold=True, color=GREEN)

    _arrow(sl, CX(dec), B(dec), CX(out), out.top, GRAY, Pt(0.6))
    _arrow(sl, CX(knn), B(knn), R(out), out.top, GRAY, Pt(0.6))


# ═══════════════════════════════════════════════════════════════════════
#  SLIDE 3 — Figure 3: Self-Supervised Pretraining (3.35 x 2.5)
# ═══════════════════════════════════════════════════════════════════════
def slide3(prs):
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    # Content within 3.35 x 2.5 area

    # Center encoders
    bw, bh = Inches(0.82), Inches(0.38)
    y_mid = Inches(1.00)
    be = _box(sl, Inches(0.50), y_mid, bw, bh, BLUE_FILL, BLUE,
              "Block Encoder\nTransformer", fsz=6)
    ge = _box(sl, Inches(1.65), y_mid, bw, bh, BLUE_FILL, BLUE,
              "Graph Encoder\nGAT", fsz=6)
    _arrow(sl, R(be), CY(be), ge.left, CY(ge), BLUE, Pt(0.75))
    _txt(sl, R(be)+Pt(3), CY(be)-Pt(9), Inches(0.18), Pt(8), "b_i", fsz=5.5, bold=True, color=BLUE)

    # ── MLM (top) ──
    _txt(sl, Inches(1.10), Inches(0.02), Inches(1.2), Pt(10),
         "Objective 1: MLM", fsz=6.5, bold=True, color=PURPLE)

    mk = _box(sl, Inches(0.25), Inches(0.20), Inches(0.72), Inches(0.32),
              GRAY_FILL, GRAY, "Masked Tokens\n15% masked", fsz=5)
    mlm = _box(sl, Inches(1.60), Inches(0.20), Inches(0.90), Inches(0.32),
               PURPLE_FILL, PURPLE, "MLM Head\npredict masked tokens", fsz=5)

    _arrow(sl, CX(mk), B(mk), CX(be), be.top, GRAY, Pt(0.6))
    _arrow(sl, R(be)+Pt(5), be.top+Pt(3), mlm.left, CY(mlm), PURPLE, Pt(0.6))
    _txt(sl, R(be)-Pt(6), be.top-Pt(9), Inches(0.40), Pt(7), "token embs", fsz=4, color=PURPLE)

    # ── Contrastive (bottom) ──
    _txt(sl, Inches(0.90), Inches(2.02), Inches(1.5), Pt(10),
         "Objective 2: Contrastive", fsz=6.5, bold=True, color=PURPLE)

    o0 = _box(sl, Inches(0.18), Inches(1.58), Inches(0.52), Inches(0.30),
              GRAY_FILL, GRAY, "Function\n(O0)", fsz=5)
    o2 = _box(sl, Inches(0.78), Inches(1.58), Inches(0.52), Inches(0.30),
              GRAY_FILL, GRAY, "Function\n(O2)", fsz=5)
    nt = _box(sl, Inches(1.48), Inches(1.53), Inches(1.05), Inches(0.40),
              PURPLE_FILL, PURPLE, "NT-Xent Loss\npull O0/O2 together\npush diff funcs apart", fsz=4.5)

    _arrow(sl, CX(o0), o0.top, be.left+Pt(5), B(be), GRAY, Pt(0.6))
    _arrow(sl, CX(o2), o2.top, R(be)-Pt(5), B(be), GRAY, Pt(0.6))
    _arrow(sl, CX(ge), B(ge), nt.left, CY(nt), PURPLE, Pt(0.6))
    _txt(sl, CX(ge)+Pt(3), B(ge)-Pt(2), Inches(0.12), Pt(7), "f", fsz=5.5, bold=True, color=BLUE)

    # Transfer annotation
    _txt(sl, Inches(2.62), Inches(0.92), Inches(0.65), Inches(0.50),
         "After pretraining:\n84.4% embeddings\ntransferred to\nfull model", fsz=5, color=DARK)
    _dash_arrow(sl, R(ge), CY(ge), Inches(2.62), CY(ge), GRAY, Pt(0.5))


# ═══════════════════════════════════════════════════════════════════════
#  SLIDE 4 — Figure 4: Cascaded Gated Fusion (3.35 x 4.0)
# ═══════════════════════════════════════════════════════════════════════
def slide4(prs):
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    # Content within 3.35 x 4.0 area

    # Top: f input
    fw = Inches(1.30)
    cx_s = Inches(1.00)  # left edge of centered content
    f = _box(sl, cx_s, Inches(0.06), fw, Inches(0.26),
             BLUE_FILL, BLUE, "f \u2208 R^1024 (from Graph Encoder)", fsz=5.5, bold=True)

    # Three stages
    gw = Inches(1.52)
    gh = Inches(0.48)
    ew = Inches(0.90)
    eh = Inches(0.40)
    dw_ = Inches(0.78)
    dh_ = Inches(0.32)
    bw_ = Inches(0.52)
    bh_ = Inches(0.20)
    vgap = Inches(0.12)

    stages = [
        ("Gate 1: External Calls",
         "g = \u03c3(W[z;c_ext])\nz = g\u2299f + (1-g)\u2299c_ext",
         "Ext Call Encoder\nEmb(656,256)\n\u2192 BiGRU \u2192 1024d",
         "PLT names:\nmalloc, printf\nsocket, bind",
         "bypass if\nno ext calls"),
        ("Gate 2: Callee Context",
         "g = \u03c3(W[z;c_callee])\nz = g\u2299z + (1-g)\u2299c_callee",
         "Callee Encoder\nEmb(2279,128)\n\u2192 BiGRU \u2192 1024d",
         "Callee token sigs\ntop 5 callees\n\u00d7 10 tokens",
         "bypass if\nno callees"),
        ("Gate 3: Caller Context",
         "g = \u03c3(W[z;c_caller])\nz = g\u2299z + (1-g)\u2299c_caller",
         "Caller Encoder\nEmb(2279,128)\n\u2192 BiGRU \u2192 1024d",
         "Caller token sigs\ntop 5 callers\n\u00d7 10 tokens",
         "bypass if\nno callers"),
    ]

    y = Inches(0.44)
    prev = None
    bx = Inches(0.02)  # bypass x

    for title, eq, enc_txt, data_txt, byp_txt in stages:
        # Gate
        gate = _box(sl, cx_s, y, gw, gh, GREEN_FILL, GREEN,
                    title + "\n" + eq, fsz=5)
        gate.text_frame.paragraphs[0].runs[0].font.bold = True

        # Encoder (right)
        enc = _box(sl, R(gate)+Inches(0.05), y+Inches(0.02), ew, eh,
                   ORANGE_FILL, ORANGE, enc_txt, fsz=4.5)
        # Data (below encoder)
        dat = _box(sl, R(gate)+Inches(0.05), B(enc)+Inches(0.02), ew, dh_,
                   GRAY_FILL, GRAY, data_txt, fsz=3.5)
        _arrow(sl, CX(dat), dat.top, CX(enc), B(enc), GRAY, Pt(0.4))
        _arrow(sl, enc.left, CY(enc), R(gate), CY(gate), ORANGE, Pt(0.6))

        # Bypass (left)
        byp = _box(sl, bx, y+Inches(0.12), bw_, bh_,
                   RED_FILL, RED, byp_txt, fsz=4, bw=Pt(0.5))
        # dashed border
        try:
            ln = byp.line._ln
            pd = ln.find(qn('a:prstDash'))
            if pd is None: pd = etree.SubElement(ln, qn('a:prstDash'))
            pd.set('val', 'dash')
        except: pass
        _dash_arrow(sl, gate.left, CY(gate), R(byp), CY(byp), RED, Pt(0.5))

        # Vertical connection
        if prev is None:
            _arrow(sl, CX(f), B(f), CX(gate), gate.top, BLUE, Pt(0.6))
            _txt(sl, CX(f)+Pt(3), B(f)-Pt(1), Inches(0.22), Pt(7), "z = f", fsz=4, color=BLUE)
        else:
            _arrow(sl, CX(prev), B(prev), CX(gate), gate.top, GREEN, Pt(0.6))

        prev = gate
        y = B(gate) + vgap

    # Output
    z = _box(sl, cx_s, y+Inches(0.02), Inches(1.52), Inches(0.26),
             RED_FILL, RED, "z \u2208 R^1024 \u2192 Decoder / k-NN", fsz=5.5, bold=True)
    _arrow(sl, CX(prev), B(prev), CX(z), z.top, GREEN, Pt(0.6))


# ═══════════════════════════════════════════════════════════════════════
def main():
    prs = Presentation()
    prs.slide_width = Inches(7.0)
    prs.slide_height = Inches(4.0)

    slide1(prs)
    slide2(prs)
    slide3(prs)
    slide4(prs)

    out = "/home/apradipta/cs785-project/paper/FuncR_Paper_Figures.pptx"
    prs.save(out)
    print(f"Saved {out}")
    print("4 slides: Fig1 (pipeline 7x3.5), Fig2 (architecture 7x4),")
    print("          Fig3 (pretraining 3.35x2.5), Fig4 (fusion 3.35x4)")

if __name__ == "__main__":
    main()
