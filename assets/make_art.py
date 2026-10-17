"""Generate the SVG art for the profile README: hero banner, section divider and toolbox.

Run from the repo root:
    pip install -r assets/requirements.txt
    python3 assets/make_art.py

Everything is plain SVG + CSS animation, so GitHub renders it with no external services.
Edit PROFILE / TOOLS below and re-run to update the images.
"""
import os
import random
from math import cos, pi, sin, sqrt
from xml.sax.saxutils import escape

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- content
PROFILE = dict(
    name="Timur Afandiyev",
    role="Chemoinformatics · Chemical Engineering · Data Science",
    where="M2 Chemoinformatics · Université de Strasbourg",
    prompt="~/timur $ ./explore --space chemical",
)

# (symbol, name, family)
TOOLS = [
    ("Py", "Python", "code"),
    ("Rd", "RDKit", "mol"),
    ("Sq", "SQL", "code"),
    ("Gt", "GTM", "space"),
    ("Um", "UMAP", "space"),
    ("Ts", "t-SNE", "space"),
    ("Pc", "PCA", "space"),
    ("So", "SOM", "space"),
    ("Mf", "Morgan FP", "mol"),
    ("Is", "ISIDA", "mol"),
    ("Sm", "SMILES", "mol"),
    ("Hy", "Aspen HYSYS", "proc"),
    ("Th", "Thermo", "proc"),
    ("Jp", "Jupyter", "code"),
    ("Hc", "HTML/CSS", "code"),
]
FAMILIES = [
    ("space", "chemical space"),
    ("mol", "molecular representation"),
    ("proc", "process engineering"),
    ("code", "code & data"),
]

THEMES = {
    "dark": dict(
        BG="#0d1117", PANEL="#0f1620", EDGE="#21262d", TEXT="#e6edf3", MUTED="#8b949e",
        ACCENT="#38bdf8", LINE="#38bdf8", TILE="#0d1117",
        FAM={"space": "#38bdf8", "mol": "#a78bfa", "proc": "#f59e0b", "code": "#34d399"},
        CLUSTER=["#38bdf8", "#a78bfa", "#f472b6", "#34d399"],
    ),
    "light": dict(
        BG="#f6f8fa", PANEL="#ffffff", EDGE="#d0d7de", TEXT="#1f2328", MUTED="#59636e",
        ACCENT="#0369a1", LINE="#0284c7", TILE="#ffffff",
        FAM={"space": "#0369a1", "mol": "#6d28d9", "proc": "#b45309", "code": "#047857"},
        CLUSTER=["#0284c7", "#7c3aed", "#db2777", "#059669"],
    ),
}

SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,'SF Mono',Menlo,Consolas,monospace"
PERIOD = 14  # seconds for one full hero animation loop
REDUCED = "@media (prefers-reduced-motion: reduce){*{animation:none!important}}"


# ---------------------------------------------------------------- landscape (GTM-style density map)
CENTERS = [(-1.6, 0.4, 0.55), (0.2, -0.5, 0.45), (1.5, 0.6, 0.5), (0.4, 1.1, 0.35)]  # x, y, sigma


def density(X, Y):
    Z = np.zeros_like(X)
    for (cx, cy, s), w in zip(CENTERS, (1.0, 0.85, 0.9, 0.6)):
        Z += w * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s * s))
    return Z


def contour_paths(x0, y0, w, h):
    """Contour polylines of the density, mapped into the box (x0, y0, w, h)."""
    xs = np.linspace(-3, 3, 220)
    ys = np.linspace(-1.2, 2.0, 120)
    X, Y = np.meshgrid(xs, ys)
    Z = density(X, Y)
    cs = plt.contour(X, Y, Z, levels=np.linspace(0.12, 0.95, 8))
    out = []
    for li, segs in enumerate(cs.allsegs):
        for seg in segs:
            if len(seg) < 6:
                continue
            pts = [(x0 + (px + 3) / 6 * w, y0 + h - (py + 1.2) / 3.2 * h) for px, py in seg]
            d = "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in pts)
            out.append((li, d))
    plt.close("all")
    return out


def landscape(x0, y0, w, h, t, uid, n_points=64):
    """Contours drawing themselves, molecules (points) popping in cluster by cluster, a probe hopping between them."""
    rnd = random.Random(7)
    g = []
    g.append(f'<g mask="url(#fade{uid})">')
    for i, (li, d) in enumerate(contour_paths(x0, y0, w, h)):
        op = 0.18 + 0.07 * li
        g.append(f'<path d="{d}" class="ct" style="stroke-opacity:{op:.2f};animation-delay:{li * 0.25:.2f}s"/>')
    pts = []
    for k in range(n_points):
        c = k % len(CENTERS)
        cx, cy, s = CENTERS[c]
        px, py = rnd.gauss(cx, s * 0.6), rnd.gauss(cy, s * 0.6)
        if not (-3 < px < 3 and -1.2 < py < 2.0):
            continue
        sx = x0 + (px + 3) / 6 * w
        sy = y0 + h - (py + 1.2) / 3.2 * h
        pts.append((sx, sy, c))
    for i, (sx, sy, c) in enumerate(pts):
        delay = 1.2 + c * 0.9 + rnd.random() * 0.8
        g.append(
            f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="3.2" class="pt" '
            f'style="fill:{t["CLUSTER"][c]};animation-delay:{delay:.2f}s"/>'
        )
    g.append("</g>")

    # probe: a ring that visits each cluster centre and shows its label
    labels = ["kinase-like", "fragments", "natural products", "solvents"]
    steps = []
    for c, (cx, cy, _) in enumerate(CENTERS):
        sx = x0 + (cx + 3) / 6 * w
        sy = y0 + h - (cy + 1.2) / 3.2 * h
        steps.append((sx, sy, labels[c], t["CLUSTER"][c]))
    n = len(steps)
    kf = []
    for i, (sx, sy, _, _) in enumerate(steps):
        a, b = i * 100 / n, (i + 0.7) * 100 / n
        kf.append(f"{a:.1f}%,{b:.1f}%{{transform:translate({sx:.1f}px,{sy:.1f}px)}}")
    sx0, sy0 = steps[0][:2]
    kf.append(f"100%{{transform:translate({sx0:.1f}px,{sy0:.1f}px)}}")
    css = f"@keyframes probe{uid}{{{''.join(kf)}}}"
    g.append(f'<g class="probe" style="animation:probe{uid} {PERIOD}s ease-in-out infinite">')
    g.append(f'<circle r="16" fill="none" style="stroke:{t["TEXT"]}" stroke-width="1.4" stroke-dasharray="4 4" class="spin"/>')
    g.append(f'<circle r="2.5" style="fill:{t["TEXT"]}"/>')
    for i, (sx, _, lab, col) in enumerate(steps):
        right = sx > x0 + 0.6 * w  # flip the label left when the cluster sits near the right edge
        g.append(
            f'<text x="{-22 if right else 22}" y="-14" text-anchor="{"end" if right else "start"}" class="lab" '
            f'style="fill:{col};animation-delay:{i * PERIOD / n:.2f}s">{escape(lab)}</text>'
        )
    g.append("</g>")
    return "\n".join(g), css


def hero(t, mobile=False):
    uid = ("m" if mobile else "d")
    if mobile:
        W, H = 600, 440
        lx, ly, lw, lh = 10, 190, 580, 240
        fade = f'<linearGradient id="g{uid}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset=".25" stop-color="#fff"/></linearGradient>'
        tx, ty = 32, 82
        name_size, role_size = 42, 16
    else:
        W, H = 1200, 320
        lx, ly, lw, lh = 540, 24, 650, 272
        fade = f'<linearGradient id="g{uid}" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset=".22" stop-color="#fff"/></linearGradient>'
        tx, ty = 56, 98
        name_size, role_size = 54, 17
    body, probe_css = landscape(lx, ly, lw, lh, t, uid)
    css = f"""
.ct{{fill:none;stroke:{t['LINE']};stroke-width:1.3;stroke-dasharray:1400;stroke-dashoffset:1400;
  animation:draw {PERIOD}s ease-in-out infinite}}
@keyframes draw{{0%{{stroke-dashoffset:1400}}25%,85%{{stroke-dashoffset:0}}100%{{stroke-dashoffset:-1400}}}}
.pt{{opacity:0;transform-box:fill-box;transform-origin:center;animation:pop {PERIOD}s ease-out infinite}}
@keyframes pop{{0%{{opacity:0;transform:scale(0)}}4%{{opacity:.95;transform:scale(1.5)}}7%,80%{{opacity:.9;transform:scale(1)}}90%,100%{{opacity:0;transform:scale(1)}}}}
.lab{{font:600 12px {MONO};opacity:0;animation:lab {PERIOD}s infinite}}
@keyframes lab{{0%,2%{{opacity:0}}5%,20%{{opacity:1}}23%,100%{{opacity:0}}}}
.spin{{transform-origin:0 0;animation:spin 6s linear infinite}}
@keyframes spin{{to{{transform:rotate(360deg)}}}}
.cur{{animation:blink 1.1s steps(1) infinite}}
@keyframes blink{{50%{{opacity:0}}}}
{probe_css}
{REDUCED}
"""
    p = PROFILE
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="{escape(p["name"])} — {escape(p["role"])}">',
        f"<style>{css}</style>",
        f"<defs>{fade}<mask id=\"fade{uid}\"><rect x=\"0\" y=\"0\" width=\"{W}\" height=\"{H}\" fill=\"url(#g{uid})\"/></mask></defs>",
        f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="14" style="fill:{t["PANEL"]};stroke:{t["EDGE"]}"/>',
        body,
        f'<text x="{tx}" y="{ty - 44}" style="font:500 14px {MONO};fill:{t["MUTED"]}">{escape(p["prompt"])}<tspan class="cur" style="fill:{t["ACCENT"]}">▍</tspan></text>',
        f'<text x="{tx}" y="{ty + 8}" style="font:700 {name_size}px {SANS};fill:{t["TEXT"]};letter-spacing:-1px">{escape(p["name"])}</text>',
        f'<text x="{tx}" y="{ty + 44}" style="font:500 {role_size}px {SANS};fill:{t["ACCENT"]}">{escape(p["role"])}</text>',
        f'<text x="{tx}" y="{ty + 72}" style="font:400 {role_size - 2}px {SANS};fill:{t["MUTED"]}">{escape(p["where"])}</text>',
    ]
    if not mobile:
        lines.append(
            f'<text x="{W - 24}" y="{H - 18}" text-anchor="end" style="font:500 11px {MONO};fill:{t["MUTED"]}">latent space · density landscape</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines)


# ---------------------------------------------------------------- divider
def divider(t):
    W, H = 1200, 24
    ring = []
    r, cx, cy = 9, W / 2, H / 2
    pts = [(cx + r * cos(pi / 6 + k * pi / 3), cy + r * sin(pi / 6 + k * pi / 3)) for k in range(6)]
    ring.append('<polygon points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + f'" fill="none" style="stroke:{t["ACCENT"]}" stroke-width="1.5"/>')
    ring.append(f'<circle cx="{cx}" cy="{cy}" r="4.2" fill="none" style="stroke:{t["ACCENT"]}" stroke-width="1.2"/>')
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
<defs>
<linearGradient id="l" x1="0" x2="1"><stop offset="0" stop-color="{t['ACCENT']}" stop-opacity="0"/><stop offset="1" stop-color="{t['ACCENT']}" stop-opacity=".7"/></linearGradient>
<linearGradient id="r" x1="0" x2="1"><stop offset="0" stop-color="{t['ACCENT']}" stop-opacity=".7"/><stop offset="1" stop-color="{t['ACCENT']}" stop-opacity="0"/></linearGradient>
</defs>
<rect x="80" y="{cy - 0.75}" width="{cx - 100 - 80}" height="1.5" fill="url(#l)"/>
<rect x="{cx + 20}" y="{cy - 0.75}" width="{cx - 100 - 80}" height="1.5" fill="url(#r)"/>
{''.join(ring)}
</svg>"""


# ---------------------------------------------------------------- toolbox (honeycomb of hexagons)
def hexagon(cx, cy, r):
    return " ".join(f"{cx + r * cos(pi / 6 + k * pi / 3):.1f},{cy + r * sin(pi / 6 + k * pi / 3):.1f}" for k in range(6))


def toolbox(t):
    R = 50
    dx = sqrt(3) * R + 6
    dy = 1.5 * R + 5
    rows = [8, 7]
    W = int(rows[0] * dx + 40)
    top = 20
    hexes = []
    i = 0
    for ri, n in enumerate(rows):
        for k in range(n):
            if i >= len(TOOLS):
                break
            cx = 20 + dx / 2 + k * dx + (dx / 2 if ri % 2 else 0)
            cy = top + R + ri * dy
            sym, name, fam = TOOLS[i]
            col = t["FAM"][fam]
            fs = 10 if len(name) > 9 else 11
            hexes.append(
                f'<g class="hx" style="animation-delay:{i * 0.12:.2f}s">'
                f'<polygon points="{hexagon(cx, cy, R)}" style="fill:{t["TILE"]};stroke:{col}" stroke-width="1.6"/>'
                f'<polygon points="{hexagon(cx, cy, R - 6)}" fill="none" style="stroke:{col}" stroke-opacity=".25"/>'
                f'<text x="{cx:.1f}" y="{cy + 2:.1f}" text-anchor="middle" style="font:700 26px {SANS};fill:{col}">{escape(sym)}</text>'
                f'<text x="{cx:.1f}" y="{cy + 22:.1f}" text-anchor="middle" style="font:500 {fs}px {MONO};fill:{t["TEXT"]}">{escape(name)}</text>'
                f'<text x="{cx - 20:.1f}" y="{cy - 22:.1f}" style="font:500 9px {MONO};fill:{t["MUTED"]}">{i + 1}</text>'
                "</g>"
            )
            i += 1
    legend_y = top + 2 * R + dy + 26
    leg, lx = [], 40
    for fam, label in FAMILIES:
        col = t["FAM"][fam]
        leg.append(f'<polygon points="{hexagon(lx + 6, legend_y - 4, 7)}" style="fill:{col}"/>')
        leg.append(f'<text x="{lx + 20}" y="{legend_y}" style="font:500 13px {MONO};fill:{t["MUTED"]}">{escape(label)}</text>')
        lx += 40 + len(label) * 8.2
    H = legend_y + 18
    css = f""".hx{{opacity:0;animation:in .5s ease-out forwards}}
@keyframes in{{from{{opacity:0;transform:translateY(6px)}}to{{opacity:1;transform:none}}}}
{REDUCED}"""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
        f'aria-label="Toolbox: {escape(", ".join(n for _, n, _ in TOOLS))}">\n<style>{css}</style>\n'
        + "\n".join(hexes) + "\n" + "\n".join(leg) + "\n</svg>"
    )


def main():
    for name, t in THEMES.items():
        files = {
            f"hero-{name}.svg": hero(t),
            f"hero-mobile-{name}.svg": hero(t, mobile=True),
            f"divider-{name}.svg": divider(t),
            f"toolbox-{name}.svg": toolbox(t),
        }
        for fn, svg in files.items():
            with open(os.path.join(HERE, fn), "w", encoding="utf-8") as f:
                f.write(svg)
            print("wrote", fn)


if __name__ == "__main__":
    main()
