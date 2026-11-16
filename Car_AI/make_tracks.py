
import json
import math
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw

W, H = 800, 600
WALL = 5                 # wall thickness in pixels
SS = 4                   # draw 4x bigger, then shrink, so the walls look smooth
MIN_WIDTH, MAX_WIDTH = 36, 115
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_tracks")


def catmull_rom_closed(points, per_segment=24):
    """Smooth closed curve that passes through every control point."""
    n = len(points)
    out = []
    for i in range(n):
        p0, p1, p2, p3 = points[(i - 1) % n], points[i], points[(i + 1) % n], points[(i + 2) % n]
        for k in range(per_segment):
            t = k / per_segment
            t2, t3 = t * t, t * t * t
            x = 0.5 * (2 * p1[0] + (-p0[0] + p2[0]) * t
                       + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                       + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * (2 * p1[1] + (-p0[1] + p2[1]) * t
                       + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                       + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            out.append((x, y))
    return np.array(out)


def resample(curve, spacing=4.0):
    """Points spaced evenly along the closed curve."""
    closed = np.vstack([curve, curve[:1]])
    seg = np.hypot(np.diff(closed[:, 0]), np.diff(closed[:, 1]))
    arc = np.concatenate([[0], np.cumsum(seg)])
    total = arc[-1]
    n = int(total // spacing)
    s = np.linspace(0, total, n, endpoint=False)
    return np.column_stack([np.interp(s, arc, closed[:, 0]), np.interp(s, arc, closed[:, 1])]), total


def fourier_curve(rng, twisty):
    """A closed loop built from a few sine waves: can make lobes, hairpins and chicanes."""
    n_modes = 3 + int(twisty * 8)
    decay = rng.uniform(0.5, 1.1) if twisty > 0.5 else rng.uniform(1.3, 2.2)
    t = np.linspace(0, 2 * math.pi, 360, endpoint=False)
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    for k in range(1, n_modes + 1):
        a = 1.0 / (k ** decay)
        x += a * rng.uniform(0.5, 1.0) * np.cos(k * t + rng.uniform(0, 2 * math.pi))
        y += a * rng.uniform(0.5, 1.0) * np.cos(k * t + rng.uniform(0, 2 * math.pi))
    margin = rng.uniform(40, 70)
    x = (x - x.min()) / (x.max() - x.min()) * (W - 2 * margin) + margin
    y = (y - y.min()) / (y.max() - y.min()) * (H - 2 * margin) + margin
    return np.column_stack([x, y])


def radial_curve(rng, twisty):
    """A blobby loop around the middle of the picture (simpler shapes)."""
    n_ctrl = rng.randint(5 + int(twisty * 4), 8 + int(twisty * 8))
    r_min = 0.55 - 0.33 * twisty
    pts = []
    for i in range(n_ctrl):
        ang = 2 * math.pi * i / n_ctrl + rng.uniform(-0.35, 0.35) * 2 * math.pi / n_ctrl
        r = rng.uniform(r_min, 1.0)
        pts.append((W / 2 + math.cos(ang) * 290 * r, H / 2 + math.sin(ang) * 200 * r))
    return catmull_rom_closed(pts)


def grid_polygon(rng, cell):
    """A random racing-circuit outline made only of straight lines and 90 degree corners.
    Think of colouring some squares on graph paper and walking round the edge of the coloured area."""
    cols = int(680 // cell)
    rows = int(480 // cell)
    if cols < 3 or rows < 2:
        return None
    total = cols * rows
    n = rng.randint(max(5, int(total * 0.28)), max(7, int(total * 0.58)))
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    occ = {(rng.randrange(cols), rng.randrange(rows))}
    guard = 0
    while len(occ) < n and guard < 8000:
        guard += 1
        bx, by = rng.choice(sorted(occ))
        dx, dy = rng.choice(dirs)
        cand = (bx + dx, by + dy)
        if cand in occ or not (0 <= cand[0] < cols and 0 <= cand[1] < rows):
            continue
        touching = sum((cand[0] + ex, cand[1] + ey) in occ for ex, ey in dirs)
        if touching > 1 and rng.random() < 0.9:               # prefer thin branches over solid blobs
            continue
        pinch = False
        for ex, ey in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            if (cand[0] + ex, cand[1] + ey) in occ:
                if (cand[0] + ex, cand[1]) not in occ and (cand[0], cand[1] + ey) not in occ:
                    pinch = True
        if pinch:
            continue
        occ.add(cand)
    # no holes: every empty square must be able to reach the outside
    seen, stack = set(), [(-1, -1)]
    while stack:
        x, y = stack.pop()
        if (x, y) in seen or (x, y) in occ or not (-1 <= x <= cols and -1 <= y <= rows):
            continue
        seen.add((x, y))
        stack += [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    for x in range(cols):
        for y in range(rows):
            if (x, y) not in occ and (x, y) not in seen:
                return None
    # no pinch points (two squares touching only at a corner)
    for x in range(-1, cols):
        for y in range(-1, rows):
            a, b, c, d = (x, y) in occ, (x + 1, y) in occ, (x, y + 1) in occ, (x + 1, y + 1) in occ
            if (a and d and not b and not c) or (b and c and not a and not d):
                return None
    # walk round the outline
    edges = {}
    for (x, y) in occ:
        if (x, y - 1) not in occ:
            edges[(x, y)] = (x + 1, y)
        if (x + 1, y) not in occ:
            edges[(x + 1, y)] = (x + 1, y + 1)
        if (x, y + 1) not in occ:
            edges[(x + 1, y + 1)] = (x, y + 1)
        if (x - 1, y) not in occ:
            edges[(x, y + 1)] = (x, y)
    start = next(iter(edges))
    loop = [start]
    cur = edges[start]
    while cur != start:
        loop.append(cur)
        cur = edges[cur]
        if len(loop) > len(edges) + 2:
            return None
    if len(loop) != len(edges):
        return None
    # keep only the corners
    corners = []
    m = len(loop)
    for i in range(m):
        p, q, r = loop[i - 1], loop[i], loop[(i + 1) % m]
        if (q[0] - p[0], q[1] - p[1]) != (r[0] - q[0], r[1] - q[1]):
            corners.append(q)
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    ox = (W - (max(xs) - min(xs)) * cell) / 2 - min(xs) * cell
    oy = (H - (max(ys) - min(ys)) * cell) / 2 - min(ys) * cell
    poly = [(ox + p[0] * cell, oy + p[1] * cell) for p in corners]
    if rng.random() < 0.5:
        poly.reverse()
    return poly


def chamfer(poly, rng, chance):
    """Cut some 90 degree corners into two 45 degree turns."""
    out = []
    n = len(poly)
    for i in range(n):
        p, q, r = np.array(poly[i - 1]), np.array(poly[i]), np.array(poly[(i + 1) % n])
        l1, l2 = np.linalg.norm(p - q), np.linalg.norm(r - q)
        if rng.random() < chance and min(l1, l2) > 120:
            cut = min(0.4 * min(l1, l2), rng.uniform(35, 75))
            out.append(tuple(q + (p - q) / l1 * cut))
            out.append(tuple(q + (r - q) / l2 * cut))
        else:
            out.append(tuple(q))
    return out


def rounded_path(poly, rng, r_min, sharp_chance):
    """Turn a polygon into a smooth road line by rounding every corner. None if a corner will not fit."""
    n = len(poly)
    pieces = []
    for i in range(n):
        u, v, x = np.array(poly[i - 1]), np.array(poly[i]), np.array(poly[(i + 1) % n])
        l1, l2 = np.linalg.norm(u - v), np.linalg.norm(x - v)
        a, b = (u - v) / l1, (x - v) / l2
        theta = math.acos(max(-1.0, min(1.0, float(a @ b))))        # angle between the two sides
        if theta > math.pi - 0.05:
            pieces.append(([tuple(v)], v, v))
            continue
        room = 0.5 * min(l1, l2) * math.tan(theta / 2)               # biggest radius that still fits
        want = r_min if rng.random() < sharp_chance else r_min + rng.uniform(10, 70)
        r = min(want, room)
        if r < r_min - 0.5:
            return None
        t = r / math.tan(theta / 2)
        t1, t2 = v + a * t, v + b * t
        bis = (a + b) / np.linalg.norm(a + b)
        centre = v + bis * (r / math.sin(theta / 2))
        a1 = math.atan2(*(t1 - centre)[::-1])
        a2 = math.atan2(*(t2 - centre)[::-1])
        sweep = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
        k = max(3, int(abs(sweep) * r / 3))
        arc = [tuple(centre + r * np.array([math.cos(a1 + sweep * j / k), math.sin(a1 + sweep * j / k)]))
               for j in range(k + 1)]
        pieces.append((arc, t1, t2))
    pts = []
    for i in range(n):
        pts.extend(pieces[i][0])
        nxt = pieces[(i + 1) % n][0][0]
        end = pieces[i][0][-1]
        gap = math.hypot(nxt[0] - end[0], nxt[1] - end[1])
        for j in range(1, int(gap // 4)):
            f = j / (gap / 4)
            pts.append((end[0] + (nxt[0] - end[0]) * f, end[1] + (nxt[1] - end[1]) * f))
    return np.array(pts)


def try_make(rng, wide=False):
    """Draw a random loop first, then pick the widest road that physically fits it."""
    preferred = None
    if wide or rng.random() < 0.9:                            # circuit with straights and sharp corners
        if wide:                                              # every 4th track gets a really wide road
            cell = rng.choice([125, 145, 165])
            width = rng.uniform(80, min(MAX_WIDTH, cell - 38))
        else:
            cell = rng.choice([84, 96, 110, 125, 145, 165])
            width = MIN_WIDTH + (min(MAX_WIDTH, cell - 38) - MIN_WIDTH) * rng.random()
        poly = grid_polygon(rng, cell)
        if poly is None:
            return None
        poly = chamfer(poly, rng, rng.choice([0.0, 0.0, 0.25, 0.5]))
        curve = rounded_path(poly, rng, r_min=width / 2 + 13, sharp_chance=0.65)
        if curve is None:
            return None
        preferred = width
    else:                                                     # smooth organic loop
        twisty = rng.random() ** 0.6
        curve = fourier_curve(rng, twisty) if rng.random() < 0.75 else radial_curve(rng, twisty)
    c, total = resample(curve)
    if not 900 <= total <= 5200:
        return None
    n = len(c)
    ds = total / n
    t = np.roll(c, -1, axis=0) - np.roll(c, 1, axis=0)
    t /= np.hypot(t[:, 0], t[:, 1])[:, None]
    heading = np.unwrap(np.arctan2(t[:, 1], t[:, 0]))
    kappa = np.abs(np.gradient(heading)) / ds
    # how wide can the road be before the walls fold over, touch another part of the road, or leave the picture?
    w_curve = 2 * (1.0 / max(kappa.max(), 1e-6) - 6)
    sub = c[::2]
    m = len(sub)
    d = np.hypot(sub[:, None, 0] - sub[None, :, 0], sub[:, None, 1] - sub[None, :, 1])
    idx = np.arange(m)
    along = np.abs(idx[:, None] - idx[None, :])
    along = np.minimum(along, m - along) * ds * 2
    far = along > 200
    w_prox = (d[far].min() if far.any() else 999) - (2 * WALL + 18)
    edge = min(c[:, 0].min(), W - c[:, 0].max(), c[:, 1].min(), H - c[:, 1].max())
    w_edge = 2 * (edge - 10)
    w_max = min(MAX_WIDTH, w_curve, w_prox, w_edge)
    if w_max < MIN_WIDTH:
        return None
    if preferred is not None:
        if preferred > w_max + 0.5:
            return None
        width = preferred
    else:
        width = MIN_WIDTH + (w_max - MIN_WIDTH) * rng.random() ** 1.3
    normal = np.column_stack([-t[:, 1], t[:, 0]])
    left = c + normal * width / 2
    right = c - normal * width / 2
    # start on the straightest stretch
    window = int(200 / ds)
    smooth = np.convolve(np.tile(kappa, 3), np.ones(window) / window, mode="same")[n:2 * n]
    start_i = int(np.argmin(smooth))
    start = c[start_i]
    angle = (math.degrees(heading[start_i]) + 180) % 360 - 180
    return {"left": left, "right": right, "start": start, "angle": angle, "width": width, "length": total}


def draw(track):
    big = Image.new("L", (W * SS, H * SS), 255)
    d = ImageDraw.Draw(big)
    for edge in (track["left"], track["right"]):
        pts = [(x * SS, y * SS) for x, y in edge] + [(edge[0][0] * SS, edge[0][1] * SS)]
        d.line(pts, fill=0, width=WALL * SS, joint="curve")
    small = big.resize((W, H), Image.LANCZOS).point(lambda v: 0 if v < 140 else 255)
    return small.convert("RGB")


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    rng = random.Random(seed)
    os.makedirs(OUT, exist_ok=True)
    made = 0
    attempts = 0
    while made < count and attempts < 5000:
        attempts += 1
        track = try_make(rng, wide=(made % 4 == 3))
        if track is None:
            continue
        made += 1
        stem = f"gen_{made:02d}"
        draw(track).save(os.path.join(OUT, stem + ".png"))
        info = {"start": [round(float(track["start"][0]), 1), round(float(track["start"][1]), 1)],
                "angle": round(track["angle"], 1)}
        with open(os.path.join(OUT, stem + ".json"), "w") as f:
            json.dump(info, f)
        print(f"{stem}: road width {track['width']:.0f}px, length {track['length']:.0f}px, start {info['start']} angle {info['angle']}")
    print(f"done: {made} tracks in {OUT}  ({attempts} attempts)")


if __name__ == "__main__":
    main()