
import bisect
import json
import math
import os
import random
import sys

import numpy as np
import pygame
import torch
from pygame import gfxdraw

import car_sim as cs

# ---------------------------------------------------------------- settings
TITLE = "AI Race Car"
SUBTITLE = "A Deep Q-Learning driver trained in simulation"
PANEL_W = 200                       # width of the dashboard on the right
WIN_W, WIN_H = cs.W + PANEL_W, cs.H
FPS = 60
SPEED_LEVELS = [0.25, 0.5, 1, 2, 4, 8, 16]   # simulation ticks per frame
DEFAULT_SPEED_INDEX = 2
AUTO_NEXT_FRAMES = 180              # "run all" mode waits this long on the result card
CUSTOM_DIR = os.path.join(cs.BASE, "custom_tracks")
RECORDS_PATH = os.path.join(cs.BASE, "showcase_records.json")
SHOT_DIR = os.path.join(cs.BASE, "screenshots")
TRAINED = set(cs.NAMES)             # tracks the AI saw during training

# colours
BG = (14, 17, 23)
PANEL = (21, 25, 34)
CARD = (28, 34, 46)
CARD_EDGE = (58, 68, 90)
TEXT = (232, 236, 244)
MUTED = (140, 150, 170)
ACCENT = (0, 209, 255)
AMBER = (255, 176, 0)
GOOD = (46, 204, 113)
BAD = (235, 77, 75)


# ---------------------------------------------------------------- helpers
def visible_names():
    """Tracks shown in the menu. The random training tracks (gen_...) stay hidden."""
    return [n for n in cs.NAMES if not n.startswith("gen_")]


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(lerp(c1[i], c2[i], t)) for i in range(3))


def speed_color(frac):
    """0 = slow (red) ... 1 = fast (cyan)."""
    stops = [(235, 77, 75), (255, 176, 0), (46, 204, 113), (0, 209, 255)]
    frac = max(0.0, min(1.0, frac)) * (len(stops) - 1)
    i = min(int(frac), len(stops) - 2)
    return lerp_color(stops[i], stops[i + 1], frac - i)


_fonts = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _fonts:
        _fonts[key] = pygame.font.SysFont("helveticaneue,arial,dejavusans", size, bold=bold)
    return _fonts[key]


def draw_text(surf, s, size, color, pos, bold=False, anchor="topleft"):
    img = font(size, bold).render(s, True, color)
    rect = img.get_rect(**{anchor: pos})
    surf.blit(img, rect)
    return rect


def rounded(surf, color, rect, radius=12, width=0):
    pygame.draw.rect(surf, color, rect, width, border_radius=radius)


def translucent_rect(surf, color, rect, radius=12):
    tmp = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
    pygame.draw.rect(tmp, color, (0, 0, rect[2], rect[3]), border_radius=radius)
    surf.blit(tmp, (rect[0], rect[1]))


def round_top(surf, radius=14):
    """Copy of a picture with its two top corners rounded."""
    w, h = surf.get_size()
    out = pygame.Surface((w, h), pygame.SRCALPHA)
    out.blit(surf, (0, 0))
    mask = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, w, h + radius), border_radius=radius)
    out.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    return out


def fmt_ticks(ticks):
    return f"{ticks} ticks ({ticks / FPS:.1f} s)"


# ---------------------------------------------------------------- track artwork
def make_skin(name):
    """Turns the black-and-white map into asphalt, grass and red/white curbs."""
    road = cs.ROADS[name]
    block = np.array(road["block"], dtype=np.uint8)      # [x][y]
    dist = np.array(road["dist"], dtype=np.int32)
    w, h = block.shape
    shade = 0.82 + 0.28 * np.linspace(0, 1, h)
    img = np.zeros((w, h, 3), dtype=np.uint8)
    grass = np.array([36, 92, 56], dtype=float)
    img[:, :, :] = (grass[None, None, :] * shade[None, :, None]).astype(np.uint8)
    img[dist >= 0] = (60, 64, 72)                         # asphalt
    xs = np.arange(w)[:, None]
    ys = np.arange(h)[None, :]
    stripe = ((xs + ys) // 7) % 2 == 0
    walls = block == 1
    img[walls & stripe] = (214, 48, 49)
    img[walls & ~stripe] = (244, 244, 244)
    skin = pygame.surfarray.make_surface(img).convert()
    draw_finish_line(skin, road["cut"], road["angle"])
    return skin


def draw_finish_line(surf, ends, angle):
    (x1, y1), (x2, y2) = ends
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 2:
        return
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    fx, fy = math.cos(angle), math.sin(angle)
    size = 5
    for i in range(int(length // size) + 1):
        for row in (0, 1):
            col = (255, 255, 255) if (i + row) % 2 == 0 else (18, 18, 18)
            ox = x1 + ux * i * size + fx * (row - 1) * size
            oy = y1 + uy * i * size + fy * (row - 1) * size
            pts = [
                (ox, oy),
                (ox + ux * size, oy + uy * size),
                (ox + ux * size + fx * size, oy + uy * size + fy * size),
                (ox + fx * size, oy + fy * size),
            ]
            pygame.draw.polygon(surf, col, pts)


def local_point(car, u, v):
    fx, fy = math.cos(car.angle), math.sin(car.angle)
    return (car.x + fx * u - fy * v, car.y + fy * u + fx * v)


def aa_poly(surf, color, pts):
    gfxdraw.filled_polygon(surf, [(int(round(x)), int(round(y))) for x, y in pts], color)
    gfxdraw.aapolygon(surf, [(int(round(x)), int(round(y))) for x, y in pts], color)


def draw_car_fx(fx_layer, car, braking):
    """Soft shadow, headlight beam and brake glow (drawn BEFORE the car body)."""
    hl, hw = cs.CAR_LENGTH / 2, cs.CAR_WIDTH / 2
    shadow = [local_point(car, u, v) for u, v in
              [(hl, -hw), (hl, hw), (-hl, hw), (-hl, -hw)]]
    aa_poly(fx_layer, (0, 0, 0, 90), [(x + 2, y + 2) for x, y in shadow])
    beam = [local_point(car, hl - 1, -hw + 2), local_point(car, hl - 1, hw - 2),
            local_point(car, hl + 46, 15), local_point(car, hl + 46, -15)]
    aa_poly(fx_layer, (255, 255, 190, 34), beam)
    if braking:
        for v in (-hw + 2, hw - 2):
            x, y = local_point(car, -hl + 0.5, v)
            gfxdraw.filled_circle(fx_layer, int(x), int(y), 4, (255, 40, 40, 70))


def draw_car_body(surf, car, braking):
    hl, hw = cs.CAR_LENGTH / 2, cs.CAR_WIDTH / 2
    for u in (hl - 4, -hl + 4):                       # wheels
        for v in (-hw - 0.8, hw + 0.8):
            wheel = [local_point(car, u + du, v + dv) for du, dv in
                     [(-2, -1), (2, -1), (2, 1), (-2, 1)]]
            aa_poly(surf, (20, 20, 24), wheel)
    body = [local_point(car, u, v) for u, v in [
        (hl - 3, -hw), (hl, -hw + 2), (hl, hw - 2), (hl - 3, hw),
        (-hl + 2, hw), (-hl, hw - 2), (-hl, -hw + 2), (-hl + 2, -hw)]]
    aa_poly(surf, (0, 168, 232), body)
    stripe = [local_point(car, u, v) for u, v in
              [(hl - 1, -0.9), (hl - 1, 0.9), (-hl + 1, 0.9), (-hl + 1, -0.9)]]
    aa_poly(surf, (235, 245, 255), stripe)
    cabin = [local_point(car, u, v) for u, v in
             [(2.5, -hw + 2), (2.5, hw - 2), (-3.5, hw - 2), (-3.5, -hw + 2)]]
    aa_poly(surf, (16, 28, 48), cabin)
    for v in (-hw + 2, hw - 2):                       # lights
        x, y = local_point(car, hl - 0.5, v)
        gfxdraw.filled_circle(surf, int(x), int(y), 1, (255, 244, 170))
        x, y = local_point(car, -hl + 0.5, v)
        gfxdraw.filled_circle(surf, int(x), int(y), 1, (255, 40, 40) if braking else (120, 22, 22))


# ---------------------------------------------------------------- records
def load_records():
    try:
        with open(RECORDS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def save_records(rec):
    try:
        with open(RECORDS_PATH, "w") as f:
            json.dump(rec, f, indent=2)
    except Exception as e:
        print("could not save records:", e)


# ---------------------------------------------------------------- custom tracks
def road_axis(wall, start):
    """Direction (degrees, 0..180) of the road at `start`: the line along which you can see furthest both ways."""
    w, h = wall.shape

    def free(a):
        ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
        d = 0.0
        while d < 300:
            x, y = int(start[0] + ca * d), int(start[1] + sa * d)
            if not (0 <= x < w and 0 <= y < h) or wall[x, y]:
                break
            d += 1.0
        return d

    best_a, best = 0, -1.0
    for a in range(0, 180, 3):
        score = free(a) + free(a + 180)
        if score > best:
            best, best_a = score, a
    return best_a


def angle_gap(a, b):
    return abs((a - b + 180) % 360 - 180)


def register_custom(name, path, start, angle_deg):
    """Builds the road for a new image. Returns an error message or None."""
    cs.TRACKS[name] = {"image": path, "start": (float(start[0]), float(start[1])), "angle": float(angle_deg)}
    try:
        cs.ROADS[name] = cs.build_road(name)
    except SystemExit as e:
        del cs.TRACKS[name]
        return str(e)
    except Exception as e:
        del cs.TRACKS[name]
        return f"could not read that track: {e}"
    if name not in cs.NAMES:
        cs.NAMES.append(name)
    road = cs.ROADS[name]
    if road.get("leak"):
        return "warning: the road reaches a corner of the picture, so a wall probably has a gap"
    idx = min(bisect.bisect_left(road["arcs"], 40), len(road["center"]) - 1)
    cx, cy = road["center"][idx]
    sx, sy = road["start"]
    off = abs((math.atan2(cy - sy, cx - sx) - road["angle"] + math.pi) % (2 * math.pi) - math.pi)
    if math.degrees(off) > 60:
        return "warning: the start direction does not follow the road, the car may turn around"
    return None


def skipped_reasons():
    """Why some entries of TRACKS are missing from the menu."""
    out = []
    for n, tr in cs.TRACKS.items():
        if n in cs.NAMES or n.startswith("gen_"):
            continue
        if not os.path.exists(os.path.join(cs.BASE, tr["image"])):
            out.append(f'"{n}" skipped: image file not found ({tr["image"]})')
        elif "angle" not in tr and not tr.get("aim"):
            out.append(f'"{n}" skipped: add an "angle" or an "aim" in TRACKS')
        else:
            out.append(f'"{n}" skipped')
    return out


def scan_custom():
    """Registers custom tracks that already have a .json; returns the ones that still need a start."""
    pending = []
    if not os.path.isdir(CUSTOM_DIR):
        return pending
    for fname in sorted(os.listdir(CUSTOM_DIR)):
        if not fname.lower().endswith(".png"):
            continue
        stem = os.path.splitext(fname)[0]
        if stem in cs.NAMES:
            continue
        png = os.path.join(CUSTOM_DIR, fname)
        sidecar = os.path.join(CUSTOM_DIR, stem + ".json")
        if os.path.exists(sidecar):
            try:
                with open(sidecar) as f:
                    d = json.load(f)
                err = register_custom(stem, png, d["start"], d["angle"])
                if err:
                    print("custom track", stem, "->", err)
            except Exception as e:
                print("bad sidecar for", stem, e)
        else:
            pending.append((stem, png))
    return pending


# ================================================================== the app
class App:
    def __init__(self, model):
        self.model = model
        self.screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.SCALED | pygame.RESIZABLE)
        pygame.display.set_caption(TITLE)
        self.clock = pygame.time.Clock()
        self.records = load_records()
        self.skins = {}
        self.thumbs = {}
        self.pending = scan_custom()
        self.mode = "menu"            # menu | run | place
        self.page = 0
        self.toast = ("", 0)
        self.running = True
        # run state
        self.name = None
        self.env = None
        self.state = None
        self.trail = None
        self.fx = pygame.Surface((cs.W, cs.H), pygame.SRCALPHA)
        self.speed_index = DEFAULT_SPEED_INDEX
        self.acc = 0.0
        self.paused = False
        self.show_sensors = True
        self.show_trail = True
        self.show_debug = False
        self.result = None
        self.confetti = []
        self.run_all = False
        self.all_queue = []
        self.wait = 0
        self.action = 4
        self.prev_pos = (0, 0)
        self.last_braking = False
        # placement state
        self.place_name = None
        self.place_path = None
        self.place_img = None
        self.place_stage = 0
        self.place_start = (0, 0)
        self.place_wall = None
        self.place_axis = 0

    # ---------------------------------------------------------- utilities
    def say(self, msg, frames=240):
        self.toast = (msg, frames)

    def skin(self, name):
        if name not in self.skins:
            self.skins[name] = make_skin(name)
        return self.skins[name]

    def thumb(self, name):
        if name not in self.thumbs:
            self.thumbs[name] = round_top(pygame.transform.smoothscale(self.skin(name), (240, 180)))
        return self.thumbs[name]

    def best(self, name):
        return self.records.get(name)

    # ---------------------------------------------------------- menu layout
    def items(self):
        its = [("track", n) for n in visible_names()]
        its += [("pending", p) for p in self.pending]
        if len(visible_names()) >= 2:
            its.append(("all", None))
        return its

    def card_rects(self):
        its = self.items()
        per_page = 6
        pages = max(1, math.ceil(len(its) / per_page))
        self.page = max(0, min(self.page, pages - 1))
        chunk = its[self.page * per_page:(self.page + 1) * per_page]
        rects = []
        for i, it in enumerate(chunk):
            col, row = i % 3, i // 3
            x = 100 + col * (240 + 40)
            y = 122 + row * 218
            rects.append((pygame.Rect(x, y, 240, 204), it, self.page * per_page + i + 1))
        return rects, pages

    # ---------------------------------------------------------- starting things
    def start_run(self, name, keep_queue=False):
        if not keep_queue:
            self.run_all = False
        cs.set_track(name)
        self.name = name
        self.env = cs.Env()
        self.state = self.env.reset()
        self.trail = pygame.Surface((cs.W, cs.H), pygame.SRCALPHA)
        self.prev_pos = (self.env.car.x, self.env.car.y)
        self.result = None
        self.confetti = []
        self.acc = 0.0
        self.paused = False
        self.action = 4
        self.wait = 0
        self.mode = "run"

    def start_all(self):
        self.all_queue = visible_names()
        self.run_all = True
        self.next_in_queue()

    def next_in_queue(self):
        if not self.all_queue:
            self.all_queue = visible_names()
        name = self.all_queue.pop(0)
        self.start_run(name, keep_queue=True)

    def next_track(self):
        names = visible_names()
        if not names:
            return
        i = names.index(self.name) if self.name in names else -1
        self.start_run(names[(i + 1) % len(names)], keep_queue=self.run_all)

    def snapped_angle(self, pos):
        """The road direction closest to where the mouse is pointing (degrees)."""
        sx, sy = self.place_start
        want = math.degrees(math.atan2(pos[1] - sy, pos[0] - sx))
        options = [self.place_axis, self.place_axis + 180]
        pick = min(options, key=lambda a: angle_gap(a, want))
        return (pick + 180) % 360 - 180

    def begin_place(self, stem, png):
        try:
            img = pygame.image.load(png).convert()
        except Exception as e:
            self.say(f"cannot open image: {e}")
            return
        if img.get_size() != (cs.W, cs.H):
            self.say(f"image must be exactly {cs.W}x{cs.H}, this one is {img.get_size()}")
            return
        self.place_name, self.place_path, self.place_img = stem, png, img
        arr = pygame.surfarray.array3d(img).astype(int)
        self.place_wall = arr.sum(axis=2) < 150
        self.place_axis = 0
        self.place_stage = 0
        self.mode = "place"

    # ---------------------------------------------------------- simulation
    def step_once(self):
        action = cs.pick_action(self.model, self.state, 0.0)
        self.state, reward, done = self.env.step(action)
        self.action = action
        car = self.env.car
        self.last_braking = (action % 3 - 1) == -1
        col = speed_color((car.speed - cs.MIN_SPEED) / (cs.MAX_SPEED - cs.MIN_SPEED))
        pygame.draw.line(self.trail, (*col, 230), self.prev_pos, (car.x, car.y), 3)
        self.prev_pos = (car.x, car.y)
        if done:
            self.finish_run()

    def finish_run(self):
        env = self.env
        if env.finished:
            kind = "finish"
        elif env.steps >= cs.MAX_STEPS:
            kind = "timeout"
        else:
            kind = "crash"
        progress = min(1.0, env.max_prog / cs.LAP_LENGTH)
        new_best = False
        if kind == "finish":
            old = self.records.get(self.name)
            if old is None or env.steps < old:
                self.records[self.name] = env.steps
                save_records(self.records)
                new_best = True
            for _ in range(170):
                self.confetti.append([random.uniform(0, cs.W), random.uniform(-200, 0),
                                      random.uniform(-0.6, 0.6), random.uniform(1.5, 4.0),
                                      random.choice([ACCENT, AMBER, GOOD, (255, 255, 255), BAD]),
                                      random.uniform(0, 6.28)])
        self.result = {"kind": kind, "ticks": env.steps, "score": env.score(),
                       "progress": progress, "new_best": new_best}
        self.wait = 0

    # ---------------------------------------------------------- events
    def handle(self, ev):
        if ev.type == pygame.QUIT:
            self.running = False
        elif ev.type == pygame.KEYDOWN:
            self.on_key(ev)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            self.on_click(ev.pos)

    def on_key(self, ev):
        k = ev.key
        if k == pygame.K_f:
            pygame.display.toggle_fullscreen()
            return
        if self.mode == "menu":
            if k == pygame.K_ESCAPE or k == pygame.K_q:
                self.running = False
            elif k == pygame.K_a and len(visible_names()) >= 2:
                self.start_all()
            elif k == pygame.K_r:
                self.pending = scan_custom()
                self.say("rescanned the custom_tracks folder")
            elif k == pygame.K_RIGHT:
                self.page += 1
            elif k == pygame.K_LEFT:
                self.page -= 1
            elif pygame.K_1 <= k <= pygame.K_9:
                idx = k - pygame.K_1
                for rect, it, num in self.card_rects()[0]:
                    if num == idx + 1:
                        self.activate(it)
        elif self.mode == "place":
            if k == pygame.K_ESCAPE:
                self.mode = "menu"
        elif self.mode == "run":
            if k == pygame.K_ESCAPE or k == pygame.K_m:
                self.run_all = False
                self.mode = "menu"
            elif k == pygame.K_SPACE:
                self.paused = not self.paused
            elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS, pygame.K_UP):
                self.speed_index = min(len(SPEED_LEVELS) - 1, self.speed_index + 1)
            elif k in (pygame.K_MINUS, pygame.K_KP_MINUS, pygame.K_DOWN):
                self.speed_index = max(0, self.speed_index - 1)
            elif k == pygame.K_s:
                self.show_sensors = not self.show_sensors
            elif k == pygame.K_t:
                self.show_trail = not self.show_trail
            elif k == pygame.K_d:
                self.show_debug = not self.show_debug
            elif k == pygame.K_r:
                self.start_run(self.name, keep_queue=self.run_all)
            elif k == pygame.K_n:
                self.next_track()
            elif k == pygame.K_p:
                os.makedirs(SHOT_DIR, exist_ok=True)
                path = os.path.join(SHOT_DIR, f"{self.name}_{pygame.time.get_ticks()}.png")
                pygame.image.save(self.screen, path)
                self.say("saved " + os.path.basename(path), 120)
            elif k == pygame.K_RIGHT and self.paused and not self.result:
                self.step_once()

    def activate(self, it):
        kind, data = it
        if kind == "track":
            self.start_run(data)
        elif kind == "all":
            self.start_all()
        elif kind == "pending":
            self.begin_place(*data)

    def on_click(self, pos):
        if self.mode == "menu":
            rects, pages = self.card_rects()
            for rect, it, num in rects:
                if rect.collidepoint(pos):
                    self.activate(it)
                    return
            if pages > 1:
                if pygame.Rect(30, 300, 50, 60).collidepoint(pos):
                    self.page -= 1
                if pygame.Rect(WIN_W - 80, 300, 50, 60).collidepoint(pos):
                    self.page += 1
        elif self.mode == "place":
            x, y = pos
            if x >= cs.W:
                return
            if self.place_stage == 0:
                self.place_start = (x, y)
                self.place_axis = road_axis(self.place_wall, (x, y))
                self.place_stage = 1
            else:
                angle = self.snapped_angle((x, y))
                err = register_custom(self.place_name, self.place_path, self.place_start, angle)
                if err and not err.startswith("warning"):
                    self.say(err, 360)
                    self.place_stage = 0
                    return
                try:
                    sidecar = os.path.join(CUSTOM_DIR, self.place_name + ".json")
                    with open(sidecar, "w") as f:
                        json.dump({"start": list(self.place_start), "angle": angle}, f)
                except Exception as e:
                    print("could not save sidecar:", e)
                self.pending = [p for p in self.pending if p[0] != self.place_name]
                if err:
                    self.say(err, 360)
                self.start_run(self.place_name)

    # ---------------------------------------------------------- update
    def update(self):
        if self.toast[1] > 0:
            self.toast = (self.toast[0], self.toast[1] - 1)
        if self.mode != "run":
            return
        if not self.paused and not self.result:
            self.acc += SPEED_LEVELS[self.speed_index]
            n = int(self.acc)
            self.acc -= n
            for _ in range(n):
                self.step_once()
                if self.result:
                    break
        if self.result:
            self.wait += 1
            if self.run_all and self.wait > AUTO_NEXT_FRAMES:
                self.next_in_queue()
        for p in self.confetti:
            p[0] += p[2]
            p[1] += p[3]
            p[5] += 0.15
        self.confetti = [p for p in self.confetti if p[1] < cs.H + 20]

    # ---------------------------------------------------------- drawing: menu
    def draw_menu(self):
        s = self.screen
        s.fill(BG)
        draw_text(s, TITLE, 44, TEXT, (60, 28), bold=True)
        draw_text(s, SUBTITLE, 18, MUTED, (62, 80))
        draw_text(s, "Choose a track", 18, ACCENT, (WIN_W - 60, 52), bold=True, anchor="topright")
        for i, why in enumerate(skipped_reasons()[:3]):
            draw_text(s, why, 12, AMBER, (WIN_W - 60, 80 + i * 16), anchor="topright")
        mouse = pygame.mouse.get_pos()
        rects, pages = self.card_rects()
        for rect, (kind, data), num in rects:
            hover = rect.collidepoint(mouse)
            rounded(s, CARD, rect, 14)
            rounded(s, ACCENT if hover else CARD_EDGE, rect, 14, 2 if hover else 1)
            thumb_rect = pygame.Rect(rect.x, rect.y, 240, 180)
            if kind == "track":
                s.blit(self.thumb(data), thumb_rect)
                label = data
                best = self.best(data)
                sub = f"best {best} ticks" if best else "no run yet"
                badge = ("TRAINED", GOOD) if data in TRAINED else ("UNSEEN", AMBER)
            elif kind == "pending":
                stem, png = data
                key = ("pending", stem)
                if key not in self.thumbs:
                    try:
                        img = pygame.image.load(png).convert()
                        self.thumbs[key] = round_top(pygame.transform.smoothscale(img, (240, 180)))
                    except Exception:
                        self.thumbs[key] = round_top(pygame.Surface((240, 180)))
                s.blit(self.thumbs[key], thumb_rect)
                label = stem
                sub = "click to place the start"
                badge = ("NEW", AMBER)
            else:
                tile = pygame.Surface((240, 180))
                tile.fill((20, 26, 38))
                s.blit(round_top(tile), thumb_rect)
                tri = [(rect.centerx - 22, rect.y + 52), (rect.centerx - 22, rect.y + 128),
                       (rect.centerx + 38, rect.y + 90)]
                aa_poly(s, ACCENT, tri)
                label = "RUN ALL"
                sub = "plays them all"
                badge = None
            # rounded thumbnail corners: draw card border over the image
            rounded(s, ACCENT if hover else CARD_EDGE, rect, 14, 2 if hover else 1)
            draw_text(s, label, 18, TEXT, (rect.x + 12, rect.y + 184 - 2), bold=True)
            draw_text(s, sub, 12, MUTED, (rect.right - 12, rect.y + 190), anchor="topright")
            if badge:
                text, col = badge
                r = draw_text(s, text, 11, BG, (rect.x + 10, rect.y + 10), bold=True)
                bg = r.inflate(12, 6)
                rounded(s, col, bg, 8)
                draw_text(s, text, 11, BG, bg.center, bold=True, anchor="center")
            if num <= 9:
                nb = pygame.Rect(rect.right - 34, rect.y + 8, 26, 26)
                rounded(s, (0, 0, 0), nb, 8)
                draw_text(s, str(num), 14, TEXT, nb.center, bold=True, anchor="center")
        if pages > 1:
            draw_text(s, "<", 40, MUTED, (54, 330), bold=True, anchor="center")
            draw_text(s, ">", 40, MUTED, (WIN_W - 54, 330), bold=True, anchor="center")
            draw_text(s, f"page {self.page + 1} of {pages}", 13, MUTED, (WIN_W // 2, 566), anchor="center")
        hint = "Click a card or press its number   |   A run all   |   R rescan custom_tracks   |   F fullscreen   |   ESC quit"
        draw_text(s, hint, 13, MUTED, (WIN_W // 2, 582), anchor="center")
        if not visible_names():
            draw_text(s, "No tracks found", 18, BAD, (WIN_W // 2, 300), anchor="center")

    # ---------------------------------------------------------- drawing: placement
    def draw_place(self):
        s = self.screen
        s.fill(BG)
        s.blit(self.place_img, (0, 0))
        pygame.draw.rect(s, PANEL, (cs.W, 0, PANEL_W, WIN_H))
        draw_text(s, "NEW TRACK", 20, AMBER, (cs.W + 14, 20), bold=True)
        draw_text(s, self.place_name, 15, TEXT, (cs.W + 14, 50))
        if self.place_stage == 0:
            lines = ["Step 1 of 2", "Click where the car", "should START", "(middle of the road)."]
        else:
            lines = ["Step 2 of 2", "Move the mouse the", "way the car should", "drive, then click.", "The arrow snaps to", "the road for you."]
        for i, ln in enumerate(lines):
            draw_text(s, ln, 15, TEXT if i else ACCENT, (cs.W + 14, 100 + i * 24), bold=(i == 0))
        draw_text(s, "ESC = back", 13, MUTED, (cs.W + 14, 560))
        mx, my = pygame.mouse.get_pos()
        if self.place_stage == 1:
            sx, sy = self.place_start
            ang = math.radians(self.snapped_angle((mx, my)))
            back = ang + math.pi
            pygame.draw.line(s, (120, 100, 40), (sx, sy), (sx + math.cos(back) * 50, sy + math.sin(back) * 50), 2)
            tip = (sx + math.cos(ang) * 60, sy + math.sin(ang) * 60)
            pygame.draw.line(s, AMBER, (sx, sy), tip, 4)
            for da in (2.6, -2.6):
                pygame.draw.line(s, AMBER, tip, (tip[0] + math.cos(ang + da) * 14, tip[1] + math.sin(ang + da) * 14), 4)
            pygame.draw.circle(s, AMBER, (sx, sy), 6)
        elif mx < cs.W:
            pygame.draw.circle(s, AMBER, (mx, my), 6, 2)

    # ---------------------------------------------------------- drawing: run
    def draw_world(self):
        s = self.screen
        env, car = self.env, self.env.car
        s.blit(self.skin(self.name), (0, 0))
        if self.show_trail and self.trail is not None:
            s.blit(self.trail, (0, 0))
        self.fx.fill((0, 0, 0, 0))
        if self.show_debug:
            pts = [(int(x), int(y)) for x, y in cs.CENTER[::4]]
            if len(pts) > 1:
                pygame.draw.lines(self.fx, (255, 255, 255, 90), False, pts, 1)
            tx, ty = env.target_point() if hasattr(env, "target_point") else (car.x, car.y)
            pygame.draw.circle(self.fx, (255, 255, 255, 200), (int(tx), int(ty)), 5, 2)
        if self.show_sensors:
            for off, d in zip(cs.SENSOR_ANGLES, car.readings):
                a = car.angle + off
                end = (car.x + math.cos(a) * d, car.y + math.sin(a) * d)
                col = lerp_color((235, 77, 75), (46, 204, 113), d / cs.SENSOR_RANGE)
                pygame.draw.line(self.fx, (*col, 150), (car.x, car.y), end, 1)
                if d < cs.SENSOR_RANGE:
                    pygame.draw.circle(self.fx, (*col, 230), (int(end[0]), int(end[1])), 2)
        braking = self.last_braking and not self.result
        draw_car_fx(self.fx, car, braking)
        s.blit(self.fx, (0, 0))
        draw_car_body(s, car, braking)
        for p in self.confetti:
            r = pygame.Rect(0, 0, 5, 9)
            r.center = (int(p[0]), int(p[1]))
            pygame.draw.rect(s, p[4], r)
        if self.result:
            self.draw_result()
        if self.paused and not self.result:
            draw_text(s, "PAUSED", 30, TEXT, (cs.W // 2, 36), bold=True, anchor="center")

    def draw_result(self):
        s = self.screen
        r = self.result
        card = pygame.Rect(0, 0, 470, 220)
        card.center = (cs.W // 2, cs.H // 2)
        translucent_rect(s, (10, 13, 20, 225), card, 18)
        title, col = {"finish": ("FINISHED!", GOOD), "crash": ("CRASHED", BAD),
                      "timeout": ("TIME OUT", AMBER)}[r["kind"]]
        draw_text(s, title, 46, col, (card.centerx, card.y + 46), bold=True, anchor="center")
        if r["kind"] == "finish":
            draw_text(s, fmt_ticks(r["ticks"]), 24, TEXT, (card.centerx, card.y + 98), anchor="center")
            draw_text(s, f"score {r['score']:.0f} / 1000", 18, MUTED, (card.centerx, card.y + 128), anchor="center")
            if r["new_best"]:
                draw_text(s, "NEW BEST!", 18, AMBER, (card.centerx, card.y + 154), bold=True, anchor="center")
        else:
            draw_text(s, f"reached {r['progress'] * 100:.0f}% of the track", 22, TEXT,
                      (card.centerx, card.y + 100), anchor="center")
            draw_text(s, f"after {fmt_ticks(r['ticks'])}", 16, MUTED, (card.centerx, card.y + 130), anchor="center")
        hint = "next track coming up..." if self.run_all else "R retry     N next     M menu"
        draw_text(s, hint, 15, MUTED, (card.centerx, card.bottom - 24), anchor="center")

    def draw_panel(self):
        s = self.screen
        env, car = self.env, self.env.car
        x0 = cs.W
        pygame.draw.rect(s, PANEL, (x0, 0, PANEL_W, WIN_H))
        pygame.draw.line(s, CARD_EDGE, (x0, 0), (x0, WIN_H), 1)
        cx = x0 + PANEL_W // 2
        draw_text(s, self.name, 18, ACCENT, (cx, 18), bold=True, anchor="midtop")
        tag = "seen in training" if self.name in TRAINED else "NEW track - never seen"
        draw_text(s, tag, 11, GOOD if self.name in TRAINED else AMBER, (cx, 42), anchor="midtop")
        # ticks / time
        draw_text(s, "TICKS", 11, MUTED, (x0 + 16, 66))
        draw_text(s, f"{env.steps}", 30, TEXT, (x0 + 16, 78), bold=True)
        draw_text(s, f"{env.steps / FPS:.1f} s", 14, MUTED, (x0 + PANEL_W - 16, 92), anchor="topright")
        best = self.best(self.name)
        draw_text(s, f"best: {best}" if best else "best: -", 12, AMBER if best else MUTED, (x0 + 16, 116))
        # gauge
        gcx, gcy, gr = cx, 196, 66
        frac = (car.speed - 0.0) / cs.MAX_SPEED
        steps = 40
        for i in range(steps):
            t0, t1 = i / steps, (i + 1) / steps
            a0, a1 = math.pi + t0 * math.pi, math.pi + t1 * math.pi
            col = speed_color(t0)
            if t0 > frac:
                col = lerp_color(col, PANEL, 0.82)
            quad = [(gcx + math.cos(a0) * gr, gcy + math.sin(a0) * gr),
                    (gcx + math.cos(a0) * (gr - 11), gcy + math.sin(a0) * (gr - 11)),
                    (gcx + math.cos(a1) * (gr - 11), gcy + math.sin(a1) * (gr - 11)),
                    (gcx + math.cos(a1) * gr, gcy + math.sin(a1) * gr)]
            aa_poly(s, col, quad)
        na = math.pi + max(0.0, min(1.0, frac)) * math.pi
        pygame.draw.line(s, TEXT, (gcx, gcy), (gcx + math.cos(na) * (gr - 16), gcy + math.sin(na) * (gr - 16)), 3)
        pygame.draw.circle(s, TEXT, (gcx, gcy), 5)
        draw_text(s, f"{car.speed:.1f}", 20, TEXT, (gcx, gcy + 10), bold=True, anchor="midtop")
        draw_text(s, "speed (px/tick)", 11, MUTED, (gcx, gcy + 34), anchor="midtop")
        # progress
        prog = 1.0 if env.finished else max(0.0, min(1.0, env.max_prog / cs.LAP_LENGTH))
        draw_text(s, "PROGRESS", 11, MUTED, (x0 + 16, 256))
        draw_text(s, f"{prog * 100:.0f}%", 11, TEXT, (x0 + PANEL_W - 16, 256), anchor="topright")
        bar = pygame.Rect(x0 + 16, 274, PANEL_W - 32, 10)
        rounded(s, (40, 46, 60), bar, 5)
        if prog > 0.01:
            rounded(s, ACCENT, (bar.x, bar.y, int(bar.w * prog), bar.h), 5)
        # what the AI is doing
        steer = self.action // 3 - 1
        thr = self.action % 3 - 1
        draw_text(s, "AI CONTROLS", 11, MUTED, (x0 + 16, 300))
        lane = pygame.Rect(x0 + 16, 322, PANEL_W - 32, 12)
        rounded(s, (40, 46, 60), lane, 6)
        pygame.draw.circle(s, AMBER, (lane.centerx + steer * (lane.w // 2 - 10), lane.centery), 8)
        draw_text(s, "L", 10, MUTED, (lane.x + 2, lane.bottom + 2))
        draw_text(s, "R", 10, MUTED, (lane.right - 8, lane.bottom + 2))
        for i, (lab, val) in enumerate([("BRAKE", -1), ("COAST", 0), ("GAS", 1)]):
            bw = (PANEL_W - 32 - 8) // 3
            r = pygame.Rect(x0 + 16 + i * (bw + 4), 350, bw, 22)
            on = thr == val
            colr = {-1: BAD, 0: MUTED, 1: GOOD}[val]
            rounded(s, colr if on else (40, 46, 60), r, 6)
            draw_text(s, lab, 10, BG if on else MUTED, r.center, bold=True, anchor="center")
        # radar
        rcx, rcy, rr = cx, 440, 52
        draw_text(s, "WHAT THE AI SEES", 11, MUTED, (x0 + 16, 386))
        pygame.draw.circle(s, (40, 46, 60), (rcx, rcy + 8), rr, 1)
        pygame.draw.circle(s, (40, 46, 60), (rcx, rcy + 8), rr // 2, 1)
        pts = [(rcx, rcy + 8)]
        for off, d in zip(cs.SENSOR_ANGLES, car.readings):
            a = -math.pi / 2 + off
            ln = d / cs.SENSOR_RANGE * rr
            pts.append((rcx + math.cos(a) * ln, rcy + 8 + math.sin(a) * ln))
        tmp = pygame.Surface((PANEL_W, 140), pygame.SRCALPHA)
        shifted = [(px - x0, py - 370) for px, py in pts]
        pygame.draw.polygon(tmp, (*ACCENT, 70), shifted)
        pygame.draw.polygon(tmp, (*ACCENT, 230), shifted, 2)
        s.blit(tmp, (x0, 370))
        aa_poly(s, TEXT, [(rcx, rcy + 2), (rcx - 4, rcy + 14), (rcx + 4, rcy + 14)])
        # legend
        for i in range(PANEL_W - 32):
            pygame.draw.line(s, speed_color(i / (PANEL_W - 33)), (x0 + 16 + i, 512), (x0 + 16 + i, 518))
        draw_text(s, "slow", 10, MUTED, (x0 + 16, 522))
        draw_text(s, "fast", 10, MUTED, (x0 + PANEL_W - 16, 522), anchor="topright")
        draw_text(s, "trail colour = speed", 10, MUTED, (cx, 522), anchor="midtop")
        mult = SPEED_LEVELS[self.speed_index]
        draw_text(s, f"sim speed x{mult:g}   (+/-)", 11, TEXT, (cx, 546), anchor="midtop")
        draw_text(s, "SPACE pause  M menu  P photo", 10, MUTED, (cx, 566), anchor="midtop")
        draw_text(s, "S sensors  T trail  D debug", 10, MUTED, (cx, 580), anchor="midtop")

    def draw(self):
        if self.mode == "menu":
            self.draw_menu()
        elif self.mode == "place":
            self.draw_place()
        else:
            self.screen.fill(BG)
            self.draw_world()
            self.draw_panel()
        msg, frames = self.toast
        if frames > 0 and msg:
            r = draw_text(self.screen, msg, 15, TEXT, (WIN_W // 2, WIN_H - 40), anchor="center")
            box = r.inflate(28, 16)
            translucent_rect(self.screen, (10, 13, 20, 230), box, 12)
            draw_text(self.screen, msg, 15, TEXT, box.center, anchor="center")

    # ---------------------------------------------------------- main loop
    def loop(self):
        while self.running:
            for ev in pygame.event.get():
                self.handle(ev)
            self.update()
            self.draw()
            pygame.display.flip()
            self.clock.tick(FPS)
        pygame.quit()


def run(model):
    App(model).loop()


def main():
    if not os.path.exists(cs.BEST_PATH):
        print("best_brain.pt not found. Train first:  python car_sim.py")
        sys.exit(1)
    model = cs.QNet()
    model.load_state_dict(torch.load(cs.BEST_PATH))
    model.eval()
    run(model)


if __name__ == "__main__":
    main()