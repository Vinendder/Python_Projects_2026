import bisect
import math
import os
import random
import json
from collections import deque

import numpy as np
import pygame
import torch
import torch.nn as nn

pygame.init()
W, H = 800, 600
screen = pygame.display.set_mode((W, H))
clock = pygame.time.Clock()
BASE = os.path.dirname(os.path.abspath(__file__))
BEST_PATH = os.path.join(BASE, "best_brain.pt")

# car
CAR_LENGTH = 17
CAR_WIDTH = 10
MIN_SPEED = 1.0
MAX_SPEED = 6.0
START_SPEED = 2.0
ACCEL = 0.15
TURN_MAX = 0.09
TURN_LOSS = 0.6

# sensors
SENSOR_ANGLES = [-1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2]
SENSOR_RANGE = 200

# ------------------------------------------------------------------
# TRACKS. No checkpoints needed anymore! For each track you give:
#   image : the 800x600 painted map (dark pixels = walls)
#   start : (x, y) where the car spawns, in the MIDDLE of the road
#   angle : direction the car faces in degrees (0 = right, 90 = down,
#           180 = left, -90 = up)      ... OR ...
#   aim   : (x, y) a point the car should face (your old first checkpoint)
# A track with no angle/aim (or no image file) is skipped.
# ------------------------------------------------------------------
TRACKS = {
    "track 1": {
        "image": "track.png",
        "start": (119, 327),
        "aim": None,  # <-- paste your OLD FIRST CHECKPOINT here, like (200, 150)
    },
    "track 2": {
        "image": "track2.png",
        "start": (150, 551),
        "angle": 0,
    },
}
GEN_DIR = os.path.join(BASE, "train_tracks")
if os.path.isdir(GEN_DIR):
    for fname in sorted(os.listdir(GEN_DIR)):
        if fname.endswith(".json"):
            stem = fname[:-5]
            with open(os.path.join(GEN_DIR, fname)) as f:
                info = json.load(f)
            TRACKS[stem] = {
                "image": os.path.join("train_tracks", stem + ".png"),
                "start": tuple(info["start"]),
                "angle": info["angle"],
            }
SWITCH_EVERY = 10  # episodes between track changes during training
LOOKAHEAD = 35.0   # the "compass" points at the road this many pixels ahead (50 cut corners too much)
CUT_GAP = 6        # finish line sits this far behind the car's rear at spawn
MAX_STEPS = 1500

# rewards
TIME_COST = 0.05
PROGRESS_K = 0.1
CRASH_BASE = 15
CRASH_SPEED = 3
FINISH_BASE = 20
FINISH_SPEED = 80

# learning
GAMMA = 0.97
BATCH = 64
LR = 0.001
N_INPUTS = len(SENSOR_ANGLES) + 2
N_ACTIONS = 9
PATIENCE = 120
EVAL_EVERY = 25 # stop after this many evaluations with no new best


# ------------------------------------------------------------------
# Building a road from the picture
# ------------------------------------------------------------------
def build_road(name):
    t = TRACKS[name]
    surf = pygame.image.load(os.path.join(BASE, t["image"])).convert()
    if surf.get_size() != (W, H):
        raise SystemExit(f"{name}: image must be exactly {W}x{H}, got {surf.get_size()}")

    # 1 = wall, 0 = free. Indexed [x][y].
    arr = pygame.surfarray.array3d(surf).astype(int)
    block = (arr.sum(axis=2) < 150).astype(np.uint8)

    sx, sy = t["start"]
    if "angle" in t:
        ang = math.radians(t["angle"])
    else:
        ang = math.atan2(t["aim"][1] - sy, t["aim"][0] - sx)
    fx, fy = math.cos(ang), math.sin(ang)
    px, py = -fy, fx  # sideways direction

    if block[int(sx), int(sy)] != 0:
        raise SystemExit(f"{name}: the start point is on a wall, move it onto the road")

    # finish line: a short line across the road just behind the car.
    bx = sx - fx * (CAR_LENGTH / 2 + CUT_GAP)
    by = sy - fy * (CAR_LENGTH / 2 + CUT_GAP)
    ends = []
    for sign in (1, -1):
        d = 0.0
        while d < 150:
            cx = bx + px * d * sign
            cy = by + py * d * sign
            ix, iy = int(round(cx)), int(round(cy))
            if not (0 <= ix < W and 0 <= iy < H) or block[ix, iy] == 1:
                break
            for o in (-1.0, 0.0, 1.0):
                jx, jy = int(round(cx + fx * o)), int(round(cy + fy * o))
                if 0 <= jx < W and 0 <= jy < H and block[jx, jy] == 0:
                    block[jx, jy] = 2
            d += 0.5
        ends.append((bx + px * d * sign, by + py * d * sign))

    # flood fill from the start, layer by layer. Layer number = how far
    # along the road a pixel is. The finish line stops it going backwards.
    seen = (block != 0).tolist()
    dist = np.full((W, H), -1, dtype=np.int32)
    x0, y0 = int(sx), int(sy)
    seen[x0][y0] = True
    frontier = [(x0, y0)]
    centers = []
    level = 0
    reached = 0
    while frontier:
        xs = [p[0] for p in frontier]
        ys = [p[1] for p in frontier]
        dist[xs, ys] = level
        reached += len(frontier)
        centers.append((sum(xs) / len(xs), sum(ys) / len(ys)))
        nxt = []
        for x, y in frontier:
            if x + 1 < W and not seen[x + 1][y]:
                seen[x + 1][y] = True
                nxt.append((x + 1, y))
            if x - 1 >= 0 and not seen[x - 1][y]:
                seen[x - 1][y] = True
                nxt.append((x - 1, y))
            if y + 1 < H and not seen[x][y + 1]:
                seen[x][y + 1] = True
                nxt.append((x, y + 1))
            if y - 1 >= 0 and not seen[x][y - 1]:
                seen[x][y - 1] = True
                nxt.append((x, y - 1))
        frontier = nxt
        level += 1

    # the very last flood layers are just a sliver in a corner, so finish the
    # centerline at the middle of the finish line instead (the final compass target)
    centers.append(((ends[0][0] + ends[1][0]) / 2, (ends[0][1] + ends[1][1]) / 2))
    center = np.array(centers)
    steps = np.hypot(np.diff(center[:, 0]), np.diff(center[:, 1]))
    arcs = np.concatenate([[0.0], np.cumsum(steps)])

    avg_width = reached / max(1.0, arcs[-1])
    corners = [(0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1)]
    leak = any(dist[x, y] >= 0 for x, y in corners)
    print(f"{name}: road pixels {reached}, lap length {arcs[-1]:.0f}px, average road width {avg_width:.0f}px")
    if leak:
        print(f"  WARNING {name}: the road fill reached a corner of the picture, so there is "
              "probably a GAP in a wall line. Fill the gap in with paint.")

    return {
        "surf": surf,
        "block": block.tolist(),
        "dist": dist.tolist(),
        "center": center.tolist(),
        "arcs": arcs.tolist(),
        "start": (float(sx), float(sy)),
        "angle": ang,
        "cut": ends,
        "leak": leak,
    }


def usable(name):
    t = TRACKS[name]
    if not os.path.exists(os.path.join(BASE, t["image"])):
        print("skipping", name, "- image file not found")
        return False
    if "angle" not in t and not t.get("aim"):
        print("skipping", name, "- needs an 'angle' or an 'aim'")
        return False
    return True


NAMES = [n for n in TRACKS if usable(n)]
if not NAMES:
    raise SystemExit("No usable tracks. Check the TRACKS dictionary.")
ROADS = {n: build_road(n) for n in NAMES}
print("using tracks:", NAMES)

# These get filled in by set_track()
track = None
BLOCK = DIST = CENTER = ARCS = None
START_X = START_Y = START_ANGLE = 0
LAP_LENGTH = IDEAL_STEPS = 0
CUT_ENDS = None


def set_track(name):
    global track, BLOCK, DIST, CENTER, ARCS, START_X, START_Y, START_ANGLE
    global LAP_LENGTH, IDEAL_STEPS, MAX_STEPS, CUT_ENDS
    r = ROADS[name]
    track = r["surf"]
    BLOCK, DIST, CENTER, ARCS = r["block"], r["dist"], r["center"], r["arcs"]
    START_X, START_Y = r["start"]
    START_ANGLE = r["angle"]
    CUT_ENDS = r["cut"]
    LAP_LENGTH = ARCS[-1]
    IDEAL_STEPS = LAP_LENGTH / MAX_SPEED
    MAX_STEPS = max(1500, int(IDEAL_STEPS * 3))


set_track(NAMES[0])


def cell(x, y):
    """0 = free, 1 = wall, 2 = finish line."""
    if x < 0 or x >= W or y < 0 or y >= H:
        return 1
    return BLOCK[int(x)][int(y)]


class Car:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = START_SPEED
        self.readings = [SENSOR_RANGE] * len(SENSOR_ANGLES)

    def update(self, steer, throttle):
        self.speed += throttle * ACCEL
        self.speed = max(MIN_SPEED, min(MAX_SPEED, self.speed))
        grip = 1 - TURN_LOSS * (self.speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)
        self.angle += steer * TURN_MAX * grip
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

    def corners(self):
        fx, fy = math.cos(self.angle), math.sin(self.angle)
        rx, ry = -fy, fx
        hl, hw = CAR_LENGTH / 2, CAR_WIDTH / 2
        return [
            (self.x + fx * hl - rx * hw, self.y + fy * hl - ry * hw),
            (self.x + fx * hl + rx * hw, self.y + fy * hl + ry * hw),
            (self.x - fx * hl + rx * hw, self.y - fy * hl + ry * hw),
            (self.x - fx * hl - rx * hw, self.y - fy * hl - ry * hw),
        ]

    def hit(self):
        """Worst thing the car's outline touches: 0 nothing, 1 wall, 2 finish line."""
        pts = self.corners()
        worst = 0
        for i in range(4):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % 4]
            for step in range(9):
                t = step / 8
                c = cell(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t)
                if c > worst:
                    worst = c
        return worst

    def sense(self):
        readings = []
        for offset in SENSOR_ANGLES:
            a = self.angle + offset
            dist = 0
            while dist < SENSOR_RANGE:
                px = self.x + math.cos(a) * dist
                py = self.y + math.sin(a) * dist
                if cell(px, py) == 1:
                    break
                dist += 2
            readings.append(dist)
        self.readings = readings

    def draw(self, screen):
        pygame.draw.polygon(screen, (0, 200, 255), self.corners())
        nose_x = self.x + math.cos(self.angle) * (CAR_LENGTH / 2)
        nose_y = self.y + math.sin(self.angle) * (CAR_LENGTH / 2)
        pygame.draw.line(screen, (255, 255, 255), (self.x, self.y), (nose_x, nose_y), 3)
        for offset, dist in zip(SENSOR_ANGLES, self.readings):
            a = self.angle + offset
            end = (self.x + math.cos(a) * dist, self.y + math.sin(a) * dist)
            pygame.draw.line(screen, (255, 80, 80), (self.x, self.y), end, 1)


class Env:
    def __init__(self):
        self.reset()

    def reset(self):
        self.car = Car(START_X, START_Y, START_ANGLE)
        self.car.sense()
        self.steps = 0
        self.finished = False
        self.prog = 0.0      # how far along the road the car is (pixels)
        self.max_prog = 0.0  # furthest it ever got
        return self.get_state()

    def progress_here(self):
        x, y = int(self.car.x), int(self.car.y)
        if 0 <= x < W and 0 <= y < H:
            d = DIST[x][y]
            if d >= 0:
                return ARCS[d]
        return self.prog

    def target_point(self):
        idx = bisect.bisect_left(ARCS, self.prog + LOOKAHEAD)
        return CENTER[min(idx, len(CENTER) - 1)]

    def get_state(self):
        car = self.car
        cx, cy = self.target_point()
        target = math.atan2(cy - car.y, cx - car.x)
        diff = (target - car.angle + math.pi) % (2 * math.pi) - math.pi
        state = [r / SENSOR_RANGE for r in car.readings]
        state.append(car.speed / MAX_SPEED)
        state.append(diff / math.pi)
        return state

    def time_fraction(self):
        frac = (MAX_STEPS - self.steps) / max(1, MAX_STEPS - IDEAL_STEPS)
        return max(0, min(1, frac))

    def score(self):
        if self.finished:
            return 300 + 700 * self.time_fraction()
        return 300 * min(1, self.max_prog / LAP_LENGTH)

    def step(self, action):
        steer = action // 3 - 1
        throttle = action % 3 - 1
        self.car.update(steer, throttle)
        self.steps += 1
        code = self.car.hit()
        self.car.sense()

        reward = -TIME_COST
        new_prog = self.progress_here()
        reward += PROGRESS_K * (new_prog - self.prog)
        self.prog = new_prog
        self.max_prog = max(self.max_prog, new_prog)

        crashed = code == 1
        if code == 2:
            if self.prog > LAP_LENGTH * 0.5:
                self.finished = True
                reward += FINISH_BASE + FINISH_SPEED * self.time_fraction()
            else:
                crashed = True  # backed into the start line

        if crashed:
            reward -= CRASH_BASE + CRASH_SPEED * self.car.speed

        done = crashed or self.finished or self.steps >= MAX_STEPS
        return self.get_state(), reward, done


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_INPUTS, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, N_ACTIONS),
        )

    def forward(self, x):
        return self.net(x)


def pick_action(model, state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, N_ACTIONS - 1)
    with torch.no_grad():
        q = model(torch.tensor(state, dtype=torch.float32))
    return int(q.argmax())


def evaluate(model, current):
    """Drive every track with no randomness; return the average score."""
    total = 0
    for name in NAMES:
        set_track(name)
        env = Env()
        state = env.reset()
        done = False
        while not done:
            action = pick_action(model, state, 0.0)
            state, reward, done = env.step(action)
        total += env.score()
    set_track(current)
    return total / len(NAMES)


def train(model, episodes=5000):
    target = QNet()
    target.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    memory = deque(maxlen=50000)
    env = Env()
    epsilon = 1.0
    best = -1
    since_best = 0
    total_steps = 0
    name = NAMES[0]

    for ep in range(episodes):
        if ep % SWITCH_EVERY == 0:
            choices = [n for n in NAMES if n != name] or NAMES
            name = random.choice(choices)
            set_track(name)
            print("--- switching to", name, "---")
        pygame.event.pump()
        state = env.reset()
        done = False
        while not done:
            action = pick_action(model, state, epsilon)
            next_state, reward, done = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_steps += 1

            if len(memory) >= 2000 and total_steps % 4 == 0:
                batch = random.sample(memory, BATCH)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)

                q = model(states).gather(1, actions).squeeze(1)
                with torch.no_grad():
                    best_next = target(next_states).max(1).values
                    goal = rewards + GAMMA * best_next * (1 - dones)
                loss = nn.functional.mse_loss(q, goal)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(0.05, epsilon * 0.995)
        if ep % 5 == 0:
            target.load_state_dict(model.state_dict())
        print("ep", ep, name, "score", round(env.score()), "steps", env.steps, "eps", round(epsilon, 2))

        if ep % EVAL_EVERY == 0:
            score = evaluate(model, name)
            if score > best:
                best = score
                since_best = 0
                torch.save(model.state_dict(), BEST_PATH)
                print("new best (average over all tracks):", round(score))
            else:
                since_best += 1
            if since_best >= PATIENCE:
                print("no new best for a long time, stopping")
                break


def showcase(model):
    idx = 0
    set_track(NAMES[idx])
    env = Env()
    state = env.reset()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = pick_action(model, state, 0.0)
        state, reward, done = env.step(action)

        if done:
            print(NAMES[idx], "score:", round(env.score()), "| steps:", env.steps, "| finished:", env.finished)
            idx = (idx + 1) % len(NAMES)
            set_track(NAMES[idx])
            state = env.reset()

        screen.blit(track, (0, 0))
        pts = [(int(x), int(y)) for x, y in CENTER[::3]]
        if len(pts) > 1:
            pygame.draw.lines(screen, (0, 150, 0), False, pts, 1)
        pygame.draw.line(screen, (255, 200, 0), CUT_ENDS[0], CUT_ENDS[1], 3)
        tx, ty = env.target_point()
        pygame.draw.circle(screen, (0, 255, 0), (int(tx), int(ty)), 6)
        env.car.draw(screen)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    model = QNet()
    try:
        train(model)
    except KeyboardInterrupt:
        print("training stopped early")
    model.load_state_dict(torch.load(BEST_PATH))
    showcase(model)