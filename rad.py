import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import numpy as np
from collections import deque

PORT = "/dev/cu.usbserial-A5069RR4"
BAUD = 9600
MAX_RANGE = 100
DETECT_THRESHOLD = 30
TRAIL_LENGTH = 12
LOG_LINES = 8  # how many past readings to show in the log

ser = serial.Serial(PORT, BAUD, timeout=0.05)
ser.reset_input_buffer()

forward_angle = 0

fig = plt.figure(figsize=(10, 7))
fig.patch.set_facecolor("black")
gs = gridspec.GridSpec(1, 4, figure=fig)

# --- Radar plot on the left ---
ax = fig.add_subplot(gs[0, 0:3], polar=True)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_ylim(0, MAX_RANGE)
ax.set_facecolor("black")
ax.tick_params(colors="lime")
ax.grid(color="lime", alpha=0.3)
ax.set_yticks([25, 50, 75, 100])
ax.set_yticklabels(["25cm", "50cm", "75cm", "100cm"], color="lime", fontsize=8)
ax.set_xticklabels([])

trail = deque(maxlen=TRAIL_LENGTH)
trail_dots, = ax.plot([], [], "o", color="lime", alpha=0.25, markersize=6, animated=True)
main_blip, = ax.plot([], [], "o", color="red", markersize=14, animated=True)

# --- Info panel on the right ---
ax_info = fig.add_subplot(gs[0, 3])
ax_info.set_facecolor("black")
ax_info.axis("off")

header_text = ax_info.text(
    0.05, 0.97, "SONAR STATUS", transform=ax_info.transAxes,
    ha="left", va="top", fontsize=13, color="lime", fontweight="bold", animated=True
)
distance_text = ax_info.text(
    0.05, 0.88, "", transform=ax_info.transAxes,
    ha="left", va="top", fontsize=12, color="white", animated=True
)
status_text = ax_info.text(
    0.05, 0.80, "", transform=ax_info.transAxes,
    ha="left", va="top", fontsize=12, fontweight="bold", animated=True
)
log_header = ax_info.text(
    0.05, 0.68, "LOG", transform=ax_info.transAxes,
    ha="left", va="top", fontsize=11, color="gray", animated=True
)

# Fixed set of log line artists, stacked top-to-bottom, newest at top
log_texts = []
log_start_y = 0.60
log_spacing = 0.055
for i in range(LOG_LINES):
    t = ax_info.text(
        0.05, log_start_y - i * log_spacing, "", transform=ax_info.transAxes,
        ha="left", va="top", fontsize=9.5, color="lime", alpha=1 - (i * 0.09),
        family="monospace", animated=True
    )
    log_texts.append(t)

log_entries = deque(maxlen=LOG_LINES)
last_distance = -1
reading_count = 0

def read_latest_distance():
    global last_distance
    latest = None
    while ser.in_waiting:
        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue
        try:
            latest = float(line)
        except ValueError:
            continue
    if latest is not None:
        last_distance = latest
    return last_distance

def update(frame):
    global reading_count
    distance = read_latest_distance()
    artists = [trail_dots, main_blip, distance_text, status_text] + log_texts

    if distance != -1 and 0 < distance <= MAX_RANGE:
        trail.append(distance)
        angles = [forward_angle] * len(trail)
        trail_dots.set_data(angles, list(trail))
        main_blip.set_data([forward_angle], [distance])

        detected = distance <= DETECT_THRESHOLD
        distance_text.set_text(f"Distance: {distance:.1f} cm")
        status_text.set_text("DETECTED" if detected else "Clear")
        status_text.set_color("red" if detected else "lime")

        reading_count += 1
        log_entries.appendleft(f"#{reading_count:04d}  {distance:5.1f} cm")
    else:
        trail.clear()
        trail_dots.set_data([], [])
        main_blip.set_data([], [])
        distance_text.set_text("Distance: -- cm")
        status_text.set_text("Out of range")
        status_text.set_color("gray")

    for i, t in enumerate(log_texts):
        t.set_text(log_entries[i] if i < len(log_entries) else "")

    return artists

ani = animation.FuncAnimation(
    fig, update, interval=50, blit=True, cache_frame_data=False
)

plt.tight_layout()
plt.show()