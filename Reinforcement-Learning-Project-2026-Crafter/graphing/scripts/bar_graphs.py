import matplotlib.pyplot as plt
import re
import os

# ========== CONFIGURATION ==========
num = 2
NAME = f"DQN_{num}"
ACHIEVEMENT_TEXT_FILE = f"results/achievements/{NAME}.txt"  # Path to your text file
OUTPUT_FIG = f"graphing/graphs/bar/{NAME}_achievements.png"
SAVE_FIG = True


# ========== LOAD & PARSE ==========
def parse_achievements(text):
    """
    Parses achievement summary text into a dict of {achievement_name: value}.
    Example line: '  collect_sapling: 7.50'
    """
    achievements = {}
    for line in text.strip().splitlines():
        match = re.match(r"\s*([\w_]+):\s*([\d.]+)", line)
        if match:
            name = match.group(1)
            value = float(match.group(2))
            achievements[name] = value
    return achievements


# Load from text file
if not os.path.exists(ACHIEVEMENT_TEXT_FILE):
    raise FileNotFoundError(f"Missing file: {ACHIEVEMENT_TEXT_FILE}")

with open(ACHIEVEMENT_TEXT_FILE, "r") as f:
    text_data = f.read()

achievements = parse_achievements(text_data)

if not achievements:
    raise ValueError("No achievements parsed. Check your text formatting.")

# ========== SORT AND PLOT ==========
achievements = dict(sorted(achievements.items(), key=lambda x: x[1], reverse=True))

names = list(achievements.keys())
values = list(achievements.values())

plt.figure(figsize=(10, 6))
bars = plt.barh(names, values, color="skyblue", edgecolor="black")

# Highlight nonzero bars
for bar, val in zip(bars, values):
    if val > 0:
        bar.set_color("seagreen")

plt.xlabel("Average Count over 10 Episodes")
plt.title("Average Achievements")
plt.gca().invert_yaxis()  # highest at top
plt.grid(axis="x", alpha=0.3)

# Annotate values on bars
for i, v in enumerate(values):
    plt.text(v + 0.1, i, f"{v:.2f}", va="center", fontsize=9)

if SAVE_FIG:
    os.makedirs(os.path.dirname(OUTPUT_FIG), exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=300)
    print(f"Achievement bar graph saved as: {OUTPUT_FIG}")
