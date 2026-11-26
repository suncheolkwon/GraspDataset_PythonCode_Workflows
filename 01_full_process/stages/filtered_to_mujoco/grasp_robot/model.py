from pathlib import Path

MODEL = Path(__file__).resolve().parents[1] / "robot_assets" / "robots" / "h1hand_pos.xml"
SIDES = ("left", "right")
DIGITS = {"Thumb": "th", "Index": "ff", "Middle": "mf", "Ring": "rf", "Pinky": "lf"}
