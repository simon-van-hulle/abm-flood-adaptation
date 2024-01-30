import sys
import os

CURRENT_DIR = os.path.dirname(os.path.realpath("__file__"))
# BASE_DIR = os.path.join(CURRENT_DIR, os.pardir)
BASE_DIR = CURRENT_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

sys.path.append(BASE_DIR)
sys.path.append(CURRENT_DIR)
