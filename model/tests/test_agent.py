# import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# BASE_DIR = os.path.join(CURRENT_DIR, os.pardir)
# OUTPUT_DIR = os.path.join(BASE_DIR, "output")
# FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

# sys.path.append(CURRENT_DIR)

# print(CURRENT_DIR)
# print(BASE_DIR)

from model.agents import Households

class TestAgent:
    def test_1(self):
        assert 1 == 1
