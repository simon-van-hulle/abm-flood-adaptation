from paths import *
import pandas as pd
from model.model import AdaptationModel
from model import utils
import matplotlib.pyplot as plt
import networkx as nx


# Initialize the Adaptation Model with 50 household agents.
model = AdaptationModel(number_of_households=50, flood_map_choice="harvey", network="watts_strogatz") # flood_map_choice can be "harvey", "100yr", or "500yr"

# Generate the initial plots at step 0.
# Plot the spatial distribution of agents. This is a function written in the model.py
model.plot_model_domain_with_agents()

# Plot the initial state of the social network.
fig, ax = plt.subplots(figsize=(7, 7))
utils.plot_network(ax, model)
utils.savefig(f"step{0:0>2d}-social-network.pdf", FIG_DIR, "demo")

# Run the model for 20 steps and generate plots every 5 steps.
for step in range(20):
    model.step()
plt.close("all")
