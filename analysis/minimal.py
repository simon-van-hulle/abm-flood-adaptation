from paths import *
import pandas as pd
from model.model import AdaptationModel
from model import utils
import matplotlib.pyplot as plt
import networkx as nx

plt.style.use("./model/myBmh.mplstyle")


HOUSEHOLDS = 100
N_STEPS = 100

# Initialize the Adaptation Model with 50 household agents.
model = AdaptationModel(
    number_of_households=HOUSEHOLDS, flood_map_choice="harvey", network="watts_strogatz"
)  # flood_map_choice can be "harvey", "100yr", or "500yr"

# Generate the initial plots at step 0.
# Plot the spatial distribution of agents. This is a function written in the model.py
model.plot_model_domain_with_agents()

# Plot the initial state of the social network.
fig, ax = plt.subplots(figsize=(7, 7))
utils.plot_network(ax, model)
utils.savefig(f"step{0:0>2d}-social-network.pdf", FIG_DIR, "demo")



for step in range(N_STEPS):
    model.step()
plt.close("all")


agent_data = model.datacollector.get_agent_vars_dataframe()
model_data = model.datacollector.get_model_vars_dataframe()

# agent_data["color"] = agent_data["IsAdapted"].apply(lambda x: "k" if x else "red")

for key in model_data.keys():
    plt.figure()
    plt.plot(model_data.index.values, model_data[key])
    plt.xlabel("Step")
    plt.ylabel(key)
    utils.savefig(f"{key}.pdf", FIG_DIR, "demo", "model")


for key in agent_data.keys():
    try:
        plt.figure()
        plt.plot(agent_data.loc[:, key].to_numpy().reshape(N_STEPS, -1), ".-", linewidth=0.5, markersize=1)
        plt.xlabel("Step")
        plt.ylabel(key)
        utils.savefig(f"{key}.pdf", FIG_DIR, "demo", "agents")
    except TypeError:
        print(f"Could not plot {key} for agents.")

pass
