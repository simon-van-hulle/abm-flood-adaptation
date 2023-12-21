import inspect
import numpy as np
import os
import matplotlib.pyplot as plt
import networkx as nx


## Logging tools


def NAME(var):
    callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def LOG(*strings, separator=" "):
    message = separator.join(["[LOG]:", *strings])
    print(message)


def LOGVAR(var, format="10.5f", space=20):
    name = NAME(var)[0]
    extra = "\n" if type(var) == np.ndarray else ""
    try:
        print(f"\t {name:{space}s} : {extra}{var:{format}}")
    except TypeError:
        print(f"\t {name:{space}s} : {extra}{var}")


LV = LOGVAR

## Plotting tools


def savefig(filename, *dirs, **kwargs):
    """Save a figure with the given filename and directories"""
    thedir = "."
    for dir in dirs:
        thedir = os.path.join(thedir, dir)
        if not os.path.exists(thedir):
            os.makedirs(thedir)
    plt.tight_layout()
    plt.savefig(os.path.join(thedir, filename), **kwargs)
    LOG(f"Saved figure to {thedir}/{filename}")


# Define a function to plot agents on the network.
# This function takes a matplotlib axes object and the model as inputs.
def plot_network(ax, model):
    # Calculate positions of nodes for the network plot.
    # The spring_layout function positions nodes using a force-directed algorithm,
    # which helps visualize the structure of the social network.
    pos = nx.spring_layout(model.G)

    # Clear the current axes.
    ax.clear()
    # Determine the color of each node (agent) based on their adaptation status.
    colors = ["blue" if agent.is_adapted else "red" for agent in model.schedule.agents]
    # Draw the network with node colors and labels.
    nx.draw(model.G, pos, node_color=colors, with_labels=True, ax=ax)
    # Set the title of the plot with the current step number.
    ax.set_title(f"Social Network State at Step {model.schedule.steps}", fontsize=12)
