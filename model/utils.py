import inspect
import numpy as np
import os
import matplotlib.pyplot as plt
import networkx as nx


## Logging tools


def NAME(var: any) -> list[str]:
    """
    Utility to retrieve the name of a variable.

    :param any var: any variable
    :return list[str]: String representation of the variable name
    """
    callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def LOG(*strings, separator: str = " ") -> None:
    """
    Utility to print a log message.

    :param str separator: separator to use between the \*strings, defaults to " "
    """
    message = separator.join(["[LOG]:", *strings])
    print(message)


def LOGVAR(var: any, format: str = "10.5f", space: int = 20):
    """
    Utility to log a single variable.

    :param any var: any variable
    :param str format: Format to print the value of the variable, defaults to "10.5f"
    :param int space: space to leave for the variable, defaults to 20
    """
    name = NAME(var)[0]
    extra = "\n" if type(var) == np.ndarray else ""
    try:
        print(f"\t {name:{space}s} : {extra}{var:{format}}")
    except TypeError:
        print(f"\t {name:{space}s} : {extra}{var}")


LV = LOGVAR

## Plotting tools


def savefig(filename: str, *dirs, **kwargs) -> None:
    """
    Save a figure with the given filename and directories

    :param str filename: name of the file (include extension - default png)
    """
    thedir = "."
    for dir in dirs:
        thedir = os.path.join(thedir, dir)
        if not os.path.exists(thedir):
            os.makedirs(thedir)
    plt.tight_layout()
    plt.savefig(os.path.join(thedir, filename), **kwargs)
    LOG(f"Saved figure to {thedir}/{filename}")


def plot_network(ax, model) -> None:
    """
    Define a function to plot agents on the network.

    :param Axis ax: Matplotlib axis to plot on
    :param Model model: Mesa model with appropriate parameters and agents
    """
    # Calculate positions of nodes for the network plot.
    # The spring_layout function positions nodes using a force-directed algorithm,
    # which helps visualize the structure of the social network.
    pos = nx.spring_layout(model.G)

    # Clear the current axes.
    ax.clear()
    # Determine the color of each node (agent) based on their adaptation status.
    colors = ["blue" if agent.is_adapted else "red" for agent in model.get_households()]
    # Draw the network with node colors and labels.
    nx.draw(model.G, pos, node_color=colors, with_labels=True, ax=ax)
    # Set the title of the plot with the current step number.
    ax.set_title(f"Social Network State at Step {model.schedule.steps}", fontsize=12)
