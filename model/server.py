import mesa

# from wolf_sheep.agents import GrassPatch, Sheep, Wolf
# from wolf_sheep.model import WolfSheep

from .agents import Households
from .model import AdaptationModel


def household_portrayal(G):
    def node_color(agent):
        return "blue" if agent.is_adapted else "red"

    def edge_color(agent1, agent2):
        # if State.RESISTANT in (agent1.state, agent2.state):
        #     return "#000000"
        return "#e8e8e8"

    def edge_width(agent1, agent2):
        # if State.RESISTANT in (agent1.state, agent2.state):
        #     return 3
        return 2

    def get_agents(source, target):
        return G.nodes[source]["agent"][0], G.nodes[target]["agent"][0]

    portrayal = {}
    portrayal["nodes"] = [
        {
            "size": 6,
            "color": node_color(agents[0]),
            "tooltip": f"id: {agents[0].unique_id}<br>state: {agents[0].is_adapted}",
        }
        for (_, agents) in G.nodes.data("agent")
    ]

    portrayal["edges"] = [
        {
            "source": source,
            "target": target,
            "color": edge_color(*get_agents(source, target)),
            "width": edge_width(*get_agents(source, target)),
        }
        for (source, target) in G.edges
    ]

    return portrayal


network = mesa.visualization.NetworkModule(household_portrayal)
chart = mesa.visualization.ChartModule(
    [
        {"Label": "Infected", "Color": "#FF0000"},
        {"Label": "Susceptible", "Color": "#008000"},
        {"Label": "Resistant", "Color": "#808080"},
    ]
)


# def get_resistant_susceptible_ratio(model):
#     ratio = model.resistant_susceptible_ratio()
#     ratio_text = "&infin;" if ratio is math.inf else f"{ratio:.2f}"
#     infected_text = str(number_infected(model))

#     return "Resistant/Susceptible Ratio: {}<br>Infected Remaining: {}".format(
#         ratio_text, infected_text
#     )


model_params = {
    "number_of_households": mesa.visualization.Slider(
        "Number of Households",
        20,
        10,
        100,
        1,
        description="Choose how many households to include in the model",
    ),
    "flood_map_choice": mesa.visualization.Choice(
        "Flood Map Choice", "harvey", ["harvey", "100yr", "500yr"], description="Avg Node Degree"
    ),
    "network": mesa.visualization.Choice(
        "Network Type",
        "watts_strogatz",
        ["erdos_renyi", "barabasi_albert", "watts_strogatz", "no_network"],
        description="Network Type",
    ),
    "probability_of_network_connection": mesa.visualization.Slider(
        "Virus Spread Chance",
        0.4,
        0.0,
        1.0,
        0.05,
        description="Likeliness of edge being created between two nodes",
    ),
    "number_of_edges": mesa.visualization.Slider(
        "Barabasi Albert Edges",
        3,
        1,
        10,
        1,
        description="Number of edges in BA network",
    ),
    "number_of_nearest_neighbours": mesa.visualization.Slider(
        "WS Number of nearest neighbours",
        5,
        0,
        10,
        1,
        description="Number of nearest neighbours for WS social network",
    ),
    # "gain_resistance_chance": mesa.visualization.Slider(
    #     "Gain Resistance Chance",
    #     0.5,
    #     0.0,
    #     1.0,
    #     0.1,
    #     description="Probability that a recovered agent will become "
    #     "resistant to this virus in the future",
    # ),
}

server = mesa.visualization.ModularServer(
    AdaptationModel,
    [
        network,
        # get_resistant_susceptible_ratio,
        chart,
    ],
    "Flood Adaptation Model",
    model_params,
)
server.port = 8522
