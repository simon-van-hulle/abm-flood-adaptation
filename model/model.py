# Importing necessary libraries
import networkx as nx
from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import geopandas as gpd
import rasterio as rs
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass
import os

# Import the agent class(es) from agents.py
from .agents import Households, Government

# Import functions from functions.py
from .functions import get_flood_map_data, calculate_basic_flood_damage
from .functions import map_domain_gdf, floodplain_gdf

# Directories
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(CURRENT_DIR, os.pardir)


@dataclass
class Wizard:
    """
    Class to store all magic numbers. For traceability and easy access.
    """

    def __init__(self, government_adaptation_strategies: list[str] = None) -> None:
        self.max_initial_savings = 100  # U
        self.house_vs_savings = 10
        self.avg_std_savings_per_step_vs_house = [0.01, 0.01]  # U
        self.avg_std_trustworthiness = [0.1, 0.2]
        self.avg_std_trustworthiness_governnment = [0.2, 0.1]  # U
        self.min_max_damage_estimation_factor = [0, 1]
        self.min_max_rationality = [0.4, 1.0]
        self.min_max_initial_risk_aversion = [0.0, 1.0]  # U
        self.min_risk_aversion = 0.03
        self.min_max_actual_depth_factor = [0.5, 1.2]
        self.avg_std_flood_influence_risk_aversion = [0.5, 0.1]
        self.initial_adaptation_cost = 100  #
        self.initial_information_abundance = 0.1
        self.information_cost = 100  #
        self.steps_with_flood = [15, 55]
        self.government_adaptation_strategies = government_adaptation_strategies or ["subsidy", "information", "dikes"]


# Define the AdaptationModel class
class AdaptationModel(Model):
    """
    The main model running the simulation. Extention of the :class:`mesa.Model` class. It sets up the network of household agents,
    simulates their behavior, and collects data. The network type can be adjusted based on study requirements.

    :param int seed: seed for the random number generator
    :param Wizard wizard: wizard object containing all magic numbers
    :param float adaptation_cost: cost of full adaptation to a flood
    :param float information_abundance: abundance of information available to the agents (between 0 and 1)
    :param list[int] steps_with_flood: list of steps at which a flood occurs
    :param list[str] government_adaptation_strategies: list of adaptation strategies the government can use
    :param float information_cost: cost of information - full information abundance per step
    :param float years_per_step: number of years per step
    :param callable subsidy_policy: function that returns the subsidy for a given household
    :param int max_initial_savings: maximum initial savings
    :param int house_vs_savings: ratio of houses to savings
    :param list[float] avg_std_savings_per_step_vs_house: average and standard deviation of savings per step per house
    :param list[float] avg_std_trustworthiness: average and standard deviation of trustworthiness
    :param list[float] avg_std_trustworthiness_governnment: average and standard deviation of government trustworthiness
    :param list[float] min_max_damage_estimation_factor: minimum and maximum damage estimation factor
    :param list[float] min_max_rationality: minimum and maximum rationality
    :param list[float] min_max_initial_risk_aversion: minimum and maximum initial risk aversion
    :param float min_risk_aversion: minimum risk aversion
    :param list[float] min_max_actual_depth_factor: minimum and maximum actual depth factor
    :param list[float] avg_std_flood_influence_risk_aversion: average and standard deviation of flood influence on risk aversion
    :param int number_of_households: number of household agents
    :param str flood_map_choice: flood map choice. Can currently be "harvey", "100yr", or "500yr".
    :param str network: network type. Can currently be "erdos_renyi", "barabasi_albert", "watts_strogatz", or "no_network"
    :param float probability_of_network_connection: likeliness of edge being created between two nodes
    :param int number_of_edges: number of edges for BA network
    :param int number_of_nearest_neighbours: number of nearest neighbours for WS social network
    :param Network grid: grid object for the network representation
    :param nx.Graph G: graph object for the network representation
    :param RandomActivation schedule: schedule for activating agents
    """

    def __init__(
        self,
        seed: int = None,
        number_of_households=25,
        flood_map_choice="100yr",
        network="watts_strogatz",
        probability_of_network_connection=0.4,
        number_of_edges=3,
        number_of_nearest_neighbours=5,
        years_per_step=0.25,
        wizard=Wizard(),
    ):
        super().__init__(seed=seed)

        # Additional parameters
        self.wizard = wizard
        self.adaptation_cost = self.wizard.initial_adaptation_cost
        self.information_abundance = self.wizard.initial_information_abundance
        self.steps_with_flood = self.wizard.steps_with_flood
        self.government_adaptation_strategies = self.wizard.government_adaptation_strategies
        self.information_cost = self.wizard.information_cost
        self.years_per_step = years_per_step

        # Default subsidy policy:
        self.subsidy_policy = lambda household: 0

        # defining the variables and setting the values
        self.number_of_households = number_of_households  # Total number of household agents
        self.seed = seed

        # network
        self.network = network  # Type of network to be created
        self.probability_of_network_connection = probability_of_network_connection
        self.number_of_edges = number_of_edges
        self.number_of_nearest_neighbours = number_of_nearest_neighbours

        # generating the graph according to the network used and the network parameters specified
        self.G = self.initialize_network()
        self.grid = NetworkGrid(self.G)

        # Initialize maps
        self.initialize_maps(flood_map_choice)

        # set schedule for agents
        self.schedule = RandomActivation(self)  # Schedule for activating agents

        # create households through initiating a household on each node of the network graph
        for i, node in enumerate(self.G.nodes()):
            household = Households(unique_id=i, model=self)
            self.schedule.add(household)
            self.grid.place_agent(agent=household, node_id=node)
        households = self.schedule.agents

        # Add government
        self.schedule.add(Government(unique_id=number_of_households + 1, model=self, households=households))

        # Data collection setup to collect data
        model_metrics = {
            "TotalAdapted": self.total_adapted_households,
            "AverageRiskAversion": self.average_risk_aversion,
            "AverageEstimationFactor": self.average_estimation_factor,
            "AdaptationCost": "adaptation_cost",
            "InformationAbundance": "information_abundance",
            "SocietalRisk": "societal_risk",
        }

        agent_metrics = {
            "FloodDepthTheoretical": "flood_depth_theoretical",
            "FloodDamageEstimated": "flood_damage_estimated",
            "FloodDepthActual": "flood_depth_actual",
            "FloodDamageActual": "flood_damage_actual",
            "FloodDamageEstimationFactor": "flood_damage_estimation_factor",
            "IsAdapted": "is_adapted",
            "RiskAversion": "risk_aversion",
            "Savings": "savings",
            "TotalBudget": "total_budget",
            "SubsidyBudget": "subsidy_budget",
            "DamageBudget": "damage_budget",
            "InformationBudget": "information_budget",
            # "FriendsCount": lambda a: a.count_friends(radius=1),
            "location": "location",
        }

        self.datacollector = DataCollector(model_reporters=model_metrics, agent_reporters=agent_metrics)

    def initialize_network(self) -> nx.Graph:
        """
        Initialize and return the social network graph based on the provided network type using pattern matching.

        :raises ValueError: If provided network type is not implemented
        :return nx.Graph: Graph object for the network representation
        """
        if self.network == "erdos_renyi":
            return nx.erdos_renyi_graph(
                n=self.number_of_households,
                p=self.number_of_nearest_neighbours / self.number_of_households,
                seed=self.seed,
            )
        elif self.network == "barabasi_albert":
            return nx.barabasi_albert_graph(n=self.number_of_households, m=self.number_of_edges, seed=self.seed)
        elif self.network == "watts_strogatz":
            return nx.watts_strogatz_graph(
                n=self.number_of_households,
                k=self.number_of_nearest_neighbours,
                p=self.probability_of_network_connection,
                seed=self.seed,
            )
        elif self.network == "no_network":
            G = nx.Graph()
            G.add_nodes_from(range(self.number_of_households))
            return G
        else:
            raise ValueError(
                f"Unknown network type: '{self.network}'. "
                f"Currently implemented network types are: "
                f"'erdos_renyi', 'barabasi_albert', 'watts_strogatz', and 'no_network'"
            )

    def initialize_maps(self, flood_map_choice:str) -> None:
        """
        Initialize and set up the flood map related data based on the provided flood map choice.

        :param str flood_map_choice: Chosen flood map (can be "harvey", "100yr", or "500yr")
        :raises ValueError: In case flood map is not implemented
        """
        # Define paths to flood maps
        flood_map_paths = {
            "harvey": os.path.join(BASE_DIR, r"input_data/floodmaps/Harvey_depth_meters.tif"),
            "100yr": os.path.join(BASE_DIR, r"input_data/floodmaps/100yr_storm_depth_meters.tif"),
            "500yr": os.path.join(
                BASE_DIR, r"input_data/floodmaps/500yr_storm_depth_meters.tif"
            ),  # Example path for 500yr flood map
        }

        # Throw a ValueError if the flood map choice is not in the dictionary
        if flood_map_choice not in flood_map_paths.keys():
            raise ValueError(
                f"Unknown flood map choice: '{flood_map_choice}'. "
                f"Currently implemented choices are: {list(flood_map_paths.keys())}"
            )

        # Choose the appropriate flood map based on the input choice
        flood_map_path = flood_map_paths[flood_map_choice]

        # Loading and setting up the flood map
        self.flood_map = rs.open(flood_map_path)
        self.band_flood_img, self.bound_left, self.bound_right, self.bound_top, self.bound_bottom = get_flood_map_data(
            self.flood_map
        )

        self.floods_per_year = {"harvey": 1 / 50, "100yr": 1 / 100, "500yr": 1 / 500}[flood_map_choice]

    def get_households(self)->list[Households]:
        """
        Get all agents of type Household

        :return list[Households]: all agents in the model of type Household
        """
        return [agent for agent in self.schedule.agents if isinstance(agent, Households)]

    def get_government(self) -> Government:
        """
        Get the government agent

        :return Government: the government agent
        """
        for agent in self.schedule.agents:
            if isinstance(agent, Government):
                return agent

    def total_adapted_households(self):
        """Return the total number of households that have adapted."""
        # BE CAREFUL THAT YOU MAY HAVE DIFFERENT AGENT TYPES SO YOU NEED TO FIRST CHECK IF THE AGENT IS ACTUALLY A HOUSEHOLD AGENT USING "ISINSTANCE"
        adapted_count = sum([1 for agent in self.get_households() if agent.is_adapted])
        return adapted_count

    def average_risk_aversion(self):
        """Return the average risk aversion of all households."""
        risk_aversion = sum([agent.risk_aversion for agent in self.schedule.agents if isinstance(agent, Households)])
        return risk_aversion / self.number_of_households

    def average_estimation_factor(self):
        """Return the average estimation factor of all households."""
        estimation_factor = sum(
            [agent.flood_damage_estimation_factor for agent in self.schedule.agents if isinstance(agent, Households)]
        )
        return estimation_factor / self.number_of_households

    def plot_model_domain_with_agents(self):
        fig, ax = plt.subplots()
        # Plot the model domain
        map_domain_gdf.plot(ax=ax, color="lightgrey")
        # Plot the floodplain
        floodplain_gdf.plot(ax=ax, color="lightblue", edgecolor="k", alpha=0.5)

        # Collect agent locations and statuses
        for agent in self.schedule.agents:
            if isinstance(agent, Government):
                continue
            color = "blue" if agent.is_adapted else "red"
            ax.scatter(
                agent.location.x,
                agent.location.y,
                color=color,
                s=10,
                label=color.capitalize() if not ax.collections else "",
            )
            ax.annotate(
                str(agent.unique_id),
                (agent.location.x, agent.location.y),
                textcoords="offset points",
                xytext=(0, 1),
                ha="center",
                fontsize=9,
            )
        # Create legend with unique entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title="Red: not adapted, Blue: adapted")

        # Customize plot with titles and labels
        plt.title(f"Model Domain with Agents at Step {self.schedule.steps}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        # plt.show()

    def step(self):
        """
        introducing a shock:
        at time step 5, there will be a global flooding.
        This will result in actual flood depth. Here, we assume it is a random number
        between 0.5 and 1.2 of the estimated flood depth. In your model, you can replace this
        with a more sound procedure (e.g., you can devide the floop map into zones and
        assume local flooding instead of global flooding). The actual flood depth can be
        estimated differently
        """

        if self.schedule.steps in self.steps_with_flood:
            for agent in self.schedule.agents:
                agent.flood_occurs()

        # Collect data and advance the model by one step
        self.datacollector.collect(self)
        self.schedule.step()
