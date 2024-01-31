# Importing necessary libraries
import random
from mesa import Agent
from shapely.geometry import Point
from shapely import contains_xy
from scipy.stats import binom
from scipy.integrate import quad
from typing import override

# Import functions from functions.py
from .functions import *


"""
Some assumptions:
* House values are a representation of the income in a household
* The order of actions within one step does not matter (save, interact, ...)
* Money is in 
"""


class Households(Agent):
    """
    An agent representing a household in the model.
    Each household has a flood depth attribute which is randomly assigned for demonstration purposes.
    In a real scenario, this would be based on actual geographical data or more complex logic.

    :param int unique_id: unique identifier for the agent
    :param AdaptationModel model: the model the agent belongs to
    :param bool is_adapted: whether the agent has adapted to flooding
    :param float savings: amount of money the agent has saved
    :param float house_value: value of the house - this is a representation of the total capital of the household
    :param float government_subsidy_money: amount of money the government has paid for adaptation
    :param float government_damage_money: amount of money the government has paid for damage
    :param Point location: location of the household on the map
    :param bool in_floodplain: whether the household is in the floodplain
    :param float flood_depth_actual: actual flood depth of the household
    :param float flood_depth_theoretical: theoretical flood depth of the household
    :param float flood_damage_theoretical: theoretical flood damage of the household
    :param float flood_damage_estimated: estimated flood damage of the household
    :param float flood_damage_actual: actual flood damage of the household
    :param list friends: list of friends of the household
    :param float risk_aversion: risk aversion of the household
    :param float flood_damage_estimation_factor: factor used to estimate flood damage
    :param float rationality: rationality of the household
    :param list trustworthiness_of_friends: list of trustworthiness of friends of the household
    :param float trustworthiness_of_government: trustworthiness of the government
    """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        self.wizard = model.wizard

        self.is_adapted = False  # Initial adaptation status set to False
        self.savings = random.uniform(0, self.wizard.max_initial_savings)
        self.house_value = self.savings * self.wizard.house_vs_savings
        self.government_subsidy_money = 0
        self.government_damage_money = 0

        # getting flood map values - Get a random location on the map
        loc_x, loc_y = generate_random_location_within_map_domain()
        self.location = Point(loc_x, loc_y)

        # Check whether the location is within floodplain
        self.in_floodplain = contains_xy(geom=floodplain_multipolygon, x=self.location.x, y=self.location.y)

        # Get the estimated flood depth at those coordinates.
        # the estimated flood depth is calculated based on the flood map (i.e., past data) so this is not the actual flood depth
        # Flood depth can be negative if the location is at a high elevation --> Made zero

        # Add an attribute for the actual flood depth. This is set to zero at the beginning of the simulation since there is not flood yet
        # and will update its value when there is a shock (i.e., actual flood). Shock happens at some point during the simulation
        self.flood_depth_actual = 0

        # Rationality of the agent
        self.risk_aversion = random.uniform(*self.wizard.min_max_initial_risk_aversion)
        self.flood_damage_estimation_factor = random.uniform(*self.wizard.min_max_damage_estimation_factor)
        self.rationality = random.uniform(*self.wizard.min_max_rationality)
        self.trustworthiness_of_friends = [
            random.normalvariate(*self.wizard.avg_std_trustworthiness) for i in range(100)
        ]
        self.trustworthiness_of_government = random.normalvariate(*self.wizard.avg_std_trustworthiness_governnment)

    @property
    def flood_depth_theoretical(self):
        """
        Get the theoretical flood depth according to the flood maps.

        :return float: maximum of estimated flood depth and zero
        """
        if self.is_adapted:
            return 0

        return max(
            get_flood_depth(
                corresponding_map=self.model.flood_map, location=self.location, band=self.model.band_flood_img
            ),
            0,
        )

    @property
    def flood_damage_theoretical(self) -> float:
        """
        Theoretical value adjusted with factor between 0 and 1

        :return float: basic flood damage
        """

        return calculate_basic_flood_damage(flood_depth=self.flood_depth_theoretical)

    @property
    def flood_damage_estimated(self) -> float:
        """
        Estimated damage is the theoretical flood map value adjusted with a factor between 0 and 1

        :return float: estimated flood damage
        """
        return self.flood_damage_theoretical * self.flood_damage_estimation_factor

    @property
    def flood_damage_actual(self) -> float:
        """
        Real flood damage with the actual flood depth

        :return float: actual flood damage
        """
        return calculate_basic_flood_damage(flood_depth=self.flood_depth_actual)

    @property
    def friends(self) -> list:
        """
        Gets a list of friends of the agent.

        :return list: al neighbours in the social graph
        """
        return self.model.grid.get_neighbors(self.pos, include_center=False)

    # Function to count friends who can be influencial.
    def count_friends(self, radius: int = 1) -> int:
        """
        Count the number of neighbors within a given radius (number of edges away). This is social relation and not spatial

        :param int radius: Number of steps away from the agent (in social connections), defaults to 1
        :return int: number of friends within the given radius
        """
        friends = self.model.grid.get_neighborhood(self.pos, include_center=False, radius=radius)
        return len(friends)

    def save(self) -> None:
        """
        Save money for adaptation, in accordance with saving behaviour and income
        """
        self.savings += random.normalvariate(*self.wizard.avg_std_savings_per_step_vs_house) * self.house_value

    def inform(self) -> None:
        """
        Look at government information and update estimated damage and likelihood (getting informed about the situation)
        """
        self.flood_damage_estimation_factor += (
            self.trustworthiness_of_government
            * self.model.information_abundance
            * (1 - self.flood_damage_estimation_factor)
        )
        self.flood_damage_estimation_factor = max(self.flood_damage_estimation_factor, 0)

    def interact(self) -> None:
        """
        Interact with friends and update risk aversion, depending on the trustworthiness of friends
        """
        for i, friend in enumerate(self.friends):
            trustworthiness = self.trustworthiness_of_friends[i]
            self.risk_aversion += trustworthiness * (friend.risk_aversion - self.risk_aversion)
            self.risk_aversion = max(
                min(self.risk_aversion, (1 - self.wizard.min_risk_aversion)), self.wizard.min_risk_aversion
            )

            # Weird ideas

            # trustworthiness =max(0, random.normalvariate(*self.wizard.avg_std_trustworthiness))
            # self.risk_aversion += trustworthiness * (friend.risk_aversion + self.risk_aversion) / 2

            # siginv_risk_averse = sigminv(friend.risk_aversion)  +  trustworthiness * (sigminv(friend.risk_aversion) - sigminv(self.risk_aversion))
            # self.risk_aversion = sigmoid(siginv_risk_averse)

            # trustworthiness = random.normalvariate(*self.wizard.avg_std_trustworthiness)
            # self.risk_aversion =  max(min(trustworthiness * (friend.risk_aversion - self.risk_aversion), 1), 0)

        pass

    def do_adaptation(self) -> None:
        """
        Logic for adaptation based on estimated flood damage and a random chance.
        For the present model, adaptation is a binary decision - to make it more accurate there should be multiple adaptations strategies
        """
        # Logic for adaptation based on estimated flood damage and a random chance.
        # These conditions are examples and should be refined for real-world applications.

        # Should be a combination of flood_damage estimated, savings and threshold
        # Also do something with the actual flood damage.

        estimated_money_damage = self.flood_damage_estimated * self.house_value

        subsidy_money = self.model.subsidy_policy(self)
        adaptation_cost = self.model.adaptation_cost - subsidy_money

        if (
            self.risk_aversion * estimated_money_damage > (1 - self.risk_aversion) * adaptation_cost
            and self.savings > adaptation_cost
            and random.random() < self.rationality
        ):
            self.is_adapted = True  # Agent adapts to flooding
            self.savings -= adaptation_cost  # Agent pays for adaptation
            self.government_subsidy_money += subsidy_money  # Government pays for adaptation

    def flood_occurs(self) -> None:
        """
        Flood occurs and damages the household. This function deals with implementing the results of that.
        """
        self.flood_depth_actual = (
            random.uniform(*self.wizard.min_max_actual_depth_factor) * self.flood_depth_theoretical
        )
        flood_cost = self.flood_damage_actual * self.house_value
        self.government_damage_money += flood_cost
        self.savings -= flood_cost
        self.risk_aversion += min(
            1 - self.risk_aversion,
            self.flood_damage_actual * random.normalvariate(*self.wizard.avg_std_flood_influence_risk_aversion),
        )

    def step(self) -> None:
        """
        Perform a single step in the model.
        """
        self.save()
        self.inform()
        self.interact()
        self.do_adaptation()


###########################################################################


class RiskModel:
    """
    A class that calculates the risk of flooding for the entire model.

    :param list[Agent] agents: list of agents in the risk model
    :param list[type] agent_type_filter: list of agent types to filter on, defaults to None
    :param int k_aversion_factor: scaling factor for the risk function, defaults to 1
    :param int alpha: scaling exponent for the risk integral function, defaults to 1
    :param callable Cx_function: scaling function for the risk integral function, defaults to lambda x: 1
    """

    def __init__(
        self,
        agents: Agent,
        agent_type_filter: list[type] = None,
        alpha: int = 1,
        k_aversion_factor: int = 1,
        Cx_function: callable = lambda x: 1,
    ):
        self.agent_type_filter = agent_type_filter
        self.agents = agents
        if agent_type_filter is not None:
            self.agents = [agent for agent in agents if type(agent) in agent_type_filter]
        self.N_agents = len(self.agents)

        self.alpha = alpha
        self.k_aversion_factor = k_aversion_factor
        self.Cx_function = Cx_function  # The scaling function defaults to 1

    def p_failure(self, agent: Agent):
        """
        Function that should be implemented by the inheriting risk model, to calculate the probability that the failure
        of interest occurs for a given agent. Be aware of the time scale considered here. Best to work per step in the
        model and scale the probability outside of the function.

        :param Agnet agent: The agent for which the probability of failure should be calculated
        :raises NotImplementedError: The inheriting risk model should implement this function
        """
        raise NotImplementedError("The risk model requires a function for the probability of failure.")

    def p_death_if_failure(self, agent: Agent):
        """
        Function to calculate the probability of death if a failure occurs for a given agent. Be aware of the time scale
        The inheriting risk model should implement this function

        :param Agent agent: The agent of interest
        :raises NotImplementedError: The inheriting risk model should implement this function
        """
        raise NotImplementedError("The risk model requires a function for the probability of death if a failure occurs")

    def IR(self, agent: Agent) -> float:
        r"""
        Individual risk for a given agent
        
        .. math:: \mathit{IR} = P_{d|f}P_f

        :param Agent agent: Agent of interest
        :return float: risk for the agent to die in a given step
        """
        return self.p_death_if_failure(agent) * self.p_failure(agent)

    def avg_IR(self) -> float:
        """
        Average individual risk for all agents

        :return float: Average risk for any agent to die in a given step
        """
        return np.mean([self.IR(agent) for agent in self.agents])

    def f_N(self, x: int) -> float:
        """
        Probability density function used to calculate the probability of x agents dying in a given step. This is an
        approximation using the average individual risk and a binomial distribution.

        :param int x: number of agents that die in a given step
        :return float: probability of x agents dying in a given step
        """
        p = self.avg_IR()
        return binom.pmf(x, self.N_agents, p)

    def F_N(self, x: int) -> float:
        """
        Cumulative probability that less than x agents die in a given step. This is an approximation using the average
        individual risk and a binomial distribution.
        
        

        :param int x: max number of agents to die
        :return float: cumulative probability that less than x agents die in a given step
        """
        return binom.cdf(x, self.N_agents, self.avg_IR())

    def P_N(self, x: int) -> float:
        """
        Probability that more than x agents die in a given step. This is an approximation using the average individual
        risk and a binomial distribution.

        :param int x: min number of agents to die
        :return float: Probability of more than x agents dying in a given step
        """
        return 1 - self.F_N(x)

    def AWR(self) -> float:
        """
        The average weighted risk. Area aspects are included in the IR function

        :return float: AWR
        """
        return sum([self.IR(agent) for agent in self.agents])

    def SRI(self, area: float, n: int = None, IR: float = None, T: float = 1) -> float:
        """
        Scaled Risk Integral in (person + person^2) / (acre * step).
        This is usally expressed per million year instead of step, so a conversion factor is required

        :param float area: total area of interest
        :param int n: number of people occupying the area, defaults to None
        :param float IR: average individual risk in the area during the time of interest, defaults to None
        :param float T: fraction of time the area is occupied by n people, defaults to 1
        :return float: Scaled Risk Integral in (person + person^2) / (acre * step)
        """
        n = n or self.N_agents
        IR = IR or self.avg_IR()
        population_factor = (n + n * n) / 2
        return (population_factor * T * IR) / area

    def FN_curve(self) -> None:
        raise NotImplementedError

    def RI(self, alpha: int = None, Cx_function: callable = None) -> float:
        r"""
        Risk Integral at the current time. This is the integral of the risk function over the number of agents.
        If alpha=1 and Cx_function=1, this is the same as the expected value of deaths.
        
        .. math:: RI = \int_0^N x^\alpha \cdot C(x) \cdot f_N(x) \cdot dx

        :param int alpha: scaling exponent for risk integral, using model attribute if None
        :param callable Cx_function: scaling function for risk integral, using model attribute if None
        :return float: RI
        """
        alpha = alpha or self.alpha
        Cx_function = Cx_function or self.Cx_function

        risk_function = lambda x: x**self.alpha * self.f_N(x) * self.Cx_function(x)
        result, _ = quad(risk_function, 0, self.N_agents)
        return result

    def E_N(self) -> float:
        """
        Expected number of deaths at the current time. This is the same as the risk integral if alpha=1 and Cx_function=1.

        :return float: Expected number of deaths
        """
        return self.RI(1, lambda x: 1)

    def sigma_N(self)-> float:
        r"""
        Standard deviation of the number of deaths at the current time.

        :return float: :math:`\sigma(N)`
        """
        E = self.E_N()
        E2 = self.RI(2, lambda x: 1)
        return np.sqrt(E2 - E**2)

    def total_risk(self) -> float:
        r"""
        Total Risk
        
        .. math:: \mathit{TR} = \mathbb{E}(N) + k \cdot \sigma(N)

        :return float: _description_
        """
        E = self.E_N()
        sigma = self.sigma_N()
        return E + sigma * self.k_aversion_factor

    def societal_risk_exceeds_threshold(self, method, threshold):
        if method == "ExpectedValue":
            return self.AWR() > threshold


# Specific  functions for the flood model


class FloodRiskModel(RiskModel):
    def __init__(self, model, agents, agent_type_filter=[Households], **kwargs):
        super().__init__(agents, agent_type_filter=agent_type_filter, **kwargs)
        self.model = model

    @override
    def p_failure(self, agent):
        return self.model.floods_per_year * self.model.years_per_step

    @override
    def p_death_if_failure(self, agent):
        return lognormal_cdf(agent.flood_depth_theoretical, 0.5, 0.8)


###########################################################################


# Define the Government agent class
class Government(Agent):
    """
    A government agent that is responsible for adaptation policies.

    :param int unique_id: unique identifier for the agent
    :param AdaptationModel model: the model the agent belongs to
    :param list households: list of households in the model
    :param float total_budget: total budget of the government
    :param float subsidy_budget: budget for subsidies
    :param float damage_budget: budget for damage
    :param float information_budget: budget for information
    :param RiskModel risk_model: risk model for the government
    :param float n_households: number of households in the model
    """

    # Ideas
    # - Do a poll

    def __init__(self, unique_id, model, households):
        super().__init__(unique_id, model)

        self.households = households
        self.n_households = len(self.households)
        self.risk_model = FloodRiskModel(model, households)
        self.total_budget = 0
        self.subsidy_budget = 0
        self.damage_budget = 0
        self.information_budget = 0

    def update_spending(self):
        # How much did we spend in the last step
        # Based on all the model parameters in the policy

        # Check how much is spent on subsidies
        self.information_budget += self.model.information_abundance * self.model.information_cost
        self.subsidy_budget = sum([household.government_subsidy_money for household in self.households])
        self.damage_budget = sum([household.government_damage_money for household in self.households])

        self.total_budget = self.subsidy_budget + self.subsidy_budget + self.damage_budget

        return

    def evaluate_risk(self):
        # Magic with the RBB
        # Update societal risk

        self.model.societal_risk = self.risk_model.AWR()

        return

    def poll():
        return

    def update_policy(self):
        # Update policies in the model based on societal risk

        # How much info do you provide?

        # update information_abundance
        # TODO:THIS IS MAGIC - FIX THE NUMBERS

        if "information" in self.model.government_adaptation_strategies:
            self.model.societal_risk = self.risk_model.AWR()
            baseline_information_abundance = 0.2
            max_societal_risk = 0.1 * self.n_households
            self.model.information_abundance = baseline_information_abundance + self.model.societal_risk * (
                (1 - baseline_information_abundance) / max_societal_risk
            )

        if "subsidy" in self.model.government_adaptation_strategies:
            if self.risk_model.societal_risk_exceeds_threshold("ExpectedValue", 0.01 * self.n_households):
                self.model.subsidy_policy = (
                    lambda household: self.model.adaptation_cost * 0.5
                    if household.savings < self.model.adaptation_cost
                    else 0
                )

            else:
                self.model.subsidy_policy = lambda household: 0

            # if self.risk_model.societal_risk_exceeds_threshold("ExpectedValue", 0.1 * self.n_households):
            #     self.model.subsidy_policy =  lambda household :  self.model.adaptation_cost * 1.0 if household.savings < self.model.adaptation_cost else 0

            # elif self.risk_model.societal_risk_exceeds_threshold("ExpectedValue", 0.05 * self.n_households):
            #     self.model.subsidy_policy = lambda household :  self.model.adaptation_cost * 0.5 if household.savings < self.model.adaptation_cost else 0

        if "dikes" in self.model.government_adaptation_strategies:
            pass

        return

    def flood_occurs(self):
        pass

    def step(self):
        # The government agent doesn't perform any actions.

        self.evaluate_risk()
        self.update_policy()
        self.update_spending()


# More agent classes can be added here, e.g. for insurance agents.
