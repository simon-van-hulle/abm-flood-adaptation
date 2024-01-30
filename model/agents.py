# Importing necessary libraries
import random
from mesa import Agent
from shapely.geometry import Point
from shapely import contains_xy
from scipy.stats import binom
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
        _summary_

        :return float: _description_
        """
        return self.flood_damage_theoretical * self.flood_damage_estimation_factor

    @property
    def flood_damage_actual(self):
        "Factor between 0 and 1"
        return calculate_basic_flood_damage(flood_depth=self.flood_depth_actual)

    @property
    def friends(self):
        return self.model.grid.get_neighbors(self.pos, include_center=False)

    # Function to count friends who can be influencial.
    def count_friends(self, radius):
        """Count the number of neighbors within a given radius (number of edges away). This is social relation and not spatial"""
        friends = self.model.grid.get_neighborhood(self.pos, include_center=False, radius=radius)
        return len(friends)

    def save(self):
        # Add some savings based on the value of the house
        self.savings += random.normalvariate(*self.wizard.avg_std_savings_per_step_vs_house) * self.house_value

        pass

    def inform(self):
        # Look at government information
        # Level of trust for government

        # Update estimated damage
        # Update estimated likelihood

        self.flood_damage_estimation_factor += (
            self.trustworthiness_of_government
            * self.model.information_abundance
            * (1 - self.flood_damage_estimation_factor)
        )
        self.flood_damage_estimation_factor = max(self.flood_damage_estimation_factor, 0)

    def interact(self):
        # Talk with social network
        # Talk with neighbours
        # Level of trust in friend(s)

        # Check if others adapted

        # Update estimated damage
        # Update estimated likelihood
        # Update risk aversion,

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

    def do_adaptation(self):
        # Logic for adaptation based on estimated flood damage and a random chance.
        # These conditions are examples and should be refined for real-world applications.

        # Should be a combination of flood_damage estimated,  savings and threshold
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

    def flood_occurs(self):
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

    def step(self):
        self.save()
        self.inform()
        self.interact()
        self.do_adaptation()


###########################################################################

"""
Some notes:
    "The functions that do not use self can also be redefined outside the class for versatility towards better models"
"""

# Risk model


class RiskModel:
    """
    A class that calculates the risk of flooding for the entire model.
    * IR = Individual risk
    * SR = Societal risk
    * N = number of households
    * AWR = Average Weighted Risk
    * SRI = Scaled Risk Integral
    * RI = Collective Risk Integral
    """

    def __init__(self, agents, agent_type_filter=None, alpha=1, k_aversion_factor=1, Cx_function=lambda x: 1):
        self.agent_type_filter = agent_type_filter
        self.agents = agents
        if agent_type_filter is not None:
            self.agents = [agent for agent in agents if type(agent) in agent_type_filter]
        self.N_agents = len(self.agents)

        self.alpha = alpha
        self.k_aversion_factor = k_aversion_factor
        self.Cx_function = Cx_function  # The scaling function defaults to 1

    def p_failure(self, agent):
        raise NotImplementedError("The risk model requires a function for the probability of failure.")

    def p_death_if_failure(self, agent):
        raise NotImplementedError("The risk model requires a function for the probability of death if a failure occurs")

    def IR(self, agent):
        return self.p_death_if_failure(agent) * self.p_failure(agent)

    def avg_IR(self):
        return np.mean([self.IR(agent) for agent in self.agents])

    def f_N(self, x):
        p = self.avg_IR()
        return binom.pmf(x, self.N_agents, p)

    def F_N(self, x):
        return binom.cdf(x, self.N_agents, self.avg_IR())

    def P_N(self, x):
        return 1 - self.F_N(x)

    def AWR(self):
        "The average weighted risk. Area aspects are included in the IR function"
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

    def FN_curve(self):
        pass

    def RI(self):
        pass

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
