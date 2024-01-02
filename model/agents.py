# Importing necessary libraries
import random
from mesa import Agent
from shapely.geometry import Point
from shapely import contains_xy
from scipy.stats import binom

# Import functions from functions.py
from .functions import *


"""
Some assumptions:
* House values are a representation of the income in a household
* The order of actions within one step does not matter (save, interact, ...)
* One tick is 0.25 year
* Money is in 
"""







class Households(Agent):
    """
    An agent representing a household in the model.
    Each household has a flood depth attribute which is randomly assigned for demonstration purposes.
    In a real scenario, this would be based on actual geographical data or more complex logic.
    """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        self.wizard = model.wizard
        
        self.is_adapted = False  # Initial adaptation status set to False
        self.savings = random.uniform(0, self.wizard.max_initial_savings)
        self.house_value = self.savings * self.wizard.house_vs_savings

        # getting flood map values
        # Get a random location on the map
        loc_x, loc_y = generate_random_location_within_map_domain()
        self.location = Point(loc_x, loc_y)

        # Check whether the location is within floodplain
        self.in_floodplain = contains_xy(geom=floodplain_multipolygon, x=self.location.x, y=self.location.y)

        # Get the estimated flood depth at those coordinates.
        # the estimated flood depth is calculated based on the flood map (i.e., past data) so this is not the actual flood depth
        # Flood depth can be negative if the location is at a high elevation --> Made zero

        # TODO This might become dynamic if building is include

        # Add an attribute for the actual flood depth. This is set to zero at the beginning of the simulation since there is not flood yet
        # and will update its value when there is a shock (i.e., actual flood). Shock happens at some point during the simulation
        self.flood_depth_actual = 0

        # Rationality of the agent
        self.risk_aversion = random.uniform(*self.wizard.min_max_initial_risk_aversion)
        self.flood_damage_estimation_factor = random.uniform(*self.wizard.min_max_damage_estimation_factor)
        self.rationality = random.uniform(*self.wizard.min_max_rationality)
        self.trustworthiness_of_friends = [random.normalvariate(*self.wizard.avg_std_trustworthiness) for i in range(100)]
        self.trustworthiness_of_government = random.normalvariate(*self.wizard.avg_std_trustworthiness_governnment)

    @property
    def flood_depth_theoretical(self):
        if self.is_adapted:
            return 0

        return max(
            get_flood_depth(
                corresponding_map=self.model.flood_map, location=self.location, band=self.model.band_flood_img
            ),
            0,
        )

    @property
    def flood_damage_theoretical(self):
        "Factor between 0 and 1"

        return calculate_basic_flood_damage(flood_depth=self.flood_depth_theoretical)

    @property
    def flood_damage_estimated(self):
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
            self.risk_aversion = max(min(self.risk_aversion, (1 - self.wizard.min_risk_aversion)), self.wizard.min_risk_aversion)

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

        if (
            self.risk_aversion * estimated_money_damage > (1 - self.risk_aversion) * self.model.adaptation_cost
            and self.savings > self.model.adaptation_cost
            and random.random() < self.rationality
        ):
            self.is_adapted = True  # Agent adapts to flooding
            self.savings -= self.model.adaptation_cost  # Agent pays for adaptation
   
    def flood_occurs(self):
        
        self.flood_depth_actual = random.uniform(*self.wizard.min_max_actual_depth_factor) * self.flood_depth_theoretical
        self.savings -= self.flood_damage_actual * self.house_value
        self.risk_aversion += min(1-self.risk_aversion, self.flood_damage_actual*random.normalvariate(*self.wizard.avg_std_flood_influence_risk_aversion))

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
def probability_drowning(depth):
    return lognormal_cdf(depth, 0.5, 0.8)

# FN curve, expected value of the number of fatalities, risk integral, total risk 

class RiskModel:
    def __init__(self, model):
        self.model = model
        self.function_flood_fatality_per_depth = probability_drowning
    
    
    def get_individual_risk_per_year(self, household):
        IR = self.function_flood_fatality_per_depth(household.flood_depth_theoretical) * self.model.floods_per_year
        IR *= (not household.is_adapted)
        return IR
    
    def get_societal_risk_per_year(self):
        return sum([self.get_individual_risk_per_year(agent) for agent in self.model.get_households()]) / self.model.number_of_households
    
    
    def get_probability_more_than_k_fatalities(self, k):
        N = self.model.number_of_households
        individual_risks = [self.get_individual_risk_per_year(agent) for agent in self.model.get_households()]    
        avg_individual_risk = np.mean(individual_risks)
        self.get_individual_risk_per_year(self.model.schedule.agents[0])
        return 1 - binom.cdf(k, N, avg_individual_risk)
    

###########################################################################

# Define the Government agent class
class Government(Agent):
    """
    A government agent that currently doesn't perform any actions.
    """

    # Ideas
    # - Do a poll

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        self.risk_model = RiskModel(model)

    def update_spending(self):
        # How much did we spend in the last step
        # Based on all the model parameters in the policy

        # Check how much is spent on subsidies
        
        return

    def evaluate_risk(self):
        # Magic with the RBB
        # Update societal risk
        
        self.model.societal_risk = self.risk_model.get_societal_risk_per_year()
        
        
        return

    def poll():
        return

    def update_policy(self):
        # Update policies in the model based on societal risk

        # How much info do you provide?

        # update information_abundance
        
        y_abundance = 0.1
        max_societal_risk = 0.1
        self.model.information_abundance = y_abundance + self.model.societal_risk * (( 1 - y_abundance) / max_societal_risk)

    
        # self.model.information_abundance = 1 - self.model.societal_risk
        
        # update subsidy_level
        

        # Build dykes (very expensive)'

        # Regulate, maybe later

        # Build dykes
        
        
        # self.model.information_abundance = 
        

        return
    
    def flood_occurs(self):
        pass

    def step(self):
        # The government agent doesn't perform any actions.

        self.evaluate_risk()
        self.update_policy()
        self.update_spending()


# More agent classes can be added here, e.g. for insurance agents.
