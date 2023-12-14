# Importing necessary libraries
import random
from mesa import Agent
from shapely.geometry import Point
from shapely import contains_xy

# Import functions from functions.py
from .functions import (
    generate_random_location_within_map_domain,
    get_flood_depth,
    calculate_basic_flood_damage,
    floodplain_multipolygon,
)

"""
Some assumptions:
* House values are a representation of the income in a household
* The order of actions within one step does not matter (save, interact, ...)
* One tick is 0.25 year
* Money is in 
"""


class Wizard:
    """
    Class to store all magic numbers. For traceability and easy access.
    """

    max_initial_savings = 10
    house_vs_savings = 10
    avg_std_savings_per_step_vs_house = [0.01, 0.01]
    avg_std_trustworthiness = [0.5, 0.1]
    min_max_damage_estimation_factor = [0.8, 1.2]
    
    # min_max_


class Households(Agent):
    """
    An agent representing a household in the model.
    Each household has a flood depth attribute which is randomly assigned for demonstration purposes.
    In a real scenario, this would be based on actual geographical data or more complex logic.
    """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        self.is_adapted = False  # Initial adaptation status set to False
        self.savings = random.uniform(0, Wizard.max_initial_savings)
        self.house_value = self.savings * Wizard.house_vs_savings

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
        self.flood_depth_theoretical = max(
            get_flood_depth(corresponding_map=model.flood_map, location=self.location, band=model.band_flood_img), 0
        )

        # Add an attribute for the actual flood depth. This is set to zero at the beginning of the simulation since there is not flood yet
        # and will update its value when there is a shock (i.e., actual flood). Shock happens at some point during the simulation
        self.flood_depth_actual = 0

        # Rationality of the agent
        self.flood_damage_estimated = self.flood_damage_theoretical * random.uniform(
            *Wizard.min_max_damage_estimation_factor
        )
        self.risk_aversion = random.random()
        self.rationality = random.random()

    @property
    def flood_damage_theoretical(self):
        "Factor between 0 and 1"
        return calculate_basic_flood_damage(flood_depth=self.flood_depth_theoretical)
    
    def flood_damage_estimated(self):
        return 

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
        self.savings += random.normalvariate(*Wizard.avg_std_savings_per_step_vs_house) * self.house_value
        pass

    def inform(self):
        # Look at government information
        # Level of trust for government

        # Update estimated damage
        # Update estimated likelihood
        
        pass

    def interact(self):
        # Talk with social network
        # Talk with neighbours
        # Level of trust in friend(s)

        # Check if others adapted

        # Update estimated damage
        # Update estimated likelihood
        # Update risk aversion,
        
        for friend in self.friends:
            
            trustworthiness = max( random.normalvariate(*Wizard.avg_std_trustworthiness), 0)
            self.risk_aversion += trustworthiness * (friend.risk_aversion - self.risk_aversion)            
        
        pass

    def do_adaptation(self):
        # Logic for adaptation based on estimated flood damage and a random chance.
        # These conditions are examples and should be refined for real-world applications.

        # Should be a combination of flood_damage estimated,  savings and threshold
        # Also do something with the actual flood damage.

        estimated_money_damage = self.flood_damage_theoretical * self.house_value

        if (
            estimated_money_damage > self.risk_aversion
            and self.savings > self.model.adaptation_cost
            and random.random() < self.rationality
        ):
            self.is_adapted = True  # Agent adapts to flooding

    def step(self):
        self.save()
        self.inform()
        self.interact()
        self.do_adaptation()


# Define the Government agent class
class Government(Agent):
    """
    A government agent that currently doesn't perform any actions.
    """

    # Ideas
    # - Do a poll

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        self.societal_risk = None

    def update_spending(self):
        # How much did we spend in the last step
        # Based on all the model parameters in the policy

        # Check how much is spent on subsidies

        return

    def evaluate_risk(self):
        # Magic with the RBB
        # Update societal risk
        return

    def poll():
        return

    def update_policy(self):
        # Update policies in the model based on societal risk

        # How much info do you provide?

        # update information_abundance

        # update subsidy_level

        # Build dykes (very expensive)'

        # Regulate, maybe later

        # Build dykes

        return

    def step(self):
        # The government agent doesn't perform any actions.

        pass


# More agent classes can be added here, e.g. for insurance agents.
