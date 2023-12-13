# Importing necessary libraries
import random
from mesa import Agent
from shapely.geometry import Point
from shapely import contains_xy

# Import functions from functions.py
from .functions import generate_random_location_within_map_domain, get_flood_depth, calculate_basic_flood_damage, floodplain_multipolygon


# Define the Households agent class
class Households(Agent):
    """
    An agent representing a household in the model.
    Each household has a flood depth attribute which is randomly assigned for demonstration purposes.
    In a real scenario, this would be based on actual geographical data or more complex logic.
    """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.is_adapted = False  # Initial adaptation status set to False
        self.savings = random.randint(0, 100)  # Initial savings set to a random value between 0 and 100
        self.house_value = self.savings*10
        

        # getting flood map values
        # Get a random location on the map
        loc_x, loc_y = generate_random_location_within_map_domain()
        self.location = Point(loc_x, loc_y)

        # Check whether the location is within floodplain
        self.in_floodplain = False
        
        if contains_xy(geom=floodplain_multipolygon, x=self.location.x, y=self.location.y):
            self.in_floodplain = True

        # Get the estimated flood depth at those coordinates. 
        # the estimated flood depth is calculated based on the flood map (i.e., past data) so this is not the actual flood depth
        # Flood depth can be negative if the location is at a high elevation
        
        # TODO This might become dynamic if building is include
        self.flood_depth_estimated = get_flood_depth(corresponding_map=model.flood_map, location=self.location, band=model.band_flood_img)
        
        # handle negative values of flood depth
        if self.flood_depth_estimated < 0:
            self.flood_depth_estimated = 0
        
        # calculate the estimated flood damage given the estimated flood depth. Flood damage is a factor between 0 and 1
        self.flood_damage_estimated = calculate_basic_flood_damage(flood_depth=self.flood_depth_estimated)

        # Add an attribute for the actual flood depth. This is set to zero at the beginning of the simulation since there is not flood yet
        # and will update its value when there is a shock (i.e., actual flood). Shock happens at some point during the simulation
        self.flood_depth_actual = 0
        
        #calculate the actual flood damage given the actual flood depth. Flood damage is a factor between 0 and 1
        self.flood_damage_actual = calculate_basic_flood_damage(flood_depth=self.flood_depth_actual)
        
        # Rationality of the agent
        self.risk_aversion =  random.random()
        self.rationality = random.random()
        
    def save():
        
        # Add some savings
        
        pass
    
    
    def inform(selft):
        
        # Look at government information
        # Level of trust for government
        
        # Update estimated damage
        # Update estimated likelihood
        # Update risk aversion, 
    
        
        pass
    
    def interact(self):
        
        # Talk with social network
        # Talk with neighbours
        # Level of trust in friend(s)
        
        # Check if others adapted
        
        # Update estimated damage
        # Update estimated likelihood
        # Update risk aversion, 
        pass
    
    
    
    def do_adaptation(self):
        # Logic for adaptation based on estimated flood damage and a random chance.
        # These conditions are examples and should be refined for real-world applications.
        
        # Should be a combination of flood_damage estimated,  savings and threshold
        
        estimated_money_damage = self.flood_damage_estimated*self.house_value
        
        if  estimated_money_damage > self.risk_aversion and self.savings>self.model.adaptation_cost and random.random() < self.rationality:
            self.is_adapted = True  # Agent adapts to flooding
            
        
    
    # Function to count friends who can be influencial.
    def count_friends(self, radius):
        """Count the number of neighbors within a given radius (number of edges away). This is social relation and not spatial"""
        friends = self.model.grid.get_neighborhood(self.pos, include_center=False, radius=radius)
        return len(friends)

    def step(self):
        
        # Assumption: Put this order in the report
        
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
