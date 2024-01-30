# -*- coding: utf-8 -*-
"""
@author: thoridwagenblast

Functions that are used in the model_file.py and agent.py for the running of the Flood Adaptation Model.
Functions get called by the Model and Agent class.
"""
import random
import numpy as np
import math
from shapely import contains_xy, Point
from shapely import prepare
import geopandas as gpd

# Directories
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(CURRENT_DIR, os.pardir)


def set_initial_values(input_data, parameter, seed):
    """
    Function to set the values based on the distribution shown in the input data for each parameter.
    The input data contains which percentage of households has a certain initial value.

    :param DataFrame input_data: the dataframe containing the distribution of paramters
    :param str parameter: parameter name that is to be set
    :param int seed: agent's seed
    :return any: the value that is set for a certain agent for the specified parameter
    """
    parameter_set = 0
    parameter_data = input_data.loc[
        (input_data.parameter == parameter)
    ]  # get the distribution of values for the specified parameter
    parameter_data = parameter_data.reset_index()
    random.seed(seed)
    random_parameter = random.randint(0, 100)
    for i in range(len(parameter_data)):
        if i == 0:
            if random_parameter < parameter_data["value_for_input"][i]:
                parameter_set = parameter_data["value"][i]
                break
        else:
            if (random_parameter >= parameter_data["value_for_input"][i - 1]) and (
                random_parameter <= parameter_data["value_for_input"][i]
            ):
                parameter_set = parameter_data["value"][i]
                break
            else:
                continue
    return parameter_set


def get_flood_map_data(flood_map: str) -> tuple[float, float, float, float]:
    """
    Get the characteristics of the flood map data.

    :param str flood_map: flood map in tif format
    :return tuple[float, float, float, float]: characteristics of the tif-file
    """
    band = flood_map.read(1)
    bound_l = flood_map.bounds.left
    bound_r = flood_map.bounds.right
    bound_t = flood_map.bounds.top
    bound_b = flood_map.bounds.bottom
    return band, bound_l, bound_r, bound_t, bound_b


shapefile_path = os.path.join(BASE_DIR, r"input_data/model_domain/houston_model/houston_model.shp")
floodplain_path = os.path.join(BASE_DIR, r"input_data/floodplain/floodplain_area.shp")

# Model area setup
map_domain_gdf = gpd.GeoDataFrame.from_file(shapefile_path)
map_domain_gdf = map_domain_gdf.to_crs(epsg=26915)
map_domain_geoseries = map_domain_gdf["geometry"]
map_minx, map_miny, map_maxx, map_maxy = map_domain_geoseries.total_bounds
map_domain_polygon = map_domain_geoseries[0]  # The geoseries contains only one polygon
prepare(map_domain_polygon)

# Floodplain setup
floodplain_gdf = gpd.GeoDataFrame.from_file(floodplain_path)
floodplain_gdf = floodplain_gdf.to_crs(epsg=26915)
floodplain_geoseries = floodplain_gdf["geometry"]
floodplain_multipolygon = floodplain_geoseries[0]  # The geoseries contains only one multipolygon
prepare(floodplain_multipolygon)


def generate_random_location_within_map_domain() -> tuple[float, float]:
    """
    Generate random location coordinates within the map domain polygon.

    :return tuple[float, float]: lists of location coordinates, longitude and latitude
    """
    while True:
        # generate random location coordinates within square area of map domain
        x = random.uniform(map_minx, map_maxx)
        y = random.uniform(map_miny, map_maxy)
        # check if the point is within the polygon, if so, return the coordinates
        if contains_xy(map_domain_polygon, x, y):
            return x, y


def get_flood_depth(corresponding_map: str, location: Point, band: np.ndarray) -> float:
    """
    To get the flood depth of a specific location within the model domain.
    Households are placed randomly on the map, so the distribution does not follow reality.

    :param str corresponding_map: flood map used
    :param Point location: household location (a Shapely Point) on the map
    :param np.ndarray band: band from the flood map
    :return float: Flood Depth
    """
    row, col = corresponding_map.index(location.x, location.y)
    depth = band[row - 1, col - 1]
    return depth


"""
To generater the position on flood map for a household.
Households are placed randomly on the map, so the distribution does not follow reality.

Parameters
----------
bound_l, bound_r, bound_t, bound_b, img: characteristics of the flood map data (.tif file)
seed: seed to generate the location on the map

Returns
-------
x, y: location on the map
row, col: location within the tif-file
"""


def get_position_flood(
    bound_l: float, bound_r: float, bound_t: float, bound_b: float, img: str, seed: int
) -> tuple[float, float, int, int]:
    """
    To generater the position on flood map for a household.
    Households are placed randomly on the map, so the distribution does not follow reality.

    :param float bound_l: characteristics of the flood map data
    :param float bound_r: characteristics of the flood map data
    :param float bound_t: characteristics of the flood map data
    :param float bound_b: characteristics of the flood map data
    :param str img: characteristics of the flood map .tif file
    :param int seed: seed to generate the location on the map
    :return tuple[float, float, int, int]: location on the map and location within the tif-file
    """
    random.seed(seed)
    x = random.randint(round(bound_l, 0), round(bound_r, 0))
    y = random.randint(round(bound_b, 0), round(bound_t, 0))
    row, col = img.index(x, y)
    return x, y, row, col


def calculate_basic_flood_damage(flood_depth: float) -> float:
    """
    To get flood damage based on flood depth of household
    from de Moer, Huizinga (2017) with logarithmic regression over it.
    If flood depth > 6m, damage = 1.

    :param float flood_depth: flood depth as given by location within model domain
    :return float: Estimated damage between 0 and 1
    """
    if flood_depth >= 6:
        flood_damage = 1
    elif flood_depth < 0.025:
        flood_damage = 0
    else:
        # see flood_damage.xlsx for function generation
        flood_damage = 0.1746 * math.log(flood_depth) + 0.6483
    return flood_damage


def sigmoid(x: float) -> float:
    """
    Calculation of the sigmoid function

    :param float x: input variable
    :return float: sigmoid(x)
    """     
    return 1 / (1 + np.exp(-x))


def sigminv(x: float) -> float:
    """
    Inverse sigmoid function

    :param float x: sigmoid function of the unknown variable
    :return float: solution inverse sigmoid function
    """
    return np.log(x / (1 - x))


def lognormal_cdf(x: float, mu: float, sigma: float) -> float:
    """
    Cumulative density function of the lognormal distribution. Integral of the lognormal probability density function 
    from minus infinity to x.

    :param float x: Upper bound of the pdf integral
    :param float mu: mean of the lognormal distribution
    :param float sigma: standard deviation of the lognormal distribution
    :return float: cumulative density function P(X <= x)
    """
    if x == 0:
        x = np.nextafter(x, 1)
    return 0.5 + 0.5 * math.erf((np.log(x) - mu) / (sigma * np.sqrt(2)))