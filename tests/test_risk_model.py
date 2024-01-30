# import sys
import os
import pytest
from mesa import Agent, Model

from model.agents import RiskModel, FloodRiskModel


class SimpleModel(Model):
    def __init__(self, floods_per_year=1 / 100, years_per_step=0.25):
        super().__init__()
        self.floods_per_year = floods_per_year
        self.years_per_step = years_per_step


class SimpleAgent(Agent):
    def __init__(self, unique_id, model, flood_depth_theoretical):
        super().__init__(unique_id, model)
        self.flood_depth_theoretical = flood_depth_theoretical


class IrrelevantAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)


model = SimpleModel()
agents = [SimpleAgent(i, model, 10 * i) for i in range(3)]
agents.append(IrrelevantAgent(100, model))
risk_model = FloodRiskModel(model, agents, [SimpleAgent])

IR_1 = 0.002469694294162453  # 0.9878777176649811 / 100 / 4
IR_2 = 0.002497736792314788  # 0.9990947169259152 / 100 / 4
P_AVG = 0.6623241448636321  # (0 + 0.9878777176649811 + 0.9990947169259152) / 3
IR_AVG = 0.0016558103621590802 # P_AVG / 100 / 4

class TestRiskModel:
    def test_simple_setup(self):
        assert len(agents) == 4
        assert agents[0].flood_depth_theoretical == 0
        assert agents[1].flood_depth_theoretical == 10
        assert agents[2].flood_depth_theoretical == 20
        with pytest.raises(AttributeError):
            agents[3].flood_depth_theoretical

        assert model.floods_per_year == 0.01
        assert model.years_per_step == 0.25

    def test_filter_agents(self):
        assert len(risk_model.agents) == 3
        assert risk_model.N_agents == 3

    def test_IR(self):
        assert risk_model.IR(agents[0]) == 0
        assert risk_model.IR(agents[1]) == IR_1
        assert risk_model.IR(agents[2]) == IR_2
        with pytest.raises(AttributeError):
            risk_model.IR(agents[3])

    def test_AWR(self):
        assert risk_model.AWR() == IR_1 + IR_2

    def test_p_avg(self):
        assert risk_model.avg_IR() == IR_AVG 

    def test_fN(self):
        expected_2 = IR_AVG ** (2) * (1 - IR_AVG) ** (1) * 3
        assert risk_model.f_N(2) == expected_2
