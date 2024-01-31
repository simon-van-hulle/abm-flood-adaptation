import pytest
from mesa import Agent, Model
from model.agents import RiskModel, FloodRiskModel

## Making the necessary simple classes setup

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

## Setting some constants and hard-coded checks

MODEL = SimpleModel()
AGENTS = [SimpleAgent(i, MODEL, 10 * i) for i in range(3)]
AGENTS.append(IrrelevantAgent(100, MODEL))
RISK_MODEL = FloodRiskModel(MODEL, AGENTS, [SimpleAgent])

IR_1 = 0.002469694294162453  # 0.9878777176649811 / 100 / 4
IR_2 = 0.002497736792314788  # 0.9990947169259152 / 100 / 4
P_AVG = 0.6623241448636321  # (0 + 0.9878777176649811 + 0.9990947169259152) / 3
IR_AVG = 0.0016558103621590802 # P_AVG / 100 / 4
P = 6 # (3 + 9) / 2
BINOM_0 = IR_AVG ** (0) * (1 - IR_AVG) ** (3) * 1
BINOM_1 = IR_AVG ** (1) * (1 - IR_AVG) ** (2) * 3
BINOM_2 = IR_AVG ** (2) * (1 - IR_AVG) ** (1) * 3
BINOM_3 = IR_AVG ** (3) * (1 - IR_AVG) ** (0) * 1

## Tests

class TestRiskModel:
    def test_simple_setup(self):
        assert len(AGENTS) == 4
        assert AGENTS[0].flood_depth_theoretical == 0
        assert AGENTS[1].flood_depth_theoretical == 10
        assert AGENTS[2].flood_depth_theoretical == 20
        with pytest.raises(AttributeError):
            AGENTS[3].flood_depth_theoretical

        assert MODEL.floods_per_year == 0.01
        assert MODEL.years_per_step == 0.25

    def test_filter_agents(self):
        assert len(RISK_MODEL.agents) == 3
        assert RISK_MODEL.N_agents == 3

    def test_IR(self):
        assert RISK_MODEL.IR(AGENTS[0]) == 0
        assert RISK_MODEL.IR(AGENTS[1]) == IR_1
        assert RISK_MODEL.IR(AGENTS[2]) == IR_2
        with pytest.raises(AttributeError):
            RISK_MODEL.IR(AGENTS[3])

    def test_AWR(self):
        assert RISK_MODEL.AWR() == IR_1 + IR_2

    def test_p_avg(self):
        assert RISK_MODEL.avg_IR() == IR_AVG 

    def test_fN(self):
        assert RISK_MODEL.f_N(2) == BINOM_2
        assert RISK_MODEL.f_N(3) == BINOM_3

    def test_FN(self):
        expected_2 = BINOM_0 + BINOM_1 + BINOM_2
        expected_3 = BINOM_0 + BINOM_1 + BINOM_2 + BINOM_3
        assert RISK_MODEL.F_N(2) - expected_2 <= 1e-15
        assert RISK_MODEL.F_N(3) - expected_3 <= 1e-15
    
    def test_PN(self):
        expected_1 = BINOM_2 + BINOM_3
        expected_2 = BINOM_3
        expected_3 = 0
        assert RISK_MODEL.P_N(1) - expected_1 <= 1e-15
        assert RISK_MODEL.P_N(2) - expected_2 <= 1e-15
        assert RISK_MODEL.P_N(3) - expected_3 <= 1e-15
        
    def test_SRI(self):
        T = 0.5
        area = 90
        expected = (P * IR_AVG * T) / area  
        assert RISK_MODEL.SRI(area, T=T) == expected
    
