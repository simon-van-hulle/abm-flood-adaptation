import sys

sys.path.insert(0, "/home/simon/Documents/Study/Delft/AE-MSc1/ABM/project/abm-flood-adaptation")


import pytest
from scipy.stats import binom
from numpy import sqrt
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


## Set-up for tests

MODEL = SimpleModel()
AGENTS = [SimpleAgent(i, MODEL, 10 * i) for i in range(3)]
AGENTS.append(IrrelevantAgent(100, MODEL))
K_FACTOR = 2
RISK_MODEL = FloodRiskModel(MODEL, AGENTS, [SimpleAgent], k_aversion_factor=K_FACTOR)

## Hard-coded checks

IR_1 = 0.002469694294162453  # 0.9878777176649811 / 100 / 4
IR_2 = 0.002497736792314788  # 0.9990947169259152 / 100 / 4
P_AVG = 0.6623241448636321  # (0 + 0.9878777176649811 + 0.9990947169259152) / 3
IR_AVG = 0.0016558103621590802  # P_AVG / 100 / 4
P = 6  # (3 + 9) / 2
BINOM_0 = IR_AVG ** (0) * (1 - IR_AVG) ** (3) * 1
BINOM_1 = IR_AVG ** (1) * (1 - IR_AVG) ** (2) * 3
BINOM_2 = IR_AVG ** (2) * (1 - IR_AVG) ** (1) * 3
BINOM_3 = IR_AVG ** (3) * (1 - IR_AVG) ** (0) * 1

EXP_N = 0 + 1 * BINOM_1 + 2 * BINOM_2 + 3 * BINOM_3
EXP_N2 = 0 + 1**2 * BINOM_1 + 2**2 * BINOM_2 + 3**2 * BINOM_3
SIGMA_N = sqrt(EXP_N2 - EXP_N**2)
TR = EXP_N + K_FACTOR * SIGMA_N

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
        assert abs(RISK_MODEL.f_N(0) - BINOM_0) <= 1e-12
        assert abs(RISK_MODEL.f_N(1) - BINOM_1) <= 1e-12
        assert abs(RISK_MODEL.f_N(2) - BINOM_2) <= 1e-12
        assert abs(RISK_MODEL.f_N(3) - BINOM_3) <= 1e-12

    def test_FN(self):
        expected_2 = BINOM_0 + BINOM_1 + BINOM_2
        expected_3 = BINOM_0 + BINOM_1 + BINOM_2 + BINOM_3
        assert abs(RISK_MODEL.F_N(2) - expected_2) <= 1e-15
        assert abs(RISK_MODEL.F_N(3) - expected_3) <= 1e-15

    def test_PN(self):
        expected_1 = BINOM_2 + BINOM_3
        expected_2 = BINOM_3
        expected_3 = 0
        assert abs(RISK_MODEL.P_N(1) - expected_1) <= 1e-15
        assert abs(RISK_MODEL.P_N(2) - expected_2) <= 1e-15
        assert abs(RISK_MODEL.P_N(3) - expected_3) <= 1e-15

    def test_SRI(self):
        T = 0.5
        area = 90
        expected = (P * IR_AVG * T) / area
        assert RISK_MODEL.SRI(area, T=T) == expected

    def test_RI(self):
        alpha = 2
        Cx = lambda x: 4 * x

        expected = 0 + 1**alpha * Cx(1) * BINOM_1 + 2**alpha * Cx(2) * BINOM_2 + 3**alpha * Cx(3) * BINOM_3
        assert abs(RISK_MODEL.RI(alpha, Cx) - expected) <= 1e-6

    def test_TR(self):
        assert abs(RISK_MODEL.E_N() - EXP_N) <= 1e-6
        assert abs(RISK_MODEL.sigma_N() - SIGMA_N) <= 1e-6
        assert abs(RISK_MODEL.total_risk() - TR) <= 1e-6

    def test_check_threshold(self):
        """The thresholds here are chosen such that they are barely above the expected value"""
        
        assert RISK_MODEL.check_threshold(threshold=EXP_N + 1, risk_method=RISK_MODEL.E_N)
        assert RISK_MODEL.check_threshold(threshold=TR + 1, risk_method=RISK_MODEL.total_risk)
        assert not RISK_MODEL.check_threshold(
            threshold=TR + 1, risk_method=RISK_MODEL.total_risk, comparator=lambda risk, threshold: risk > threshold
        )
        
        threshold_func = lambda val : binom.cdf(val, 3, IR_AVG) + 1e-3
        assert RISK_MODEL.check_threshold(threshold=threshold_func, risk_method=RISK_MODEL.F_N, values=[1, 2, 3], )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
