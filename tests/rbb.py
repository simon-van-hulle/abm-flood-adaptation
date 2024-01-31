"""
Simple risk model for the flood adaptation model. This code contains a FloodRiskModel class
specialising the RiskModel RBB 
A simple agent and model are then used to implement unit tests on the risk
measure calculations.
"""

import pytest
from mesa import Agent, Model
from typing import override, Callable
import numpy as np
from scipy.stats import binom
from math import erf


### START Reusable Building Block #############################################


class RiskModel:
    """
    A class that calculates the risk of flooding for the entire model.

    :param list[Agent] agents: list of agents in the risk model
    :param list[type] agent_type_filter: list of agent types to filter on, defaults to None
    :param int k_aversion_factor: scaling factor for the risk function, defaults to 1
    :param int alpha: scaling exponent for the risk integral function, defaults to 1
    :param Callable Cx_function: scaling function for the risk integral function, defaults to lambda x: 1
    """

    def __init__(
        self,
        agents: Agent,
        agent_type_filter: list[type] = None,
        alpha: int = 1,
        k_aversion_factor: int = 1,
        Cx_function: Callable = lambda x: 1,
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
        The aggregated weighted risk. Area aspects are included in the IR function

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

    def RI(self, alpha: int = None, Cx_function: Callable = None) -> float:
        r"""
        Risk Integral at the current time. This is the integral of the risk function over the number of agents.
        If alpha=1 and Cx_function=1, this is the same as the expected value of deaths.

        .. math:: RI = \int_0^N x^\alpha \cdot C(x) \cdot f_N(x) \cdot dx

        :param int alpha: scaling exponent for risk integral, using model attribute if None
        :param Callable Cx_function: scaling function for risk integral, using model attribute if None
        :return float: RI
        """
        alpha = alpha or self.alpha
        Cx_function = Cx_function or self.Cx_function

        risk_function = lambda x: x**alpha * self.f_N(x) * Cx_function(x)
        result = sum(risk_function(i) for i in range(self.N_agents))
        return result

    def E_N(self) -> float:
        """
        Expected number of deaths at the current time. This is the same as the risk integral if alpha=1 and Cx_function=1.

        :return float: Expected number of deaths
        """
        return self.RI(1, lambda x: 1)

    def sigma_N(self) -> float:
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

    def check_threshold(
        self,
        threshold: Callable | float,
        risk_method: Callable = None,
        values: list[float]=None,
        comparator: Callable=lambda risk, threshold: risk < threshold,
        **method_kwargs
    ) -> bool:
        """
        Versatile method for checking whether a given risk measure agrees with 
        a given threshold. The threshold can be a callable that takes the number of 
        fatalities or a fixed value. The risk measure must be a callable and defaults
        to total risk. The risk measure is evaluated at every value in the values and
        compared to the threshold using the comparator function. 

        :param Callable | float threshold: function to give threshold at input value, or constant threshold
        :param Callable risk_method: method to use to get the risk measure, self.total_risk if None
        :param list[float] values: Values to evaluate the risk measure at, defaults to None
        :param Callable comparator: Function to compare risk and threshold: return True if passing, defaults to risk < threshold
        :return bool: Whether the risk measure agrees with the threshold
        """
        risk_method = risk_method or self.total_risk
        vals = values if hasattr(values, "__len__") else list([values])
        threshold_func = threshold if callable(threshold) else lambda val: threshold

        check = True
        for val in vals:
            if val is not None:
                risk_measure = risk_method(val, **method_kwargs)
            else:
                risk_measure = risk_method(**method_kwargs)

            threshold_val = threshold_func(val)
            check *= comparator(risk_measure, threshold_val)

        return bool(check)


### END Reusable Building Block ###############################################


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
    return 0.5 + 0.5 * erf((np.log(x) - mu) / (sigma * np.sqrt(2)))


### START RBB Specific Implementation #########################################


class FloodRiskModel(RiskModel):
    def __init__(self, model, agents, agent_type_filter=[SimpleAgent], **kwargs):
        super().__init__(agents, agent_type_filter=agent_type_filter, **kwargs)
        self.model = model

    @override
    def p_failure(self, agent: SimpleAgent) -> float:
        """
        The probability of failure for the flood model is equal to the probability
        of flooding during a single step. This is simply the number of floods per year
        multiplied by the number of years per step.

        :param Houshold agent: The agent of interest is a household
        :return float: probability of failure
        """
        return self.model.floods_per_year * self.model.years_per_step

    @override
    def p_death_if_failure(self, agent: SimpleAgent) -> float:
        """
        The probability of death if a flood occurs is approximated with a
        lognormal distribution with a mean of 0.5 and a standard deviation of 0.8.
        These values are preliminary estimates that are NOT based on any data.
        Needs to be updated with real data.

        :param Households agent: Household agent of interest
        :return float: probability of death if a flood occurs
        """
        return lognormal_cdf(agent.flood_depth_theoretical, 0.5, 0.8)


### END RBB Specific Implementation ###########################################


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
SIGMA_N = np.sqrt(EXP_N2 - EXP_N**2)
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
