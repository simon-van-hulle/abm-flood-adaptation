# Imports to ensure that the model is found
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.realpath("__file__"))
BASE_DIR = os.path.join(CURRENT_DIR, os.pardir)
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

sys.path.append(BASE_DIR)
sys.path.append(CURRENT_DIR)

# Imports for the model
from model.model import AdaptationModel, Wizard
from functools import partial

from ema_workbench import (
    ReplicatorModel,
    RealParameter,
    ArrayOutcome,
    perform_experiments,
    save_results,
    ema_logging,
    Policy,
)

## General Setup for the simulations
HOUSEHOLDS = 200
N_STEPS = 80
REPLICATIONS = 100
SCENARIOS = 20
FILENAME = f"parametric-h{HOUSEHOLDS}-t{N_STEPS}-r{REPLICATIONS}-s{SCENARIOS}"
print(f"\n\nIMPORTANT: \nThe results of this run will be saved in {OUTPUT_DIR}/{FILENAME}.tar.gz. ENSURE THIS IS POSSIBLE!\n\n")

## Define the model function for EMA workbench
def model_adaptation(
    avg_trustworthiness_government=None,
    max_initial_savings=None,
    avg_savings_per_step_vs_house=None,
    avg_risk_aversion=None,
    width_risk_aversion=None,
    output_dir="tmp",
    wizard=Wizard(),
):
    outputdir = os.path.join(OUTPUT_DIR, output_dir)

    # Set parameters
    wizard.max_initial_savings = max_initial_savings
    wizard.avg_std_trustworthiness_governnment[0] = avg_trustworthiness_government
    wizard.avg_std_savings_per_step_vs_house[0] = avg_savings_per_step_vs_house
    wizard.min_max_initial_risk_aversion = [
        min(avg_risk_aversion - width_risk_aversion / 2, 0.0),
        min(avg_risk_aversion + width_risk_aversion / 2, 1.0),
    ]

    # Initialise the model
    model = AdaptationModel(number_of_households=HOUSEHOLDS, wizard=wizard)

    #  Run the model
    for step in range(N_STEPS):
        model.step()

    model_data = model.datacollector.get_model_vars_dataframe()

    return_dict = {k: model_data[k].to_list() for k in model_data.keys()}
    return return_dict

## Main model run

if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)

    model = ReplicatorModel("AdaptationModel", function=model_adaptation)

    model.uncertainties = [
        RealParameter("max_initial_savings", 90, 500),
        RealParameter("avg_trustworthiness_government", -0.1, 0.4),
        RealParameter("avg_savings_per_step_vs_house", -0.1, 0.3),  #
        RealParameter("avg_risk_aversion", 0.25, 0.75),  #
        RealParameter("width_risk_aversion", 0.25, 1.0),  #
    ]

    # Define model outcomes
    model.outcomes = [
        ArrayOutcome("InformationAbundance"),
        ArrayOutcome("SocietalRisk"),
        ArrayOutcome("AverageRiskAversion"),
        ArrayOutcome("AverageEstimationFactor"),
        ArrayOutcome("TotalAdapted"),
    ]

    model.replications = REPLICATIONS

    policies = [
        Policy("NoPolicy", function=partial(model_adaptation, wizard=Wizard(government_adaptation_strategies=["a"]))),
        Policy(
            "Info", function=partial(model_adaptation, wizard=Wizard(government_adaptation_strategies=["information"]))
        ),
        Policy(
            "Subsidy", function=partial(model_adaptation, wizard=Wizard(government_adaptation_strategies=["subsidy"]))
        ),
        Policy(
            "Info + Subsidy",
            function=partial(
                model_adaptation, wizard=Wizard(government_adaptation_strategies=["information", "subsidy"])
            ),
        ),
        # Policy("Info + Subsidy + Dikes", function=partial(model_adaptation, wizard=Wizard(government_adaptation_strategies=["information", "subsidy", "dikes"]))),
    ]

    # Run experiments with the aforementioned parameters and outputs
    results = perform_experiments(models=model, scenarios=SCENARIOS, policies=policies)

    save_results(results, f"{OUTPUT_DIR}/{FILENAME}.tar.gz")
