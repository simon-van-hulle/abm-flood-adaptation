from paths import *
from model.model import AdaptationModel, Wizard
from model import utils
from pandas import DataFrame
import numpy as np
import os
from functools import partial

from ema_workbench import (
    ReplicatorModel,
    RealParameter,
    BooleanParameter,
    IntegerParameter,
    Constant,
    ArrayOutcome,
    perform_experiments,
    save_results,
    ema_logging,
    Policy,
)

HOUSEHOLDS = 100
N_STEPS = 80

print(OUTPUT_DIR)


def model_adaptation(initial_adaptation_cost, max_initial_savings=10, output_dir="tmp"):
    outputdir = os.path.join(OUTPUT_DIR, output_dir)

    # Set parameters
    wizard = Wizard()
    wizard.max_initial_savings = max_initial_savings
    wizard.initial_adaptation_cost = initial_adaptation_cost

    # Initialise the model
    model = AdaptationModel(number_of_households=HOUSEHOLDS, wizard=wizard)

    #  Run the model
    for step in range(N_STEPS):
        model.step()

    model_data = model.datacollector.get_model_vars_dataframe()

    # os.mkdir(outputdir)
    # model_data.to_pickle(f"{outputdir}/model_data.pkl")

    return {"TotalAdapted": model_data["TotalAdaptedHouseholds"].to_list()}


if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)

    model = ReplicatorModel("AdaptationModel", function=model_adaptation)

    model.uncertainties = [
        IntegerParameter("initial_adaptation_cost", 10, 200),
    ]

    # # Define model parameters that will remain constant
    # model.constants = [Constant("initial_outbreak_size", 1), Constant("steps", 30)]

    # Define model outcomes
    model.outcomes = [
        ArrayOutcome("TotalAdapted"),
    ]

    model.replications = 2

    policies = [
        Policy("Poor", function=partial(model_adaptation, max_initial_savings=10)),
        Policy("Rich", function=partial(model_adaptation, max_initial_savings=100)),
        Policy("Richest", function=partial(model_adaptation, max_initial_savings=1000)),
    ]

    # Run experiments with the aforementioned parameters and outputs
    results = perform_experiments(models=model, scenarios=5, policies=policies)

    # Get the results
    experiments, outcomes = results

    save_results(results, f"{OUTPUT_DIR}/temporary.tar.gz")

    # DataFrame(experiments).to_pickle(f"{OUTPUT_DIR}/experiments.pkl")

    # for key in outcomes:
    #     np.save(os.path.join(OUTPUT_DIR, f"{key}.npy"), outcomes[key])
