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


def model_adaptation(avg_trustworthiness_government=0.5 , initial_adaptation_cost=100, max_initial_savings=10, output_dir="tmp", wizard = Wizard()):
    outputdir = os.path.join(OUTPUT_DIR, output_dir)

    # Set parameters
    wizard.max_initial_savings = max_initial_savings
    wizard.initial_adaptation_cost = initial_adaptation_cost
    wizard.avg_std_trustworthiness_governnment[0] = avg_trustworthiness_government

    # Initialise the model
    model = AdaptationModel(number_of_households=HOUSEHOLDS, wizard=wizard)

    #  Run the model
    for step in range(N_STEPS):
        model.step()

    model_data = model.datacollector.get_model_vars_dataframe()

    return_dict  = {k: model_data[k].to_list() for k in model_data.keys()}
    return return_dict


if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)

    model = ReplicatorModel("AdaptationModel", function=model_adaptation)

    model.uncertainties = [
        RealParameter("max_initial_savings", 100, 101),
        RealParameter("avg_trustworthiness_government", 0.1, 0.11) # Don't make this too big
    ]

    # # Define model parameters that will remain constant
    # model.constants = [Constant("initial_outbreak_size", 1), Constant("steps", 30)]

    # Define model outcomes
    model.outcomes = [
        ArrayOutcome("InformationAbundance"),
        ArrayOutcome("SocietalRisk"),
        ArrayOutcome("AverageRiskAversion"),
        ArrayOutcome("AverageEstimationFactor"),
        ArrayOutcome("TotalAdapted"),
    ]

    model.replications = 2

    policies = [
        Policy("NoPolicy", function=partial(model_adaptation, wizard=Wizard(government_adaptation_strategies=['a']))),
        Policy("Info", function=partial(model_adaptation, wizard=Wizard(government_adaptation_strategies=["information"]))),
        Policy("Subsidy", function=partial(model_adaptation, wizard=Wizard(government_adaptation_strategies=["subsidy"]))),
        # Policy("Info + Subsidy", function=partial(model_adaptation, wizard=Wizard(government_adaptation_strategies=["information", "subsidy"]))),
        # Policy("Info + Subsidy + Dikes", function=partial(model_adaptation, wizard=Wizard(government_adaptation_strategies=["information", "subsidy", "dikes"]))),
    ]

    # Run experiments with the aforementioned parameters and outputs
    results = perform_experiments(models=model, scenarios=20, policies=policies)
    # results = perform_experiments(models=model, policies=policies)

    # Get the results
    experiments, outcomes = results

    save_results(results, f"{OUTPUT_DIR}/temporary.tar.gz")

    # DataFrame(experiments).to_pickle(f"{OUTPUT_DIR}/experiments.pkl")

    # for key in outcomes:
    #     np.save(os.path.join(OUTPUT_DIR, f"{key}.npy"), outcomes[key])
