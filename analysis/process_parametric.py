from paths import *
from model.model import AdaptationModel, Wizard
from model import utils
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

plt.style.use("./model/myBmh.mplstyle")


experiments = pd.read_pickle(f"{OUTPUT_DIR}/experiments.pkl")
outcomes = np.load(f"{OUTPUT_DIR}/TotalAdapted.npy")


plt.plot(outcomes.reshape(6, 80).T)
plt.show()
