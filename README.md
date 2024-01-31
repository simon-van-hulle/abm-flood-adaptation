# Start Here

Please visit the complete documentation at https://abm-flood-adaptation.readthedocs.io/en/latest. You can also build the documentation yourself (See section [Documentation](#documentation))

## Introduction
This directory contains a minimal agent-based model (ABM) implemented in Python, focused on simulating household adaptation to flood events in a social network context. It uses the Mesa framework for ABM and incorporates geographical data processing for flood depth and damage calculations.

## Installation
To set up the project environment, follow these steps:
1. Clone the repository to your local machine.
2. Install required dependencies:
   ```bash
   pip install -U mesa geopandas shapely rasterio networkx
   ```

## Folder Structure

abm-flood-adaptation <br>
├── analysis <br>
│   ├── `analyse.ipynb` (Main processing file for the results) <br>
│   ├── `demo.ipynb` (Legacy demonstration from the original model) <br>
│   ├── `minimal.py` (Minimal runnable simulation example) <br>
│   ├── `parametric.py` (Using EMA workbench to sweep parameter space and find batch solutions) <br>
│   └── `paths.py` (Utility to ensure correct path availability) <br>
├── docs <br>
│   └── ... (Builds the documentation) <br>
├── documents <br>
│   └── ... (Documents and articles related to the work) <br>
├── input_data <br>
│   ├── floodmaps (.tif files) <br>
│   ├── floodplain (.cpg, .dbf, .prj, .shp, .shx filex) <br>
│   └── model_domain (Houston shapes) <br>
├── model <br>
│   ├── `functions.py`(Functions related to the flood model)  <br>
│   ├── `agents.py` (Agent classes and implementation of the RiskModel) <br>
│   ├── `model.py` (Model implementation) <br>
│   ├── `server.py` (Server description for running visualisations) <br>
│   └── `utils.py` (Utility functions) <br>
├── output <br>
│   └── ... (Stores images, output data and other results) <br>
├── tests <br>
│   ├── `test_agent.py` (Testing simple agent functionality) <br>
│   └── `test_risk_model` (Testing solely the implementation of the risk model) <br>
├── `README.md` (This readme) <br>
├── `requirements.txt`  <br>
└── ... (dot-files for setup and documentation) <br>



## Usage

To ensure the model runs on your computer, please follow the following steps:
1. Create a new virtual environment with your favourite environment manager. For example:
```bash
$ python3 -m venv -n abm-env python=3.12
$ source abm-env/bin/activate
```
2. Install the required packages. For example:
```bash
$ pip install -r requirements.txt
```
3. Ensure the build is successful by running the minimal example
```bash
$ python3 analysis/minimal.py
```
4. You can now adapt the implementations in the analysis folder to adapt the model

## Documentation

You can either find the documentation at https://abm-flood-adaptation.readthedocs.io/en/latest or build the documentation yourself. For the latter, first ensure your environment is activated and sphinx is installed. Then run the following command from the main directory.
```bash
$ sphinx-build docs build
```
The documentation will be stored in the build folder and you can access it by opening `build/index.html` in your favourite browser.


## Extending the model

Feel free to extend the model at will and play around with anything. An Reusable Building Block of the Risk Model is also available at https://www.agentblocks.org/

