# Start Here

Please visit the complete documentation at https://abm-flood-adaptation.readthedocs.io/en/latest

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

abm-flood-adaptation
├── analysis
│ ├── build
│ ├── make.bat
│ ├── Makefile
│ └── source
├── analysis
│ ├── `analyse.ipynb` (Main processing file for the results)
│ ├── `demo.ipynb` (Legacy demonstration from the original model)
│ ├── `minimal.py` (Minimal runnable simulation example)
│ ├── `parametric.py` (Using EMA workbench to sweep parameter space and find batch solutions)
│ └── `paths.py` (Utility to ensure correct path availability)
├── docs
│ └── ... (Builds the documentation)
├── documents
│ └── ... (Documents and articles related to the work)
├── input_data
│ ├── floodmaps (.tif files)
│ ├── floodplain (.cpg, .dbf, .prj, .shp, .shx filex)
│ └── model_domain (Houston shapes)
├── model
│ ├── `functions.py`(Functions related to the flood model) 
│ ├── `agents.py` (Agent classes and implementation of the RiskModel)
│ ├── `model.py` (Model implementation)
│ ├── `server.py` (Server description for running visualisations)
│ └── `utils.py` (Utility functions)
├── output
│ └── ... (Stores images, output data and other results)
├── tests
│ ├── `test_agent.py` (Testing simple agent functionality)
│ └── `test_risk_model` (Testing solely the implementation of the risk model)
├── README.md (This readme)
├── requirements.txt 
└── ... (dot-files for setup and documentation)

## Usage
Feel free to extend the model at will and play around with anything. An Reusable Building Block of the Risk Model is also available at https://www.agentblocks.org/

