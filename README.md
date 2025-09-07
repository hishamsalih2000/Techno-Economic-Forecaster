# Techno-Economic Forecaster for Solar Irrigation Systems

## 1. Project Overview

This project is a complete, end-to-end machine learning solution designed to solve a critical business problem: **automating the techno-economic feasibility analysis of solar-powered irrigation systems.**

The final product is a web-based application that allows a user to input basic farm parameters (location, size, and water depth) and receive an instant, data-driven forecast for the required PV system size and its lifetime cost (Net Present Cost). This transforms a multi-day, expert-driven simulation process into a real-time decision-making tool.

This project demonstrates a professional "bridge candidate" workflow, combining domain expertise in Mechanical & Renewable Energy Engineering with a robust Data Science and MLOps toolchain.

## 2. Foundational Research & Business Problem

The motivation for this project is grounded in my Bachelor of Science thesis, which conducted a deep-dive techno-economic analysis of solar vs. diesel irrigation for a 10-hectare farm in Sudan.

**[View the full thesis here.](https://github.com/hishamsalih2000/academic-work/blob/main/BSc_Thesis_Solar_Energy_Feasibility_Study.pdf)**

The thesis uncovered a clear and compelling business case:
*   **Massive Long-Term Savings:** While a diesel system has a lower initial cost, its 25-year lifetime cost (NPC) is **over 3.5 times higher** than a solar PV alternative ($45,177 vs. $12,890).
*   **Elimination of Operating Costs:** The solar system reduces annual operating costs by over **97%** ($3,278/yr for diesel vs. just $85/yr for solar).
*   **Significant Environmental Impact:** A single solar-powered system eliminates approximately **4.67 metric tons of CO2 emissions annually** compared to its diesel counterpart.

**The Bottleneck:** While the conclusion is clear, arriving at it is a slow, manual, and expert-driven process requiring specialized software (CROPWAT and HOMER Pro). This project's goal was to build a machine learning model to eliminate this bottleneck and provide these powerful insights instantly.

## 3. The End-to-End Workflow

The project was executed in three professional phases:

**Phase 1: Data Generation & Automation**
1.  **Simulation Blueprint:** A systematic plan was created to generate 50 unique scenarios, varying farm location, size, and borehole depth.
2.  **Input Automation:** A Python script (`src/create_homer_load_files.py`) was developed to automate the creation of 50 unique, hourly load profile (`.dmd`) files for HOMER Pro.
3.  **Simulation Execution:** Each scenario was simulated in HOMER Pro to find the optimal PV system configuration.
4.  **Results Aggregation:** A second Python script (`src/aggregate_results.py`) was created to automatically parse all 50 raw result files and build a clean, final dataset.

**Phase 2: Professional Model Training**
A robust training pipeline (`src/train_model.py`) was built, incorporating ML best practices like `Pipelines`, `RandomizedSearchCV`, and 5-fold cross-validation.

**Phase 3: Deployment (In Progress)**
The trained models are being deployed in an interactive web application using Streamlit (`app.py`).

## 4. Key Results & Performance

Two separate models were trained to predict the key outputs of the HOMER Pro simulation.

### Model 1: PV System Size Predictor
*   **Best Cross-Validated R²:** 0.7796
*   **Final Test Set R²:** 0.9312
*   **Final Test Set MAE:** 6.06 kW

The model demonstrates strong predictive power. The plot below shows a tight correlation between the model's predictions and the actual values from the test set.

![PV Size Performance Plot](./images/pv_size_predictor_performance_plot.png)

### Model 2: Net Present Cost (NPC) Predictor
*   **Best Cross-Validated R²:** 0.7734
*   **Final Test Set R²:** 0.9372
*   **Final Test Set MAE:** 2443.87 USD

![NPC Performance Plot](./images/npc_predictor_performance_plot.png)

### Feature Importance Analysis

As expected, `Farm_Size_ha` is the most dominant predictive feature for both models, accounting for over 70% of the decision-making process. This is followed by `Borehole_Depth_m` and the `Total_Irr_Req_mm_per_ha`, confirming that the model has learned the correct underlying physical and economic relationships.

## 5. How to Run This Project

1.  Clone this repository.
2.  Create and activate the conda environment:
    ```bash
    conda create --name homer-ml-predictor python=3.9
    conda activate homer-ml-predictor
    ```
3.  Install the required libraries:
    ```bash
    pip install -r requirements.
