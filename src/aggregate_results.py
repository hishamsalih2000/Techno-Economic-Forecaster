# src/aggregate_results.py

import pandas as pd
import pathlib

print("Starting script to aggregate HOMER simulation results...")

# --- Configuration ---
# Define all the paths we will be using
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
SIMULATION_PARAMETERS_FILE = PROJECT_ROOT / "data" / "raw" / "homer_simulations.csv"
RESULTS_DIR = PROJECT_ROOT / "data" / "homer_sim_results"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FINAL_DATASET_FILE = PROCESSED_DATA_DIR / "final_ml_dataset.csv"

# Create the processed data folder if it doesn't exist
PROCESSED_DATA_DIR.mkdir(exist_ok=True)

# --- Load the master list of simulation inputs ---
try:
    main_df = pd.read_csv(SIMULATION_PARAMETERS_FILE)
    print(f"Successfully loaded {len(main_df)} simulation parameters.")
except FileNotFoundError:
    print(f"ERROR: Could not find the main simulation file at {SIMULATION_PARAMETERS_FILE}")
    exit()

# --- Loop through each simulation, find its result file, and extract the data ---
results_list = []
for index, row in main_df.iterrows():
    sim_id = int(row['Simulation_ID'])
    
    # Construct the path to the result file for this simulation
    results_file_path = RESULTS_DIR / f"sim_{sim_id}_results.csv"
    
    if results_file_path.exists():
        try:
            print(f"  -> Processing results for Simulation #{sim_id}...")
            # Load the results CSV
            result_df = pd.read_csv(results_file_path, sep=',', skiprows=1)
            
            # The best result is always the first row of the file
            optimal_result = result_df.iloc[0]
            
            # Extract the three key values using their exact column names
            pv_size = optimal_result['Architecture/PV (kW)']
            npc = optimal_result['Cost/NPC ($)']
            lcoe = optimal_result['Cost/LCOE ($/kWh)']
            
            # Store the extracted data in a dictionary
            extracted_data = {
                'Simulation_ID': sim_id,
                'Required_PV_Size_kW': pv_size,
                'NPC_USD': npc,
                'LCOE_USD_per_kWh': lcoe
            }
            results_list.append(extracted_data)

        except Exception as e:
            print(f"  -> ERROR processing file {results_file_path}: {e}. Skipping.")
    else:
        print(f"  -> WARNING: Results file not found for Simulation #{sim_id}. Skipping.")

print(f"\nSuccessfully processed {len(results_list)} result files.")

# --- Combine the inputs and results and save the final dataset ---
if results_list:
    # Convert the list of results into a DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Merge the original input parameters with the new results based on the Simulation_ID
    final_dataset = pd.merge(main_df, results_df, on='Simulation_ID')
    
    # Save the final, clean dataset to the processed folder
    final_dataset.to_csv(FINAL_DATASET_FILE, index=False)
    
    print(f"Successfully created the final dataset with {len(final_dataset)} entries.")
    print(f"Final dataset saved to: {FINAL_DATASET_FILE}")
else:
    print("No results were processed. The final dataset was not created.")