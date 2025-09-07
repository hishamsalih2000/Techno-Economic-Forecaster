# src/aggregate_results.py

import pandas as pd
import sqlite3
import config 

print("Starting script to aggregate HOMER results into SQL database...")

# --- Load the master list of simulation inputs ---
try:
    main_df = pd.read_csv(config.RAW_DATA_INPUT_FILE)
    print(f"Successfully loaded {len(main_df)} simulation parameters.")
except FileNotFoundError:
    print(f"ERROR: Could not find the raw input file at {config.RAW_DATA_INPUT_FILE}")
    print("Please ensure your 'homer_simulations.csv' is in the 'data/raw' folder.")
    exit()

# --- Loop through each simulation and extract the data ---
results_list = []
for index, row in main_df.iterrows():
    sim_id = int(row['Simulation_ID'])
    # Use the new config variable for the results directory
    results_file_path = config.HOMER_RESULTS_DIR / f"sim_{sim_id}_results.csv"
    
    if results_file_path.exists():
        try:
            result_df = pd.read_csv(results_file_path, sep=',', skiprows=1)
            optimal_result = result_df.iloc[0]
            
            extracted_data = {
                'Simulation_ID': sim_id,
                'Required_PV_Size_kW': optimal_result['Architecture/PV (kW)'],
                'NPC_USD': optimal_result['Cost/NPC ($)'],
                'LCOE_USD_per_kWh': optimal_result['Cost/LCOE ($/kWh)']
            }
            results_list.append(extracted_data)
        except Exception as e:
            print(f"  -> ERROR processing file {results_file_path}: {e}. Skipping.")
    else:
        print(f"  -> WARNING: Results file not found for Simulation #{sim_id}. Skipping.")

print(f"\nSuccessfully processed {len(results_list)} result files.")

# --- Combine inputs and results ---
if results_list:
    results_df = pd.DataFrame(results_list)
    final_dataset = pd.merge(main_df, results_df, on='Simulation_ID')
    
    # Write to SQL Database ---
    print(f"Connecting to database at: {config.DATABASE_PATH}")
    # This creates a connection; the file will be created if it doesn't exist.
    con = sqlite3.connect(config.DATABASE_PATH)
    
    # Use pandas' .to_sql() method to write the DataFrame to a table.
    # if_exists='replace' means it will overwrite the table every time you run it.
    final_dataset.to_sql(
        config.TABLE_NAME, 
        con, 
        if_exists='replace', 
        index=False
    )
    
    print(f"Successfully wrote {len(final_dataset)} entries to table '{config.TABLE_NAME}' in the database.")
    con.close() # Close the connection
else:
    print("No results were processed. Database was not written.")