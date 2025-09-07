# src/create_homer_load_files.py

import pandas as pd
import pathlib

print("Starting script to generate HOMER load profile files...")

# --- Configuration ---
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "homer_simulations.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "homer_load_profiles"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Load the Main Simulation Data ---
try:
    sim_data = pd.read_csv(INPUT_FILE)
    print(f"Successfully loaded {len(sim_data)} simulations from {INPUT_FILE}")
except FileNotFoundError:
    print(f"ERROR: Could not find the input file at {INPUT_FILE}")
    print("Please make sure your main spreadsheet is saved there.")
    exit()

# --- Generate a File for Each Simulation ---
for index, row in sim_data.iterrows():
    sim_id = int(row['Simulation_ID'])
    location = row['Location_Name']
    
    print(f"  -> Processing Simulation #{sim_id} ({location})...")

    # Create a full year's worth of hourly timestamps (8760 hours)
    hourly_index = pd.date_range(start='2025-01-01 00:00', end='2025-12-31 23:00', freq='h')
    load_profile = pd.DataFrame(index=hourly_index)
    load_profile['load_kW'] = 0.0

    # Get the 12 monthly kW values for this simulation
    monthly_kw_values = {
        1: row['kW_Jan'], 2: row['kW_Feb'], 3: row['kW_Mar'],
        4: row['kW_Apr'], 5: row['kW_May'], 6: row['kW_Jun'],
        7: row['kW_Jul'], 8: row['kW_Aug'], 9: row['kW_Sep'],
        10: row['kW_Oct'], 11: row['kW_Nov'], 12: row['kW_Dec']
    }

    # Apply the correct kW load to the pumping hours for each month
    for hour_timestamp in load_profile.index:
        if 8 <= hour_timestamp.hour <= 15:
            month = hour_timestamp.month
            load_profile.loc[hour_timestamp, 'load_kW'] = monthly_kw_values[month]

    # --- SAVE THE FILE IN THE CORRECT HOMER FORMAT ---
    # The filename must end in .dmd for HOMER to recognize it
    output_filename = OUTPUT_DIR / f"sim_{sim_id}_load_profile.dmd"
    
    # Save to a CSV with the specific timestamp format and no header
    load_profile.to_csv(
        output_filename, 
        header=False, 
        date_format='%m/%d/%Y %H:%M:%S'
    )

print(f"\nScript finished. All {len(sim_data)} load profile files have been generated in:")
print(f"{OUTPUT_DIR}")