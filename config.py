# config.py
import pathlib

# --- Core Paths ---
# Define the absolute root of the project
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent

# Define key folder paths
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
IMAGES_DIR = PROJECT_ROOT / "images"

# Create the folders if they don't exist to prevent errors
MODELS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
(DATA_DIR / "processed").mkdir(exist_ok=True)
(DATA_DIR / "raw").mkdir(exist_ok=True)


# --- File Paths for Data and Models ---
FINAL_DATASET_PATH = DATA_DIR / "processed" / "final_ml_dataset.csv"
PV_MODEL_PATH = MODELS_DIR / "pv_size_predictor.joblib"
NPC_MODEL_PATH = MODELS_DIR / "npc_predictor.joblib"


# --- Model Training Configuration ---
# The "clues" our model will use to make predictions
FEATURES = [
    'Location_Name', 
    'Farm_Size_ha', 
    'Borehole_Depth_m', 
    'Total_Irr_Req_mm_per_ha'
]

# Settings for reproducible experiments
RANDOM_STATE = 42
TEST_SIZE = 0.2