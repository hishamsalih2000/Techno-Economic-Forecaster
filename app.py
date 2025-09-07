import streamlit as st
import pandas as pd
import numpy as np
import joblib
import config
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="Solar Irrigation Forecaster",
    page_icon="☀️",
    layout="wide"
)

# --- Helper Function for Plotting ---
def generate_cash_flow_plot(predicted_pv_size, predicted_npc):
    """Generates a 25-year cumulative cost comparison plot."""
    
    # --- Baseline Cost Data (from your thesis, Table 5.4) ---
    # This is for a 30.59 kW solar system and its diesel equivalent.
    BASELINE_PV_SIZE = 30.59
    SOLAR_CAPITAL_BASELINE = 11791
    SOLAR_OM_BASELINE = 85
    DIESEL_CAPITAL_BASELINE = 2800
    DIESEL_OP_COST_BASELINE = 3278 # Fuel + O&M
    
    # --- Scale Costs Based on Prediction ---
    # Create a fair comparison by scaling the diesel costs relative to the predicted PV size.
    scaling_factor = predicted_pv_size / BASELINE_PV_SIZE
    
    # Estimate Solar capital cost as a fraction of the total predicted lifetime cost (NPC)
    # This is a reasonable assumption for visualization.
    solar_initial_capital = predicted_npc * 0.9  # Assume 90% of NPC is upfront cost
    solar_annual_om = predicted_npc * 0.1 / 25 # Spread the other 10% over 25 years

    diesel_initial_capital = DIESEL_CAPITAL_BASELINE * scaling_factor
    diesel_annual_op_cost = DIESEL_OP_COST_BASELINE * scaling_factor
    
    # --- Generate Cash Flow Data for 25 Years ---
    years = np.arange(0, 26)
    solar_cash_flow = np.zeros(26)
    diesel_cash_flow = np.zeros(26)
    
    solar_cash_flow[0] = -solar_initial_capital
    diesel_cash_flow[0] = -diesel_initial_capital
    
    for year in range(1, 26):
        solar_cash_flow[year] = solar_cash_flow[year-1] - solar_annual_om
        diesel_cash_flow[year] = diesel_cash_flow[year-1] - diesel_annual_op_cost
        
    # --- Create the Plot ---
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    ax.plot(years, solar_cash_flow, marker='o', linestyle='--', color='g', label='Solar PV System')
    ax.plot(years, diesel_cash_flow, marker='o', linestyle='--', color='r', label='Diesel System')
    
    ax.set_title("25-Year Cumulative Cost Comparison", fontsize=16)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Cumulative Cost (USD)", fontsize=12)
    ax.legend()
    ax.grid(True)
    
    # Format y-axis to show dollars
    ax.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'${x:,.0f}')
    )
    
    return fig

# --- Load Models ---
@st.cache_resource
def load_models():
    pv_model = joblib.load(config.PV_MODEL_PATH)
    npc_model = joblib.load(config.NPC_MODEL_PATH)
    return pv_model, npc_model

pv_model, npc_model = load_models()

# --- Application Header ---
st.title("☀️ Techno-Economic Forecaster for Solar Irrigation Systems")
st.markdown("This tool predicts the optimal PV system size and its lifetime cost, and visually compares the 25-year financial impact of choosing solar over diesel.")

# --- User Input Sidebar ---
st.sidebar.header("Enter Farm Parameters")
with st.sidebar.form(key='input_form'):
    # ... (rest of your sidebar code is the same) ...
    location = st.selectbox(
        label="Select Farm Location",
        options=['Abu Hamed', 'Kadugli', 'Kassala', 'Khartoum', 'Wadi-Halfa'],
        index=0
    )
    farm_size = st.slider(
        label="Select Farm Size (hectares)",
        min_value=5.0, max_value=25.0, value=10.0, step=2.5
    )
    borehole_depth = st.slider(
        label="Select Borehole Depth (meters)",
        min_value=20.0, max_value=80.0, value=40.0, step=5.0
    )
    irr_req_map = {
        'Abu Hamed': 2311.5, 'Kadugli': 1587.5, 'Kassala': 1267.1,
        'Khartoum': 2046.4, 'Wadi-Halfa': 1941.0
    }
    irr_req = irr_req_map[location]
    submit_button = st.form_submit_button(label='▶️ Run Financial Forecast')

# --- Prediction Logic and Display ---
if submit_button:
    if not (40.0 <= borehole_depth <= 60.0):
        st.warning(
            "Warning: The selected borehole depth is outside the model's training range (40m-60m). "
            "The prediction is an extrapolation and may have reduced accuracy.", 
            icon="⚠️"
        )
        
    input_data = pd.DataFrame({
        'Location_Name': [location], 'Farm_Size_ha': [farm_size],
        'Borehole_Depth_m': [borehole_depth], 'Total_Irr_Req_mm_per_ha': [irr_req]
    })
    
    st.header("Financial Forecast Results")
    
    with st.spinner('Calculating...'):
        predicted_pv_size = pv_model.predict(input_data)[0]
        predicted_npc = npc_model.predict(input_data)[0]

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Predicted PV System Size", value=f"{predicted_pv_size:.2f} kW")
    with col2:
        st.metric(label="Predicted Solar Lifetime Cost (NPC)", value=f"${predicted_npc:,.2f} USD")

    st.info("The forecast below compares the lifetime cost of your predicted solar system against a similarly scaled diesel-powered alternative.", icon="📊")

    # --- Display the new plot ---
    cash_flow_plot = generate_cash_flow_plot(predicted_pv_size, predicted_npc)
    st.pyplot(cash_flow_plot)

else:
    st.info("Please enter your farm parameters in the sidebar and click 'Run Financial Forecast' to see your results.")