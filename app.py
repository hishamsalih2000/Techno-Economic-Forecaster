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

# --- DATA-DRIVEN CONSTANTS (from Thesis & Research) ---
# These are our engineering assumptions for the calculation.
DIESEL_LITERS_PER_KWH = 0.4  # Fuel consumption rate from research
DIESEL_CAPITAL_PER_KW = 150  # Estimated cost for a diesel generator ($/kW)
DIESEL_MAINTENANCE_PER_HOUR = 0.05 # Estimated O&M cost ($/hr)
SOLAR_CAPITAL_RATIO = 0.915  # % of Solar NPC that is upfront capital

# --- Calculation and Plotting Function ---
def generate_dynamic_plot(predicted_solar_npc, farm_inputs, diesel_price_per_liter):
    """
    Generates a 25-year comparison plot based on a predicted solar cost
    and a dynamically calculated diesel cost.
    """
    # 1. Calculate Solar Costs
    solar_initial_capital = predicted_solar_npc * SOLAR_CAPITAL_RATIO
    solar_annual_op_cost = predicted_solar_npc * (1 - SOLAR_CAPITAL_RATIO) / 25

    # 2. Calculate Diesel Costs (The New, Dynamic Part)
    # First, estimate the annual kWh demand from the farm inputs. We'll use a simplified
    # version of our scaling logic, calibrated to the thesis baseline.
    baseline_kwh_per_day = 10.77 # From the 10ha Abu Hamed case
    scaling_factor = (farm_inputs['Farm_Size_ha'][0] / 10) * \
                     (farm_inputs['Borehole_Depth_m'][0] / 40) * \
                     (farm_inputs['Total_Irr_Req_mm_per_ha'][0] / 2311.5)
    
    annual_kwh_demand = baseline_kwh_per_day * 365 * scaling_factor
    
    # Estimate the required diesel generator size (kW)
    # Assuming peak demand is ~2.5x the average
    diesel_kw_size = (annual_kwh_demand / (365 * 8)) * 2.5 # 8 operating hours/day
    
    # Calculate the components of diesel cost
    diesel_initial_capital = diesel_kw_size * DIESEL_CAPITAL_PER_KW
    annual_fuel_liters = annual_kwh_demand * DIESEL_LITERS_PER_KWH
    annual_fuel_cost = annual_fuel_liters * diesel_price_per_liter
    # Assume generator runs ~2200 hours/year for this demand
    annual_maintenance_cost = 2200 * DIESEL_MAINTENANCE_PER_HOUR 
    diesel_annual_op_cost = annual_fuel_cost + annual_maintenance_cost

    # 3. Generate the cumulative cost data
    years = np.arange(0, 26)
    solar_cumulative = np.array([solar_initial_capital + year * solar_annual_op_cost for year in years])
    diesel_cumulative = np.array([diesel_initial_capital + year * diesel_annual_op_cost for year in years])

    # 4. Calculate the payback period
    payback_period = "> 25 Years"
    try:
        payback_year_idx = np.where(solar_cumulative < diesel_cumulative)[0][0]
        payback_period = years[payback_year_idx]
    except IndexError:
        pass
        
    # 5. Create the Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.style.use('seaborn-v0_8-whitegrid')
    ax.plot(years, solar_cumulative, marker='o', linestyle='-', color='g', label='Total Investment in Solar')
    ax.plot(years, diesel_cumulative, marker='o', linestyle='-', color='r', label='Total Investment in Diesel')

    if isinstance(payback_period, int):
        ax.axvline(x=payback_period, color='blue', linestyle='--', label=f'Payback Point (~{payback_period} Years)')
    
    ax.set_title("Investment Payback Forecast: Solar vs. Diesel", fontsize=16)
    ax.set_xlabel("Year of Operation", fontsize=12)
    ax.set_ylabel("Total Money Spent (USD)", fontsize=12)
    ax.legend()
    ax.grid(True)
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    return fig, payback_period

# --- Load Models (We only need two now) ---
@st.cache_resource
def load_models():
    pv_model = joblib.load(config.PV_MODEL_PATH)
    solar_npc_model = joblib.load(config.NPC_MODEL_PATH)
    return pv_model, solar_npc_model

pv_model, solar_npc_model = load_models()

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
    diesel_price = st.slider(
        label="Assumed Diesel Price ($ / Liter)",
        min_value=0.5, max_value=2.5, value=1.5, step=0.1
    )
    irr_req_map = {
        'Abu Hamed': 2311.5, 'Kadugli': 1587.5, 'Kassala': 1267.1,
        'Khartoum': 2046.4, 'Wadi-Halfa': 1941.0
    }
    irr_req = irr_req_map[location]
    submit_button = st.form_submit_button(label='▶️ Calculate Payback')

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
    
    with st.spinner('Running forecast...'):
        predicted_solar_npc = solar_npc_model.predict(input_data)[0]
        payback_plot, payback_period = generate_dynamic_plot(predicted_solar_npc, input_data, diesel_price)

    st.subheader("Investment Payback Period")
    st.metric(label=f"At ${diesel_price:.2f}/liter for diesel, your solar investment pays back in:", value=f"~ {payback_period} Years")
    st.info("Change the diesel price in the sidebar to see how it impacts your return on investment.", icon="💡")

    st.pyplot(payback_plot)

else:
    st.info("Enter your farm and diesel price assumptions in the sidebar and click 'Calculate Payback'.")