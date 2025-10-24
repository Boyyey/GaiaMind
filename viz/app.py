import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import os
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

# Set page config for a professional look
st.set_page_config(page_title="GaiaMind Simulator", page_icon="🌍", layout="wide")

# Title and Description with enhanced styling
st.title("🌍 GaiaMind: Advanced Planetary Intelligence Simulator")
st.markdown("""
**Welcome to GaiaMind!**  
This is a sophisticated hybrid C-Python simulation platform modeling Earth's ecosystems, energy dynamics, and human impacts.  
Built with AI-driven agents, real-time data integration, and interactive visualizations.  
Adjust parameters, run simulations, and explore scenarios for global sustainability.  
*Powered by Reinforcement Learning and Multi-Agent Systems for PhD-level analysis.*
""")

# Sidebar for parameters with expanded options
st.sidebar.header("🔧 Simulation Parameters")
st.sidebar.markdown("Customize initial conditions below:")
population = st.sidebar.slider("🌍 Initial Population (billions)", 1, 15, 8, help="Total world population in billions") * 1000000000
energy = st.sidebar.slider("⚡ Initial Energy Consumption (units)", 500, 3000, 1000, help="Base energy demand")
pollution = st.sidebar.slider("☣️ Initial Pollution Level (0-100)", 0, 100, 50, help="Starting pollution index")
renewable_ratio = st.sidebar.slider("🌿 Renewable Energy Ratio (0-1)", 0.0, 1.0, 0.2, step=0.01, help="Fraction of energy from renewables")
economy = st.sidebar.slider("💰 Initial Economy Index", 50, 200, 100, help="Economic strength metric")
temp_rise_rate = st.sidebar.slider("🌡️ Temperature Rise Rate (per year)", 0.0, 0.1, 0.02, step=0.01, help="Annual global temperature increase")
pop_growth = st.sidebar.slider("📈 Population Growth Rate", 0.0, 0.05, 0.01, step=0.001, help="Annual population growth")
ai_efficiency = st.sidebar.slider("🤖 AI Policy Efficiency (0-1)", 0.0, 1.0, 0.5, step=0.01, help="How effective AI agents are in policy decisions")

# Additional options in sidebar
st.sidebar.subheader("Advanced Options")
use_climate_data = st.sidebar.checkbox("Integrate Real Climate Data", value=True, help="Load NASA climate trends")
run_ai_prediction = st.sidebar.checkbox("Enable AI Policy Predictions", value=True, help="Use trained ML model for insights")
num_scenarios = st.sidebar.selectbox("Number of Scenarios", [1, 3, 5], index=1, help="Run multiple simulation variants")
save_results = st.sidebar.checkbox("Save Results to File", value=False, help="Export simulation data")

# Function to load and process data
def load_data():
    """Load initial conditions and climate data with error handling."""
    try:
        initial_data = pd.read_csv('../data/initial_conditions.csv')
        climate_data = pd.read_csv('../data/nasa_climate.csv') if use_climate_data else None
        return initial_data, climate_data
    except FileNotFoundError as e:
        st.error(f"Data file missing: {e}. Please ensure data/ folder has required CSV files.")
        return None, None

# Function to update initial conditions
def update_conditions(data, params):
    """Update CSV with new parameters."""
    updates = {
        'Population': params['population'],
        'Energy': params['energy'],
        'Pollution': params['pollution'],
        'RenewableRatio': params['renewable_ratio'],
        'Economy': params['economy'],
        'TemperatureRiseRate': params['temp_rise_rate'],
        'PopulationGrowth': params['pop_growth'],
        'AIPolicyEfficiency': params['ai_efficiency']
    }
    for key, value in updates.items():
        data.loc[data['Parameter'] == key, 'Value'] = value
    data.to_csv('../data/initial_conditions.csv', index=False)

# Function to run C simulation
def run_simulation():
    """Execute the C engine simulation via subprocess."""
    try:
        os.chdir('../core_c')
        st.info("Compiling and running C simulation...")
        compile_result = subprocess.run(['make'], capture_output=True, text=True)
        if compile_result.returncode != 0:
            st.error(f"Compilation failed: {compile_result.stderr}")
            return False
        run_result = subprocess.run(['./simulator'], capture_output=True, text=True)
        if run_result.returncode != 0:
            st.error(f"Simulation failed: {run_result.stderr}")
            return False
        os.chdir('../viz')
        st.success("Simulation completed successfully!")
        return True
    except Exception as e:
        st.error(f"Error running simulation: {str(e)}")
        return False

# Function to load results
def load_results():
    """Load simulation results with validation."""
    try:
        results = pd.read_csv('../results/world_log.csv')
        if results.empty:
            st.warning("No simulation results found. Please run the simulation first.")
            return None
        return results
    except FileNotFoundError:
        st.error("Results file not found. Run the simulation to generate data.")
        return None

# Function for AI predictions
def load_ai_model():
    """Load the trained AI model if available."""
    try:
        model = joblib.load('../ai_python/sustainability_model.pkl')
        return model
    except FileNotFoundError:
        st.warning("AI model not found. Train it using ai_python/train_model.py first.")
        return None

# Advanced plotting functions
def plot_pollution_over_time(results):
    """Plot pollution trends with annotations."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results['Year'], results['Pollution'], color='red', linewidth=2, label='Pollution Level')
    ax.fill_between(results['Year'], results['Pollution'], alpha=0.3, color='red')
    ax.axhline(y=80, color='orange', linestyle='--', label='Collapse Threshold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Pollution Index', fontsize=12)
    ax.set_title('🌫️ Pollution Trends Over Time', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.5)
    st.pyplot(fig)

def plot_economy_vs_pollution(results):
    """Scatter plot with color mapping and trend line."""
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(results['Pollution'], results['Economy'], c=results['Year'], cmap='viridis', s=50, alpha=0.7)
    # Add trend line with error handling
    try:
        z = np.polyfit(results['Pollution'], results['Economy'], 1)
        p = np.poly1d(z)
        ax.plot(results['Pollution'], p(results['Pollution']), color='blue', linestyle='--', label='Trend Line')
    except np.linalg.LinAlgError:
        st.warning("Trend line could not be computed due to insufficient data variation. Displaying scatter plot only.")
    ax.set_xlabel('Pollution Level')
    ax.set_ylabel('Economy Index')
    ax.set_title('💰 Economy vs. Pollution Scatter', fontsize=14)
    ax.legend()
    plt.colorbar(scatter, label='Year')
    st.pyplot(fig)

def plot_sustainability_index(results):
    """Sustainability plot with dual axes."""
    sustainability = 100 - results['Pollution']
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(results['Year'], sustainability, color='green', linewidth=2, label='Sustainability Index')
    ax1.set_ylabel('Sustainability (%)', color='green')
    ax1.tick_params(axis='y', labelcolor='green')
    ax2 = ax1.twinx()
    ax2.plot(results['Year'], results['Economy'], color='blue', linestyle=':', linewidth=2, label='Economy Index')
    ax2.set_ylabel('Economy Index', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax1.set_title('🌱 Sustainability and Economy Over Time', fontsize=14)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid(True, alpha=0.5)
    st.pyplot(fig)

def plot_energy_distribution(results):
    """Bar chart for energy sources."""
    fossil_energy = results['Energy'] * (1 - results['RenewableRatio'])
    renewable_energy = results['Energy'] * results['RenewableRatio']
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(results['Year'], fossil_energy, color='brown', label='Fossil Fuels')
    ax.bar(results['Year'], renewable_energy, bottom=fossil_energy, color='green', label='Renewables')
    ax.set_xlabel('Year')
    ax.set_ylabel('Energy Consumption')
    ax.set_title('⚡ Energy Source Distribution', fontsize=14)
    ax.legend()
    st.pyplot(fig)

def plot_climate_impact(climate_data, results):
    """Overlay climate data with simulation."""
    if climate_data is None:
        st.info("Climate data integration disabled.")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(climate_data['Year'], climate_data['TemperatureRise'], color='orange', marker='o', label='Real Temperature Rise')
    ax.plot(results['Year'], results['Pollution'] / 10, color='red', linestyle='--', label='Simulated Impact')
    ax.set_xlabel('Year')
    ax.set_ylabel('Temperature Rise / Impact')
    ax.set_title('🌡️ Climate Data Integration', fontsize=14)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def plot_heatmap(results):
    """Heatmap of correlations."""
    corr = results.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, vmin=-1, vmax=1)
    ax.set_title('🔥 Correlation Heatmap of Variables', fontsize=14)
    st.pyplot(fig)

def ai_policy_suggestions(results, model):
    """Use AI model for policy insights."""
    if model is None:
        return
    latest = results.iloc[-1]
    features = np.array([[latest['Year'], latest['Population'], latest['Energy'], latest['Pollution'], latest['RenewableRatio']]])
    prediction = model.predict(features)[0]
    st.subheader("🤖 AI Policy Predictions")
    st.metric("Predicted Sustainability Score", f"{prediction:.2f}")
    if prediction < 70:
        st.error("AI Suggests: Immediate action needed – invest in renewables and reduce emissions.")
    else:
        st.success("AI Suggests: Policies are effective; monitor for long-term stability.")

# Main simulation logic
if st.button("🚀 Run Full Simulation", type="primary"):
    with st.spinner("Processing simulation parameters..."):
        initial_data, climate_data = load_data()
        if initial_data is None:
            st.stop()
        params = {
            'population': population, 'energy': energy, 'pollution': pollution,
            'renewable_ratio': renewable_ratio, 'economy': economy,
            'temp_rise_rate': temp_rise_rate, 'pop_growth': pop_growth, 'ai_efficiency': ai_efficiency
        }
        update_conditions(initial_data, params)
    if run_simulation():
        results = load_results()
        if results is not None:
            model = load_ai_model()
            st.success("Simulation data loaded! Displaying results below.")

# Tabs for organized display
tab1, tab2, tab3, tab4 = st.tabs(["📊 Basic Results", "🔬 Advanced Analysis", "🌍 Climate Integration", "🤖 AI Insights"])

with tab1:
    st.header("📊 Basic Simulation Results")
    results = load_results()
    if results is not None:
        st.dataframe(results.head(10))
        plot_pollution_over_time(results)
        plot_economy_vs_pollution(results)
        plot_sustainability_index(results)
        plot_energy_distribution(results)
        # Final status with details
        final = results.iloc[-1]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Final Population", f"{final['Population'] / 1e9:.1f}B")
        with col2:
            st.metric("Final Pollution", f"{final['Pollution']:.1f}")
        with col3:
            st.metric("Final Economy", f"{final['Economy']:.1f}")
        if final['Pollution'] > 80:
            st.error("🚨 Collapse Scenario: High pollution indicates ecosystem breakdown.")
        elif final['Economy'] < 50:
            st.warning("⚠️ Economic Crisis: Adjust strategies for recovery.")
        else:
            st.success("✅ Sustainable Path: Earth is thriving!")

with tab2:
    st.header("🔬 Advanced Analysis")
    results = load_results()
    if results is not None:
        plot_heatmap(results)
        st.markdown("**Key Insights:**")
        st.write("- Population growth correlates with energy demand.")
        st.write("- High renewable ratios improve sustainability.")
        # Additional plots
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.stackplot(results['Year'], results['Population'] / 1e9, results['Economy'], labels=['Population (B)', 'Economy'])
        ax.set_title('📈 Population and Economy Stacked', fontsize=14)
        ax.legend()
        st.pyplot(fig)

with tab3:
    st.header("🌍 Real-World Climate Integration")
    results = load_results()
    initial_data, climate_data = load_data()
    if results is not None and climate_data is not None:
        plot_climate_impact(climate_data, results)
        st.markdown("**Climate Data Notes:**")
        st.write("Integrating NASA data for realistic temperature projections.")
        st.dataframe(climate_data)

with tab4:
    st.header("🤖 AI-Powered Insights")
    results = load_results()
    model = load_ai_model()
    if results is not None:
        ai_policy_suggestions(results, model)
        # More AI features
        st.subheader("Scenario Planning")
        if num_scenarios > 1:
            for i in range(num_scenarios):
                st.write(f"Scenario {i+1}: Vary renewable ratio by {i*0.1}")
        st.markdown("**Model Details:** Trained on simulation data using Random Forest.")

# Footer
st.markdown("---")
st.markdown("**GaiaMind v1.0** | Built for Sustainability Research | © 2025 AmirHosseinRasti | MIT License")
st.info("For advanced features like distributed computing or API integrations, check the README.")
