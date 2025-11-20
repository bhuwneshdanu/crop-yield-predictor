# Save this file as 'app.py'

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- CONFIGURATION ---
DATA_FILE = 'yield_df.csv'
PRICE_DATA_FILE = 'Price_Agriculture_commodities_Week.csv' 
TARGET_COLUMN = 'hg/ha_yield'
SEED = 42

# --- UNIT CONVERSION CONSTANT ---
HG_HA_TO_KG_ACRE_FACTOR = 0.040469

# --- Price Integration Function ---
@st.cache_data(ttl=3600) 
def get_current_price(commodity_name):
    """
    Loads price data, filters for a specific commodity, and converts the price unit.
    Returns price_per_hg and price_per_kg for display.
    """
    try:
        df_price = pd.read_csv(PRICE_DATA_FILE)
    except FileNotFoundError:
        st.sidebar.error(f"Price file '{PRICE_DATA_FILE}' not found. Using default price.")
        return 3.00, 30.00 
    
    # CRITICAL FIX: Ensure commodity_name is treated as a string before .upper()
    commodity_name_str = str(commodity_name)
    commodity_name_upper = commodity_name_str.upper()
    
    if 'Commodity' in df_price.columns:
        df_filtered = df_price[df_price['Commodity'].str.upper() == commodity_name_upper]
    else:
        df_filtered = pd.DataFrame() 

    if not df_filtered.empty and 'Modal Price' in df_price.columns:
        df_filtered['Modal Price'] = pd.to_numeric(df_filtered['Modal Price'], errors='coerce')
        df_filtered = df_filtered.dropna(subset=['Modal Price'])
        
        if not df_filtered.empty:
            latest_modal_price = df_filtered['Modal Price'].mean() 
            
            price_per_hg = latest_modal_price / 1000.0
            price_per_kg = latest_modal_price / 100.0
            
            st.sidebar.caption(f"Price Source: ₹{latest_modal_price:,.0f}/Quintal (Avg)")
            return price_per_hg, price_per_kg
    
    return 3.00, 30.00


# --- MODEL TRAINING FUNCTION (Remains the same) ---
@st.cache_resource
def load_and_train_model():
    """Loads data, preprocesses, trains, selects the best model, and returns data for plotting."""
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        st.error(f"Error: Could not find '{DATA_FILE}'. Please ensure the file is in the same directory as app.py.")
        st.stop()
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        st.stop()

    # Data Cleaning and Preparation
    columns_to_drop = ['Year', 'average_item_value', 'Unnamed: 0']
    df = df.drop(columns_to_drop, axis=1, errors='ignore') 
    
    df = df.rename(columns={'hg/ha_yield': TARGET_COLUMN}, errors='ignore')
    df = df.dropna()

    if df.empty:
        st.error("Dataset is empty after cleaning.")
        st.stop()

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    numerical_features = X.select_dtypes(include=['number']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Model Training and Selection 
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=SEED),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=SEED)
    }

    best_r2 = -float('inf')
    best_pipeline = None

    results = {}
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results[name] = {'R2 Score': r2, 'MAE': mae, 'RMSE': rmse}

        if r2 > best_r2:
            best_r2 = r2
            best_pipeline = pipeline

    results_df = pd.DataFrame.from_dict(results, orient='index').sort_values(by='R2 Score', ascending=False)

    area_options = sorted(df['Area'].unique().tolist())
    item_options = sorted(df['Item'].unique().tolist())

    return best_pipeline, results_df, X_test, y_test, area_options, item_options, df

# --- Prediction and Revenue Functions ---
def predict_new_yield(model_pipeline, data):
    """Predicts crop yield based on user input (Output is hg/ha)."""
    predicted_yield = model_pipeline.predict(data)[0]
    return predicted_yield

def calculate_revenue(predicted_yield_hg_ha, crop_item):
    """Calculates total revenue based on predicted yield and dynamically fetched market price."""
    # FIX APPLIED: Passing crop_item to get_current_price
    price_per_hg, price_per_kg = get_current_price(crop_item)
        
    base_revenue = predicted_yield_hg_ha * price_per_hg
    risk_low_revenue = base_revenue * 0.90 
    return base_revenue, risk_low_revenue, price_per_kg

# --- STREAMLIT WEB APPLICATION (GUI) ---

st.set_page_config(
    page_title="Crop Yield Predictor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌾 Machine Learning Crop Yield Prediction System")
st.markdown("Developed for the BCA Final Project: **Crop Yield Prediction Using ML**")

# Load and train the model (This function runs only once)
results = load_and_train_model()
best_pipeline, results_df, X_test, y_test, area_options, item_options, full_df = results

st.subheader(f"🥇 Best Model Selected: **{best_pipeline.steps[-1][0]}** (based on R² Score)")

# Use two columns for layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Input Parameters")
    st.info("Adjust the sliders and selectors to predict the expected crop yield.")

    # 1. Numerical Inputs 
    max_rainfall = full_df['average_rain_fall_mm_per_year'].max()
    min_rainfall = full_df['average_rain_fall_mm_per_year'].min()
    
    rainfall = st.slider('Average Annual Rainfall (mm)', 
                         min_value=min_rainfall, 
                         max_value=max_rainfall, 
                         value=1500.0, step=10.0)
    temperature = st.slider('Average Temperature (°C)', min_value=0.0, max_value=40.0, value=25.0, step=0.1)
    pesticides = st.slider('Pesticides (tonnes)', min_value=0.0, max_value=100000.0, value=50000.0, step=1000.0)

    # 2. Categorical Inputs
    area = st.selectbox('Country/Area', options=area_options, index=area_options.index('India') if 'India' in area_options else 0)
    item = st.selectbox('Crop/Item', options=item_options, index=item_options.index('Rice') if 'Rice' in item_options else 0)

    # Prepare input data and Run Prediction
    new_data = pd.DataFrame({
        'average_rain_fall_mm_per_year': [rainfall],
        'avg_temp': [temperature],
        'pesticides_tonnes': [pesticides],
        'Area': [area],
        'Item': [item]
    })

    # Output is in hg/ha
    final_prediction_hg_ha = predict_new_yield(best_pipeline, new_data)
    
    # CONVERSION: hg/ha to kg/acre for display
    final_prediction_kg_acre = final_prediction_hg_ha * HG_HA_TO_KG_ACRE_FACTOR
    
    # --- ENHANCED FEATURE: Revenue Forecast Display ---
    base_revenue, risk_low_revenue, price_per_kg = calculate_revenue(final_prediction_hg_ha, item)
    
    st.success(f"## ✅ Predicted Yield: {final_prediction_kg_acre:,.2f} kg/acre")
    
    st.subheader("💰 Financial Decision Support")
    
    # Use columns for clean, side-by-side display of metrics
    col_metric_1, col_metric_2, col_metric_3 = st.columns(3)
    
    # METRIC 1: Expected Revenue
    col_metric_1.metric(
        label="Expected Revenue (₹/ha)", 
        value=f"₹ {base_revenue:,.0f}"
    )

    # METRIC 2: Price Risk Scenario
    col_metric_2.metric(
        label="Revenue @ 10% Risk (₹/ha)", 
        value=f"₹ {risk_low_revenue:,.0f}",
        delta=f"₹ {-1 * (base_revenue - risk_low_revenue):,.0f} loss", 
        delta_color="inverse"
    )
    
    # METRIC 3: Live Price Rate
    col_metric_3.metric(
        label="Current Market Price (Avg)", 
        value=f"₹ {price_per_kg:.2f}",
        help="Price per kg is dynamically calculated from the AGMARKNET dataset."
    )

    st.caption("_Yield is shown in kg/acre; Revenue is shown in ₹/hectare._")


with col2:
    st.header("Model Evaluation & Live Analysis")

    st.subheader("Performance Metrics (Static)")
    st.dataframe(results_df, use_container_width=True)

    st.subheader("Live Rainfall Sensitivity Plot")
    st.markdown("Shows how predicted yield changes across the full range of rainfall, keeping all other inputs fixed.")

    # --- LIVE PLOT GENERATION (CONVERTED Y-AXIS) ---
    rainfall_range = np.linspace(min_rainfall, max_rainfall, 100)
    
    plot_data = pd.DataFrame({
        'average_rain_fall_mm_per_year': rainfall_range,
        'avg_temp': [temperature] * 100,
        'pesticides_tonnes': [pesticides] * 100,
        'Area': [area] * 100,
        'Item': [item] * 100
    })

    # Predict in hg/ha, then convert for the plot
    sensitivity_predictions_hg_ha = best_pipeline.predict(plot_data)
    sensitivity_predictions_kg_acre = sensitivity_predictions_hg_ha * HG_HA_TO_KG_ACRE_FACTOR

    plot_df = pd.DataFrame({
        'Rainfall (mm)': rainfall_range,
        'Predicted Yield (kg/acre)': sensitivity_predictions_kg_acre
    })
    
    # Plot the sensitivity curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(plot_df['Rainfall (mm)'], plot_df['Predicted Yield (kg/acre)'], label='Predicted Yield', color='darkgreen')
    
    # Add a marker for the current input point
    ax.scatter(rainfall, final_prediction_kg_acre, color='red', s=100, label='Current Input', zorder=5)

    ax.set_xlabel('Rainfall (mm)')
    ax.set_ylabel('Predicted Yield (kg/acre)') 
    ax.set_title(f'Sensitivity of Yield for {item}')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)