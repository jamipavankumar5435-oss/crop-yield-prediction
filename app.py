import streamlit as st
import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Crop Yield Prediction System",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Crop Yield Prediction System")

st.markdown("""
### Scalable Big Data Analytics for Crop Yield Prediction
Using Multi-Source Agricultural Data

Enter agricultural, soil and weather information
to estimate crop yield using Machine Learning.
""")

# ==========================================
# LOAD TRAINED MODEL PACKAGE
# ==========================================

@st.cache_resource
def load_model_package():
    model_path = Path(__file__).parent / "crop_yield_models.pkl"

    with open(model_path, "rb") as file:
        package = pickle.load(file)

    return package


try:
    package = load_model_package()

    models = package["models"]
    best_model = package["best_model"]
    best_model_name = package["best_model_name"]

    FEATURE_COLUMNS = package["feature_columns"]
    NUMERIC_FEATURES = package["numeric_features"]
    CATEGORIES = package["categories"]

except Exception as e:
    st.error(f"Unable to load model package: {e}")
    st.stop()


# ==========================================
# PREPARE INPUT DATA
# ==========================================

def prepare_input(
    district, crop, season,
    rainfall, temperature, humidity, soil_ph,
    nitrogen, phosphorus, potassium,
    irrigation, fertilizer, area
):

    values = {
        "rainfall_mm": rainfall,
        "avg_temp_c": temperature,
        "humidity_pct": humidity,
        "soil_ph": soil_ph,
        "nitrogen_kg_ha": nitrogen,
        "phosphorus_kg_ha": phosphorus,
        "potassium_kg_ha": potassium,
        "irrigation_mm": irrigation,
        "fertilizer_kg_ha": fertilizer,
        "area_ha": area
    }

    row = dict(values)

    categorical_values = {
        "district": district,
        "crop": crop,
        "season": season
    }

    for column, selected in categorical_values.items():

        for category in CATEGORIES[column]:

            feature_name = f"{column}_{category}"

            if feature_name in FEATURE_COLUMNS:
                row[feature_name] = int(selected == category)

    input_df = pd.DataFrame(
        [[row.get(col, 0) for col in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS
    )

    return input_df


# ==========================================
# USER INPUT INTERFACE
# ==========================================

st.header("1. Agricultural Information")

col1, col2 = st.columns(2)

with col1:

    district = st.selectbox(
        "District",
        options=CATEGORIES["district"]
    )

    crop = st.selectbox(
        "Crop",
        options=CATEGORIES["crop"]
    )

with col2:

    season = st.selectbox(
        "Season",
        options=CATEGORIES["season"]
    )

    model_name = st.selectbox(
        "Machine Learning Model",
        options=list(models.keys()),
        index=list(models.keys()).index(best_model_name)
    )


st.header("2. Weather and Soil Information")

col1, col2, col3 = st.columns(3)

with col1:
    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        value=720.0
    )

with col2:
    temperature = st.number_input(
        "Average Temperature (°C)",
        value=28.0
    )

with col3:
    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=65.0
    )

col1, col2 = st.columns(2)

with col1:
    soil_ph = st.number_input(
        "Soil pH",
        min_value=0.0,
        max_value=14.0,
        value=6.7
    )


st.header("3. Fertilizer and Farm Information")

col1, col2, col3 = st.columns(3)

with col1:
    nitrogen = st.number_input(
        "Nitrogen (kg/ha)",
        min_value=0.0,
        value=110.0
    )

with col2:
    phosphorus = st.number_input(
        "Phosphorus (kg/ha)",
        min_value=0.0,
        value=55.0
    )

with col3:
    potassium = st.number_input(
        "Potassium (kg/ha)",
        min_value=0.0,
        value=66.0
    )

col1, col2, col3 = st.columns(3)

with col1:
    irrigation = st.number_input(
        "Irrigation (mm)",
        min_value=0.0,
        value=250.0
    )

with col2:
    fertilizer = st.number_input(
        "Fertilizer (kg/ha)",
        min_value=0.0,
        value=180.0
    )

with col3:
    area = st.number_input(
        "Farm Area (ha)",
        min_value=0.01,
        value=3.0
    )


# ==========================================
# PREDICTION
# ==========================================

st.divider()

st.header("4. Crop Yield Prediction")

if st.button("🌱 Predict Crop Yield", type="primary"):

    try:

        input_df = prepare_input(
            district, crop, season,
            rainfall, temperature, humidity, soil_ph,
            nitrogen, phosphorus, potassium,
            irrigation, fertilizer, area
        )

        selected_model = models[model_name]

        prediction = float(
            selected_model.predict(input_df)[0]
        )

        st.success("Crop yield prediction completed!")

        st.subheader("Predicted Crop Yield")

        st.metric(
            label="Estimated Yield",
            value=f"{prediction:.3f} tonnes/hectare"
        )

        st.write(f"**Selected Model:** {model_name}")

        st.write(
            f"**Default Best Model:** {best_model_name}"
        )

        # ==================================
        # EXPLAINABLE AI USING SHAP
        # ==================================

        st.divider()

        st.header("5. Explainable AI (XAI)")
        st.subheader("SHAP Feature Impact")

        if model_name != "KNN":

            try:

                explainer = shap.TreeExplainer(
                    selected_model
                )

                shap_values = explainer.shap_values(
                    input_df
                )

                if isinstance(shap_values, list):
                    shap_values = shap_values[0]

                contributions = np.asarray(
                    shap_values
                ).reshape(-1)

                importance = pd.DataFrame({
                    "Feature": FEATURE_COLUMNS,
                    "SHAP Value": contributions
                })

                importance["Absolute Impact"] = (
                    importance["SHAP Value"].abs()
                )

                importance = importance.nlargest(
                    10, "Absolute Impact"
                ).sort_values("Absolute Impact")

                fig, ax = plt.subplots(figsize=(9, 5))

                ax.barh(
                    importance["Feature"],
                    importance["SHAP Value"]
                )

                ax.axvline(
                    0,
                    color="black",
                    linewidth=0.8
                )

                ax.set_title("SHAP Feature Impact")
                ax.set_xlabel(
                    "Impact on predicted yield"
                )
                ax.set_ylabel("Input Features")

                plt.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

                st.info("""
                Positive SHAP values indicate features
                that increase the prediction relative to
                the model baseline.

                Negative SHAP values indicate features
                that decrease the prediction.
                """)

            except Exception as e:

                st.warning(
                    f"SHAP explanation unavailable: {e}"
                )

        else:

            st.info("""
            SHAP TreeExplainer is not supported for KNN.
            Prediction is available, but the SHAP chart
            is not displayed for this model.
            """)

    except Exception as e:

        st.error(f"Prediction error: {e}")


# ==========================================
# PROJECT INFORMATION
# ==========================================

st.divider()

st.markdown("""
### About This Project

**Project:** Crop Yield Prediction Using Machine Learning

**Models:** Random Forest, Extra Trees, Gradient Boosting,
Decision Tree and KNN.

**Explainable AI:** SHAP-based feature explanations
for supported tree-based models.

**Prediction Unit:** Tonnes per hectare.
""")
