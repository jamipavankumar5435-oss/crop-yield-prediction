
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)

@st.cache_resource
def load_models():
    with open("crop_yield_models.pkl", "rb") as f:
        return pickle.load(f)

package = load_models()

models = package["models"]
best_model = package["best_model"]
best_model_name = package["best_model_name"]
feature_columns = package["feature_columns"]
categories = package["categories"]

numeric_features = package["numeric_features"]

st.title("🌾 Crop Yield Prediction System")
st.write(
    "Predict agricultural crop yield using multiple "
    "machine learning models and explainable AI."
)

st.info("Enter your farm details below to predict crop yield.")

with st.form("prediction_form"):

    st.subheader("Farm Information")

    col1, col2 = st.columns(2)

    with col1:
        district = st.selectbox(
            "Select District",
            categories["district"]
        )

        crop = st.selectbox(
            "Select Crop",
            categories["crop"]
        )

        season = st.selectbox(
            "Select Season",
            categories["season"]
        )

        rainfall_mm = st.number_input(
            "Rainfall (mm)", min_value=0.0, value=500.0
        )

        avg_temp_c = st.number_input(
            "Average Temperature (°C)", value=25.0
        )

        humidity_pct = st.number_input(
            "Humidity (%)", min_value=0.0,
            max_value=100.0, value=60.0
        )

        soil_ph = st.number_input(
            "Soil pH", min_value=0.0,
            max_value=14.0, value=6.5
        )

    with col2:
        nitrogen_kg_ha = st.number_input(
            "Nitrogen (kg/ha)", min_value=0.0, value=50.0
        )

        phosphorus_kg_ha = st.number_input(
            "Phosphorus (kg/ha)", min_value=0.0, value=40.0
        )

        potassium_kg_ha = st.number_input(
            "Potassium (kg/ha)", min_value=0.0, value=40.0
        )

        irrigation_mm = st.number_input(
            "Irrigation (mm)", min_value=0.0, value=100.0
        )

        fertilizer_kg_ha = st.number_input(
            "Fertilizer (kg/ha)", min_value=0.0, value=100.0
        )

        area_ha = st.number_input(
            "Farm Area (hectares)", min_value=0.01, value=1.0
        )

    selected_model_name = st.selectbox(
        "Choose Machine Learning Model",
        list(models.keys()),
        index=(
            list(models.keys()).index(best_model_name)
            if best_model_name in models else 0
        )
    )

    predict_button = st.form_submit_button(
        "Predict Crop Yield 🌱",
        use_container_width=True
    )

if predict_button:

    input_values = {
        "rainfall_mm": rainfall_mm,
        "avg_temp_c": avg_temp_c,
        "humidity_pct": humidity_pct,
        "soil_ph": soil_ph,
        "nitrogen_kg_ha": nitrogen_kg_ha,
        "phosphorus_kg_ha": phosphorus_kg_ha,
        "potassium_kg_ha": potassium_kg_ha,
        "irrigation_mm": irrigation_mm,
        "fertilizer_kg_ha": fertilizer_kg_ha,
        "area_ha": area_ha
    }

    for category_name, selected_value in [
        ("district", district),
        ("crop", crop),
        ("season", season)
    ]:
        for category in categories[category_name]:
            input_values[
                f"{category_name}_{category}"
            ] = int(category == selected_value)

    input_df = pd.DataFrame(
        [input_values]
    ).reindex(columns=feature_columns, fill_value=0)

    selected_model = models[selected_model_name]

    try:
        prediction = float(selected_model.predict(input_df)[0])

        st.success("Prediction completed successfully!")

        st.subheader("Crop Yield Prediction")

        st.metric(
            "Predicted Yield",
            f"{prediction:.3f} tonnes/ha"
        )

        st.write("**Selected model:**", selected_model_name)

        if selected_model_name == best_model_name:
            st.caption("This is the best-performing model from your evaluation.")

        st.divider()

        st.subheader("Explainable AI - SHAP Analysis")

        if selected_model_name == "KNN":
            st.warning(
                "TreeExplainer is not supported for KNN. "
                "SHAP chart is not displayed for this model."
            )

        else:
            try:
                explainer = shap.TreeExplainer(selected_model)
                shap_values = explainer.shap_values(input_df)

                values = np.asarray(shap_values)

                if values.ndim == 3:
                    values = values[0, :, 0]
                elif values.ndim == 2:
                    values = values[0]

                importance = pd.DataFrame({
                    "Feature": feature_columns,
                    "SHAP Impact": values
                })

                importance["Absolute Impact"] = (
                    importance["SHAP Impact"].abs()
                )

                importance = importance.nlargest(
                    10, "Absolute Impact"
                ).sort_values("Absolute Impact")

                fig, ax = plt.subplots(figsize=(9, 5))

                ax.barh(
                    importance["Feature"],
                    importance["SHAP Impact"]
                )

                ax.set_xlabel("SHAP value")
                ax.set_title("Top 10 Feature Contributions")

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.write(
                    "Features with larger absolute SHAP values "
                    "have a greater influence on this prediction."
                )

            except Exception as e:
                st.warning(
                    "SHAP explanation could not be generated: "
                    + str(e)
                )

    except Exception as e:
        st.error(f"Prediction error: {e}")

st.divider()

st.subheader("About This Project")

st.write("""
This project uses multi-source agricultural data to predict crop yield.
It compares five machine learning models and uses SHAP to explain
feature contributions for supported tree-based models.
""")

st.caption("Crop Yield Prediction | Machine Learning + Explainable AI")
