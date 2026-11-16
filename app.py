import gradio as gr
import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# ==========================================
# LOAD TRAINED MODEL PACKAGE
# ==========================================

with open("crop_yield_models.pkl", "rb") as file:
    package = pickle.load(file)

models = package["models"]
best_model = package["best_model"]
best_model_name = package["best_model_name"]

FEATURE_COLUMNS = package["feature_columns"]
NUMERIC_FEATURES = package["numeric_features"]
CATEGORIES = package["categories"]

# ==========================================
# CREATE INPUT DATA
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

    # Create original numerical features
    row = dict(values)

    # Create one-hot categorical features
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

    # Preserve exact model feature order
    input_df = pd.DataFrame(
        [[row.get(col, 0) for col in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS
    )

    return input_df


# ==========================================
# PREDICTION AND XAI
# ==========================================

def predict_yield(
    model_name,
    district, crop, season,
    rainfall, temperature, humidity, soil_ph,
    nitrogen, phosphorus, potassium,
    irrigation, fertilizer, area
):

    try:
        input_df = prepare_input(
            district, crop, season,
            rainfall, temperature, humidity, soil_ph,
            nitrogen, phosphorus, potassium,
            irrigation, fertilizer, area
        )

        selected_model = models[model_name]

        prediction = float(selected_model.predict(input_df)[0])

        # SHAP explanation
        fig = None
        explanation_text = ""

        if model_name != "KNN":
            try:
                explainer = shap.TreeExplainer(selected_model)
                shap_values = explainer.shap_values(input_df)

                if isinstance(shap_values, list):
                    shap_values = shap_values[0]

                contributions = np.asarray(shap_values).reshape(-1)

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

                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_title("SHAP Feature Impact")
                ax.set_xlabel("Impact on predicted yield")
                ax.set_ylabel("Input features")

                plt.tight_layout()

                explanation_text = (
                    "SHAP explains how each feature contributes "
                    "to this model's prediction. Positive values "
                    "increase the prediction relative to its "
                    "baseline; negative values decrease it."
                )

            except Exception as e:
                explanation_text = (
                    f"SHAP explanation unavailable: {str(e)}"
                )

        else:
            explanation_text = (
                "SHAP TreeExplainer is not supported for KNN. "
                "Prediction is available, but this model's "
                "SHAP chart is not displayed."
            )

        result = (
            f"### Predicted Crop Yield\n"
            f"# {prediction:.3f} tonnes/hectare\n\n"
            f"**Selected Model:** {model_name}\n\n"
            f"**Default Best Model:** {best_model_name}"
        )

        return result, fig, explanation_text

    except Exception as e:
        return f"Prediction error: {str(e)}", None, "Please check your inputs."


# ==========================================
# GRADIO INTERFACE
# ==========================================

with gr.Blocks(
    title="Crop Yield Prediction System",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown("""
    # 🌾 Crop Yield Prediction System

    ### Scalable Big Data Analytics for Crop Yield Prediction
    Using Multi-Source Agricultural Data

    Enter agricultural, soil and weather information
    to estimate crop yield using Machine Learning.
    """)

    with gr.Row():

        with gr.Column():
            gr.Markdown("### Agricultural Information")

            district = gr.Dropdown(
                choices=CATEGORIES["district"],
                value=CATEGORIES["district"][0],
                label="District"
            )

            crop = gr.Dropdown(
                choices=CATEGORIES["crop"],
                value=CATEGORIES["crop"][0],
                label="Crop"
            )

            season = gr.Dropdown(
                choices=CATEGORIES["season"],
                value=CATEGORIES["season"][0],
                label="Season"
            )

            model_name = gr.Dropdown(
                choices=list(models.keys()),
                value=best_model_name,
                label="Machine Learning Model"
            )

        with gr.Column():
            gr.Markdown("### Weather and Soil Information")

            rainfall = gr.Number(
                value=720, label="Rainfall (mm)"
            )

            temperature = gr.Number(
                value=28, label="Average Temperature (°C)"
            )

            humidity = gr.Number(
                value=65, label="Humidity (%)"
            )

            soil_ph = gr.Number(
                value=6.7, label="Soil pH"
            )

    gr.Markdown("### Fertilizer and Farm Information")

    with gr.Row():
        nitrogen = gr.Number(
            value=110, label="Nitrogen (kg/ha)"
        )

        phosphorus = gr.Number(
            value=55, label="Phosphorus (kg/ha)"
        )

        potassium = gr.Number(
            value=66, label="Potassium (kg/ha)"
        )

    with gr.Row():
        irrigation = gr.Number(
            value=250, label="Irrigation (mm)"
        )

        fertilizer = gr.Number(
            value=180, label="Fertilizer (kg/ha)"
        )

        area = gr.Number(
            value=3, label="Farm Area (ha)"
        )

    predict_button = gr.Button(
        "Predict Crop Yield",
        variant="primary"
    )

    prediction_output = gr.Markdown()

    gr.Markdown("## Explainable AI (XAI)")

    shap_plot = gr.Plot(
        label="SHAP Feature Importance"
    )

    explanation_output = gr.Markdown()

    predict_button.click(
        fn=predict_yield,
        inputs=[
            model_name,
            district, crop, season,
            rainfall, temperature, humidity, soil_ph,
            nitrogen, phosphorus, potassium,
            irrigation, fertilizer, area
        ],
        outputs=[
            prediction_output,
            shap_plot,
            explanation_output
        ]
    )

    gr.Markdown("""
    ---
    **Project:** Crop Yield Prediction Using Machine Learning

    **Models:** Random Forest, Extra Trees, Gradient Boosting,
    Decision Tree and KNN.

    **XAI:** SHAP-based feature explanations for supported
    tree-based models.

    **Prediction unit:** Tonnes per hectare.
    """)


if __name__ == "__main__":
    demo.launch()
