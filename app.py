import streamlit as st
import joblib
import pandas as pd
import json

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Araç Fiyat Tahmini",
    page_icon="🚗",
    layout="centered"
)

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------
models = {
    "Linear Regression": joblib.load("models/linear_regression.pkl"),
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "XGBoost": joblib.load("models/xgboost.pkl")
}

# -------------------------------------------------
# LOAD METRICS
# -------------------------------------------------
with open("metrics/model_metrics.json", "r") as f:
    model_metrics = json.load(f)

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.title("🚗 Araç Fiyat Tahmin Sistemi")
st.caption("Emirhan Özger / Y250234091")

# -------------------------------------------------
# INPUT SECTION
# -------------------------------------------------
st.subheader("🔧 Araç Bilgileri")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Model Yılı", 1990, 2025, 2018)
    km = st.number_input("Kilometre", min_value=0, value=80000)
    fuel_type = st.selectbox("Yakıt Türü", ["Benzin", "Dizel", "Hybrid", "Elektrik"])

with col2:
    engine_size = st.number_input("Motor Hacmi (L)", 0.8, 6.0, 1.6)
    hp = st.number_input("Beygir Gücü", 50, 600, 120)
    gear_type = st.selectbox("Vites", ["Manuel", "Otomatik"])

# -------------------------------------------------
# PREDICT BUTTON
# -------------------------------------------------
st.markdown("---")
predict_btn = st.button("Tahmin Et", use_container_width=True)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
if predict_btn:

    car_age = 2025 - year
    mileage_per_year = km / max(1, car_age)

    input_df = pd.DataFrame([{
        "milage": km,
        "engine displacement": engine_size,
        "hp": hp,
        "brand": "Ford",  # eğitimde olan sabit değer
        "fuel_type": fuel_type,
        "transmission_clean": gear_type,
        "exterior_color": "Black",
        "interior_color": "Black",
        "clean_title": 1,
        "turbo": 0,
        "dohc": 1,
        "is_v_engine": 0,
        "Accident_Impact": 0,
        "car_age": car_age,
        "Mileage_per_Year": mileage_per_year
    }])

    results = []

    for model_name, model in models.items():
        prediction = model.predict(input_df)[0]

        results.append({
            "Model": model_name,
            "Tahmin ($)": f"{prediction:,.0f}".replace(",", "."),
            "R²": round(model_metrics[model_name]["R2"], 3),
            "MAE": f"±{model_metrics[model_name]['MAE']:,.0f}".replace(",", ".")
        })

    results_df = pd.DataFrame(results)

    # -------------------------------------------------
    # BEST MODEL
    # -------------------------------------------------
    best_model = max(results, key=lambda x: x["R²"])

    st.success(
        f"🏆 **En İyi Model:** {best_model['Model']}\n\n"
        f"💰 **Tahmini Fiyat:** {best_model['Tahmin ($)']} $"
    )

    # -------------------------------------------------
    # RESULTS TABLE
    # -------------------------------------------------
    st.subheader("📊 Model Karşılaştırması")

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )
