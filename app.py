import streamlit as st
import joblib
import pandas as pd

model = joblib.load("car_price_model.pkl")
mae = joblib.load("model_mae.pkl")

st.title("🚗 Araç Fiyat Tahmin Sistemi")

st.write("Araç bilgilerini giriniz:")

year = st.number_input("Model Yılı", min_value=1990, max_value=2025, value=2018)
km = st.number_input("Kilometre", min_value=0, value=80000)
engine_size = st.number_input("Motor Hacmi (L)", min_value=0.8, max_value=6.0, value=1.6)
hp = st.number_input("Beygir Gücü", min_value=50, max_value=600, value=120)

fuel_type = st.selectbox("Yakıt Türü", ["Benzin", "Dizel", "Hybrid", "Elektrik"])
gear_type = st.selectbox("Vites", ["Manuel", "Otomatik"])


if st.button("Tahmin Et"):
    input_df = pd.DataFrame([{
        "year": year,
        "milage": km,
        "engine displacement": engine_size,
        "hp": hp,

        "brand": "Ford",
        "fuel_type": fuel_type,
        "transmission_clean": gear_type,

        "exterior_color": "Black",
        "interior_color": "Black",

        "clean_title": 1,
        "turbo": 0,
        "dohc": 1,
        "is_v_engine": 0,
        "Accident_Impact": 0,

        "car_age": 2025 - year,
        "Mileage_per_Year": km / max(1, (2025 - year))
    }])

    prediction = model.predict(input_df)[0]

    formatted_price = f"{prediction:,.0f}".replace(",", ".")
    st.success(f"💰 Tahmini Araç Fiyatı: {formatted_price} $")

    print(f"Tahmini Fiyat: {int(prediction)}")
    print(f"Olası Hata: ±{int(mae)}")

