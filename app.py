import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go

model = pickle.load(open("best_model.pkl", "rb"))
cats_info = json.load(open("cats_info.json", "r", encoding="utf-8"))

unique_values = cats_info["unique_values"]
brand_to_models = cats_info["brand_to_models"]

CATEGORICAL_COLS = [
    "Brand", "Model", "UsedOrNew",
    "DriveType", "FuelType", "ColourExtInt",
    "BodyType", "Transmission"
]

df_full = pd.read_csv("vehicleprice.csv")

df_full["UsedOrNew"] = (
    df_full["UsedOrNew"]
    .astype(str)
    .str.strip()
    .str.upper()
)

df_full = df_full[df_full["UsedOrNew"] != "DEMO"]

df_full["Price_clean"] = pd.to_numeric(
    df_full["Price"].astype(str).str.replace(r"[$, ]", "", regex=True),
    errors="coerce"
)

def get_model_specific_values(df, brand, model_name):
    subset = df[
        (df["Brand"] == brand) &
        (df["Model"] == model_name)
    ]

    if subset.empty:
        return None

    def extract(col):
        return sorted(
            subset[col]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

    return {
        "FuelType": extract("FuelType"),
        "DriveType": extract("DriveType"),
        "BodyType": extract("BodyType"),
        "CylindersinEngine": extract("CylindersinEngine"),
        "Doors": extract("Doors"),
        "Seats": extract("Seats"),
        "Engine": extract("Engine"),
    }

def prepare_input(input_dict):
    df = pd.DataFrame([input_dict])

    df["FuelConsumption_new"] = (
        df["FuelConsumption"]
        .astype(str)
        .str.extract(r"(\d+\.?\d*)")
        .astype(float)
    )

    df["Engine_new"] = (
        df["Engine"]
        .astype(str)
        .str.extract(r"(\d+\.?\d*)")
        .astype(float)
    )

    df["Cylinders_new"] = (
        df["CylindersinEngine"]
        .astype(str)
        .str.extract(r"(\d+)")
        .astype(float)
    )

    df["Doors_new"] = (
        df["Doors"]
        .astype(str)
        .str.extract(r"(\d+)")
        .astype(float)
    )

    df["Seats_new"] = (
        df["Seats"]
        .astype(str)
        .str.extract(r"(\d+)")
        .astype(float)
    )

    df["Car_Age"] = 2025 - df["Year"]
    df["KM_per_Year"] = df["Kilometres"] / (df["Car_Age"] + 1)
    df["log_KM"] = np.log1p(df["Kilometres"])
    df["log_Engine"] = np.log1p(df["Engine_new"])
    df["log_FuelCons"] = np.log1p(df["FuelConsumption_new"])

    df["ColourExtInt"] = "Unknown"

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    df.drop(
        ["FuelConsumption", "Engine", "CylindersinEngine", "Doors", "Seats"],
        axis=1,
        inplace=True
    )

    df = df.reindex(columns=model.feature_name())

    return df

def predict_price(input_dict):
    X = prepare_input(input_dict)
    return np.expm1(model.predict(X)[0])

st.set_page_config(
    page_title="Araç Fiyat Tahmini",
    layout="wide"
)

st.markdown("<h1>Araç Fiyat Tahmini</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([1.3, 1.7])

with col1:
    brand = st.selectbox("Marka", unique_values["Brand"])
    model_name = st.selectbox("Model", brand_to_models.get(brand, []))

    model_specific = get_model_specific_values(df_full, brand, model_name)

    if not model_specific:
        st.error("Bu marka–model için yeterli veri yok.")
        st.stop()

    year = st.number_input("Yıl", 1990, 2025, 2020)
    km = st.number_input("Kilometre", 0, 500000, 50000)
    used_new = st.selectbox("Durum", unique_values["UsedOrNew"])
    transmission = st.selectbox("Vites", unique_values["Transmission"])

    fuel_con = st.number_input("Yakıt (L/100km)", 0.0, 30.0, 8.0)

    fuel_type = st.selectbox("Yakıt Türü", model_specific["FuelType"])
    drive_type = st.selectbox("Çekiş", model_specific["DriveType"])
    body_type = st.selectbox("Kasa Tipi", model_specific["BodyType"])
    engine = st.selectbox("Motor", model_specific["Engine"])
    cylinders = st.selectbox("Silindir", model_specific["CylindersinEngine"])
    doors = st.selectbox("Kapı", model_specific["Doors"])
    seats = st.selectbox("Koltuk", model_specific["Seats"])

    predict_btn = st.button("Tahmin Yap")

with col2:
    if predict_btn:
        input_data = {
            "Brand": brand,
            "Model": model_name,
            "Year": year,
            "UsedOrNew": used_new,
            "Transmission": transmission,
            "DriveType": drive_type,
            "FuelType": fuel_type,
            "FuelConsumption": fuel_con,
            "Kilometres": km,
            "Engine": engine,
            "CylindersinEngine": cylinders,
            "BodyType": body_type,
            "Doors": doors,
            "Seats": seats,
        }

        price = predict_price(input_data)

        st.subheader(f"Tahmini Fiyat: **{round(price, 2)} $**")

        gauge_max = 650_000 if price <= 650_000 else price

        st.plotly_chart(
            go.Figure(go.Indicator(
                mode="gauge+number",
                value=price,
                number={"valueformat": ",.0f", "suffix": " $"},
                gauge={"axis": {"range": [0, gauge_max], "tickformat": ",.0f"}}
            )),
            use_container_width=True
        )
