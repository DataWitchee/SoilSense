# frontend/app.py

# --- IMPORTANT FIX FOR IMPORT PATH ---
import sys
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)
# --------------------------------------

import streamlit as st
from streamlit_folium import st_folium
import folium

from frontend.api_client import get_soil, get_weather, predict


# --------------------------------------
# STREAMLIT UI SETUP
# --------------------------------------
st.set_page_config(
    page_title="SoilSense — Crop Recommendation",
    layout="wide"
)

st.title("🌱 SoilSense — Location-based Crop Recommender")
st.write("Click on the map to auto-detect soil nutrients, fetch weather, and get crop predictions.")


# --------------------------------------
# MAP SETUP
# --------------------------------------
m = folium.Map(location=[22.0, 79.0], zoom_start=5)

st.subheader("Select a Location on Map")
map_output = st_folium(m, width=950, height=600)

clicked = map_output.get("last_clicked") if map_output else None

col1, col2 = st.columns(2)


with col1:
    st.subheader("📍 Selected Coordinates")

    if clicked:
        lat = clicked["lat"]
        lon = clicked["lng"]
        st.success(f"Lat: {lat:.5f}, Lon: {lon:.5f}")
    else:
        lat = None
        lon = None
        st.info("Click anywhere on the map to select a location.")


with col2:
    st.subheader("🧪 Soil & Weather Auto-Fetch")

    if clicked:
        try:
            soil_res = get_soil(lat, lon)
            weather_res = get_weather(lat, lon)
            st.success("Fetched successfully!")
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            soil_res = None
            weather_res = None

        if soil_res:
            st.markdown("### Soil Nutrients")
            st.json(soil_res)
        if weather_res:
            st.markdown("### Weather Info")
            st.json(weather_res)



st.markdown("---")
st.subheader("🤖 Crop Recommendation")

# ------------------------------------------------
# PREDICTION FORM
# ------------------------------------------------
with st.form("predict_form"):
    user_id = st.number_input("User ID (optional)", step=1, min_value=0)

    top_k = st.slider("Number of recommendations", min_value=1, max_value=5, value=3)

    submit_btn = st.form_submit_button("Get Recommendations")

if submit_btn:
    if not clicked:
        st.warning("Please select a location on the map first.")
    else:
        try:
            res = predict(
                lat=lat,
                lon=lon,
                top_k=top_k,
                user_id=user_id
            )

            st.success("Prediction received!")

            st.markdown("### 🌾 Recommended Crops:")
            for i, crop in enumerate(res["recommended"], 1):
                st.write(f"{i}. {crop}")

            st.markdown("### 📥 Input Used:")
            st.json(res["input"])

        except Exception as e:
            st.error(f"Prediction failed: {e}")


st.markdown("---")
st.caption("SoilSense Frontend — Powered by Streamlit + FastAPI")
