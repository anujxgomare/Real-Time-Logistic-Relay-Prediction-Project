import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("xgb_classifier_model.pkl")
scaler = joblib.load("xgb_scaler.pkl")

st.set_page_config(page_title="Delivery Delay Predictor", layout="centered")
st.title("🚚 Logistics Delivery Delay Predictor")

st.markdown("""
Enter the shipment details below to predict whether the delivery will be delayed.
""")

# 🎯 Input form
with st.form("input_form"):
    distance_km = st.number_input("Distance (in km)", min_value=1.0, max_value=2000.0, value=500.0)
    vendor_delay_score = st.slider("Vendor Delay Score", 0.0, 1.0, 0.5)
    hour_of_day = st.slider("Pickup Hour (0 to 23)", 0, 23, 10)
    day_of_week = st.selectbox("Day of Week", options=[0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
    pickup_delay_minutes = st.slider("Pickup Delay (minutes)", 0, 180, 15)
    driver_rating = st.slider("Driver Rating", 0.0, 5.0, 4.5)
    vehicle_age_years = st.slider("Vehicle Age (years)", 0, 20, 4)
    order_weight_kg = st.number_input("Order Weight (kg)", 0.1, 1000.0, 25.0)
    num_packages = st.number_input("Number of Packages", 1, 100, 5)
    holiday_flag = st.selectbox("Is it a holiday?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    submit = st.form_submit_button("Predict Delay")

# 💡 Process input and predict
if submit:
    # Combine inputs into a feature array
    input_features = np.array([[distance_km, vendor_delay_score, hour_of_day, day_of_week,
                                pickup_delay_minutes, driver_rating, vehicle_age_years,
                                order_weight_kg, num_packages, holiday_flag]])

    # Scale input features
    input_scaled = scaler.transform(input_features)

    # Predict with threshold = 0.3
    probability = model.predict_proba(input_scaled)[0][1]
    prediction = probability > 0.3

    st.markdown("### 📊 Prediction Result")

    if prediction:
        st.error(f"⚠️ Delay Likely!\n\nProbability of delay: **{probability*100:.2f}%**")
    else:
        st.success(f"✅ On-Time Delivery\n\nProbability of delay: **{probability*100:.2f}%**")
