import streamlit as st
import pandas as pd
import pickle

# Load model
with open("hotel_model.pkl", "rb") as f:
    model = pickle.load(f)

# Page Config
st.set_page_config(
    page_title="Hotel Booking Cancellation Prediction",
    page_icon="🏨",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>

.main {
    background-color: #f4f7fc;
}

.title {
    text-align: center;
    color: #1E3A8A;
    font-size: 40px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #555;
    font-size: 18px;
}

.stButton>button {
    background: linear-gradient(to right,#2563EB,#1E40AF);
    color:white;
    border-radius:10px;
    height:55px;
    width:100%;
    font-size:20px;
    font-weight:bold;
}

.stButton>button:hover {
    background: linear-gradient(to right,#1E40AF,#1D4ED8);
}

.result-success{
    background-color:#DCFCE7;
    padding:20px;
    border-radius:10px;
    color:#166534;
    font-size:22px;
    font-weight:bold;
}

.result-danger{
    background-color:#FEE2E2;
    padding:20px;
    border-radius:10px;
    color:#991B1B;
    font-size:22px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    "<div class='title'>🏨 Hotel Booking Cancellation Prediction</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Predict whether a hotel booking will be cancelled or confirmed</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# Sidebar
st.sidebar.title("About Project")
st.sidebar.info("""
Hotel Booking Cancellation Prediction System

Model Used:
✅ XGBoost Classifier

Dataset:
Hotel Booking Dataset

Developed using:
- Python
- Streamlit
- XGBoost
- Pandas
""")

# Inputs
col1, col2 = st.columns(2)

with col1:
    lead_time = st.number_input("Lead Time", min_value=0, value=50)
    weekend = st.number_input("Weekend Nights", min_value=0, value=1)
    week = st.number_input("Week Nights", min_value=0, value=2)
    adults = st.number_input("Adults", min_value=1, value=2)

with col2:
    children = st.number_input("Children", min_value=0, value=0)
    babies = st.number_input("Babies", min_value=0, value=0)
    prev_cancel = st.number_input("Previous Cancellations", min_value=0, value=0)
    adr = st.number_input("Average Daily Rate (ADR)", min_value=0.0, value=100.0)

st.markdown("###")

# Prediction
if st.button("🔍 Predict Booking Status"):

    data = pd.DataFrame({
        "lead_time":[lead_time],
        "stays_in_weekend_nights":[weekend],
        "stays_in_week_nights":[week],
        "adults":[adults],
        "children":[children],
        "babies":[babies],
        "previous_cancellations":[prev_cancel],
        "adr":[adr]
    })

    prediction = model.predict(data)[0]

    confidence = max(model.predict_proba(data)[0]) * 100

    st.markdown("### Prediction Result")

    if prediction == 1:

        st.markdown(
            f"""
            <div class='result-danger'>
            ❌ Booking Likely To Be Cancelled
            <br><br>
            Confidence: {confidence:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class='result-success'>
            ✅ Booking Likely To Be Confirmed
            <br><br>
            Confidence: {confidence:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("---")
st.caption("© Hotel Booking Cancellation Prediction System")