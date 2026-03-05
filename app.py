import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import streamlit as st
from src.inference import predict
from src.report_generator import generate_report
import cv2

st.set_page_config(
    page_title="NeuroVision AI",
    layout="wide"
)

st.title("NeuroVision AI - Brain Tumor Analysis")

uploaded_file = st.file_uploader(
    "Upload Brain MRI",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    st.image(uploaded_file,
             caption="Uploaded MRI",
             width=300)
    
    with st.spinner("Analyzing MRI..."):
        result = predict(uploaded_file)

    prediction = result["prediction"]
    confidence = result["confidence"]
    tumor_percent = result["tumor_percent"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Diagnosis")
        st.success(result["prediction"])
        
        tumor_percent = result["tumor_percent"]

        if tumor_percent < 5:
            severity = "Mild"
        elif tumor_percent < 20:
            severity = "Moderate"
        else:
            severity = "Severe"

        st.metric("Tumor Coverage %",
                  f"{tumor_percent:.2f}%")
        
        st.warning(f"Severity: {severity}")

    with col2:
        st.subheader("Tumor Localization")
        st.image(result["image"])

    report_path = generate_report(
        prediction,
        confidence,
        tumor_percent
    )

    with open(report_path, "rb") as file:
        st.download_button(
            label="📄 Download Medical Report",
            data=file,
            file_name="NeuroVision_Report.pdf",
            mime="application/pdf"
        )