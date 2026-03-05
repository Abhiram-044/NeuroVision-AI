# NeuroVision — Multi-Class Brain Tumor Detection & Clinical AI System

> **Deep Learning powered MRI Brain Tumor Classification with Explainable AI, Tumor Localization & Automated Clinical Reporting**

NeuroVision is an **end-to-end Medical AI system** that analyzes **brain MRI scans** and automatically detects and localizes different tumor types using **Deep Learning + Explainable AI (Grad-CAM)**.

Unlike basic binary classifiers, NeuroVision performs **multi-class tumor diagnosis** similar to real radiology workflows.

---

## Supported Tumor Classes

The model classifies MRI scans into **four clinically relevant categories**:

| Class                 | Description                            |
| --------------------- | -------------------------------------- |
|  **Glioma**           | Tumor originating in brain glial cells |
|  **Meningioma**       | Tumor arising from meninges layers     |
|  **Pituitary Tumor**  | Tumor near pituitary gland             |
|  **No Tumor**         | Healthy brain MRI                      |

---

## Key Features

* Multi-Class Brain Tumor Classification
* Automatic Tumor Localization (Grad-CAM Bounding Box)
* Tumor Area Percentage Estimation
* Explainable AI Visualization
* Clinical Severity Estimation
* Automated Medical Report Generator
* Interactive NeuroVision Dashboard

---

## AI Pipeline

```
MRI Scan
   ↓
CNN / Transfer Learning Model (PyTorch)
   ↓
Tumor Type Prediction
   ↓
Grad-CAM Heatmap
   ↓
Tumor Localization
   ↓
Tumor Area Estimation
   ↓
Clinical Report Generation
   ↓
Streamlit Dashboard
```

---

## Project Structure

```
NeuroVision/
│
├── dataset/
│   ├── train/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── pituitary/
│   │   └── notumor/
│   │
│   └── test/
│       ├── glioma/
│       ├── meningioma/
│       ├── pituitary/
│       └── notumor/
│
├── models/
│   └── resnet50_brain_tumor.pth        # Trained multi-class model
│
├── src/
│   ├── dataset.py             # Dataset loader
│   ├── model.py               # CNN architecture
│   ├── train.py               # Training pipeline
│   ├── evaluate.py            # Accuracy & metrics
│   ├── predict.py             # Inference logic
│   ├── gradcam.py             # Explainable AI
│   ├── localization.py        # Bounding box extraction
│   ├── tumor_area.py          # Tumor % estimation
│   └── report_generator.py    # Medical report creator
│
├── app.py                     # NeuroVision Dashboard
├── requirements.txt
└── README.md
```

---

## Training the Model

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Dataset Format

```
dataset/
   train/
      glioma/
      meningioma/
      pituitary/
      notumor/
```

---

### Train Model

```bash
python src/train.py
```

Output:

```
models/resnet50_brain_tumor.pth
```

Saved model includes:

* Learned tumor features
* Multi-class classifier
* Clinical prediction capability

---

## Model Output Example

```
Prediction: Glioma Tumor
Confidence: 96.8%

Tumor Localization: Detected
Tumor Coverage: 21.4%

Severity Level: Moderate
```

---

## Explainable AI — Grad-CAM

NeuroVision generates:

* Activation heatmaps
* Tumor focus regions
* Automatic bounding box
* Visual medical explanation

This improves **trustworthiness of AI diagnosis**.

---

## Tumor Area Percentage Estimation

Tumor severity is estimated using:

```
Tumor Area (%) =
Tumor Region Pixels / Brain Pixels × 100
```

Severity Mapping:

| Tumor Area | Severity |
| ---------- | -------- |
| < 5%       | Mild     |
| 5–20%      | Moderate |
| > 20%      | Severe   |

---

## Automated Medical Report

The system generates structured clinical summaries:

Example:

```
AI Radiology Report
-------------------
Detected Condition: Pituitary Tumor
Confidence Score: 97.2%
Tumor Coverage: 18.3%
Severity: Moderate

Observation:
Localized abnormal tissue detected
near pituitary region.
```

---

## Run NeuroVision Clinical Dashboard

Launch application:

```bash
streamlit run app.py
```

Open browser:

```
http://localhost:8501
```

Dashboard allows:

* MRI Upload
* Tumor Prediction
* Heatmap Visualization
* Localization Box
* Clinical Report Download

---

## Deployment

NeuroVision supports public deployment via:

* Hugging Face Spaces
* Streamlit Cloud
* Docker
* AWS EC2

---

## Tech Stack

* PyTorch
* OpenCV
* Grad-CAM
* Streamlit
* NumPy
* Scikit-learn
* Matplotlib

---

## Disclaimer

This system is intended for **research and educational purposes only** and should not replace professional medical diagnosis.

---

## Author

**Abhiram Chinta**
AI / ML Engineer | Deep Learning Enthusiast

---

**NeuroVision bridges Deep Learning and Clinical Intelligence.**
