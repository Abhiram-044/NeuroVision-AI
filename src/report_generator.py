from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from datetime import datetime
import uuid
import os


def severity_level(area):
    if area < 5:
        return "Mild"
    elif area < 20:
        return "Moderate"
    return "Severe"


def generate_report(prediction, confidence, tumor_area):

    os.makedirs("reports", exist_ok=True)

    report_id = str(uuid.uuid4())[:8]
    filename = f"reports/NeuroVision_Report_{report_id}.pdf"

    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    severity = severity_level(tumor_area)

    story.append(Paragraph("NEUROVISION AI MEDICAL REPORT", styles['Title']))
    story.append(Spacer(1, 20))

    story.append(Paragraph(f"Report ID: NV-{report_id}", styles['Normal']))
    story.append(Paragraph(
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        styles['Normal']
    ))

    story.append(Spacer(1, 20))

    story.append(Paragraph(f"Prediction: {prediction}", styles['Normal']))
    story.append(Paragraph(
        f"Confidence: {confidence:.2f}%",
        styles['Normal']
    ))
    story.append(Paragraph(
        f"Tumor Coverage: {tumor_area:.2f}%",
        styles['Normal']
    ))
    story.append(Paragraph(
        f"Severity Level: {severity}",
        styles['Normal']
    ))

    story.append(Spacer(1, 20))

    observation = (
        "Localized abnormal tissue activation detected "
        "based on Grad-CAM explainability analysis."
    )

    story.append(Paragraph(f"AI Observation: {observation}",
                           styles['Normal']))

    story.append(Spacer(1, 40))

    disclaimer = (
        "Disclaimer: This report is AI-assisted and "
        "should not replace professional medical diagnosis."
    )

    story.append(Paragraph(disclaimer, styles['Italic']))

    doc.build(story)

    return filename