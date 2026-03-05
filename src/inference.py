import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.model import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ["glioma", "meningioma", "notumor", "pituitary"]

model = build_model(num_classes=4, freeze_backbone=False)
model.load_state_dict(
    torch.load("weights/resnet50_brain_tumor.pth",
               map_location=device)
)

model.to(device)
model.eval()

target_layer = model.layer4[-1]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def predict(image):
    pil_img = Image.open(image).convert("RGB")
    rgb_img = np.array(pil_img.resize((224,224))) / 255.0

    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        
    confidence = probs[0][torch.argmax(probs)].item() * 100
    pred = torch.argmax(probs, 1).item()

    cam = GradCAM(model=model,
                  target_layers=[target_layer])

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    heatmap = (grayscale_cam*255).astype(np.uint8)

    _, thresh = cv2.threshold(heatmap,150,255,
                              cv2.THRESH_BINARY)

    tumor_pixels = np.sum(thresh>0)
    total_pixels = thresh.size

    tumor_percent = (tumor_pixels/total_pixels)*100

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    return {
        "prediction": classes[pred],
        "confidence": confidence,
        "tumor_percent": tumor_percent,
        "image": visualization
    }