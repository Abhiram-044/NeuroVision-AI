import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import build_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model(num_classes=4, freeze_backbone=False)
model.load_state_dict(torch.load("resnet50_brain_tumor.pth"))
model.to(device)
model.eval()

target_layer = model.layer4[-1]

transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

def generate_gradcam(image_path):

    image = Image.open(image_path).convert("RGB")
    rgb_img = np.array(image.resize((224,224))) / 255.0

    input_tensor = transform(image).unsqueeze(0).to(device)
    # flipped_tensor = torch.flip(input_tensor, dims=[3])

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        # flipped_output = model(flipped_tensor)
        # flipped_probs = torch.softmax(flipped_output, dim=1)

    print("Class Probabilities: ", probs.cpu().numpy())
    pred_class = torch.argmax(probs, dim=1).item()
    print("Predicted Class: ", pred_class)

    # print("Flipped Prediction: ", flipped_probs.cpu().numpy())

    cam = GradCAM(model=model,
                  target_layers=[target_layer])

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    bbox_img = draw_tumor_bbox(rgb_img, grayscale_cam)

    tumor_percent, mask = calculate_tumor_area(grayscale_cam)
    if tumor_percent < 1:
        tumor_percent = 0
    status = "No Significant Tumor Detected" if tumor_percent < 1 else "Tumor Detected"
    print(status)
    print(f"Tumor Area: {tumor_percent:.2f}%")

    # flipped_rgb = np.flip(rgb_img, axis=1)

    # flipped_cam = cam(input_tensor=flipped_tensor)[0]

    # flipped_vis = show_cam_on_image(
    #     flipped_rgb,
    #     flipped_cam,
    #     use_rgb=True
    # )
    cv2.putText(
        bbox_img,
        f"Tumor Area: {tumor_percent:.2f}%",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("GradCAM", visualization)
    cv2.imshow("Tumor Localization", bbox_img)
    # cv2.imshow("Flipped GradCAM", flipped_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_tumor_bbox(rgb_img, cam):
    heatmap = (cam * 255).astype(np.uint8)

    _, thresh = cv2.threshold(
        heatmap,
        150,
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    img = (rgb_img * 255).astype(np.uint8).copy()

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(c)

        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    return img

def calculate_tumor_area(cam):
    heatmap = (cam * 255).astype(np.uint8)

    _, thresh = cv2.threshold(
        heatmap,
        150,
        255,
        cv2.THRESH_BINARY
    )

    tumor_pixels = np.sum(thresh > 0)

    total_pixels = thresh.shape[0] * thresh.shape[1]

    tumor_percentage = (tumor_pixels / total_pixels) * 100

    return tumor_percentage, thresh

if __name__ == "__main__":
    generate_gradcam(r"sample.jpg") # put location of image to test 