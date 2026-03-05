import torch
from model import build_model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=4)
    model = model.to(device)

    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy_input)

    print("Output Shape: ", output.shape)
