"""Predict image classes using the saved SVM and MobileNetV2 models."""

import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

from feature_extraction import extract_image_features


CLASS_NAMES = ["blackspot", "canker", "fresh", "grenning"]

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_svm(model_path):
    """Load the saved scikit-learn SVM pipeline."""
    return joblib.load(model_path)


def load_mobilenet(model_path, device):
    """Reconstruct MobileNetV2 and load the saved checkpoint."""
    model = models.mobilenet_v2(weights=None)

    input_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(
        input_features, len(CLASS_NAMES)
    )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    # The Colab checkpoint stores the model state under state_dict.
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


def predict_svm(image_path, model):
    """Predict the class using the classical feature pipeline."""
    features = extract_image_features(image_path).reshape(1, -1)
    prediction = model.predict(features)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba(features)))

    return prediction, confidence


def predict_mobilenet(image_path, model, device):
    """Predict the class using MobileNetV2."""
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    index = int(torch.argmax(probabilities).item())

    return (
        CLASS_NAMES[index],
        float(probabilities[index].item()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--image",
        required=True,
        help="Path to an image to classify.",
    )
    parser.add_argument(
        "--svm-model",
        default="models/classical_svm.joblib",
        help="Path to the saved SVM model.",
    )
    parser.add_argument(
        "--mobilenet-model",
        default="models/mobilenet_v2_best.pth",
        help="Path to the saved MobileNetV2 checkpoint.",
    )
    parser.add_argument(
        "--model",
        choices=["svm", "mobilenet", "both"],
        default="both",
        help="Which model to use.",
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        parser.error(f"Image not found: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Image: {image_path}")

    if args.model in ("svm", "both"):
        svm = load_svm(args.svm_model)
        label, confidence = predict_svm(image_path, svm)

        print(f"\nSVM prediction: {label}")
        if confidence is not None:
            print(f"SVM confidence: {confidence:.2%}")

    if args.model in ("mobilenet", "both"):
        mobilenet = load_mobilenet(args.mobilenet_model, device)
        label, confidence = predict_mobilenet(
            image_path, mobilenet, device
        )

        print(f"\nMobileNetV2 prediction: {label}")
        print(f"MobileNetV2 confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
