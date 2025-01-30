import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
import numpy as np
from facenet_pytorch import MTCNN
import os

# Define class names
class_names = ["Mild", "Moderate", "Normal", "Severe"]

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image transformations (consistent with training)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device=device)

# Load saved models
def load_saved_models(model_names, model_save_path):
    models_list = []
    for model_name in model_names:
        # Load model architecture
        if model_name == "mobilenetv3_small_100":
            model = models.mobilenet_v3_small(weights=True)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))
        
        elif model_name == "shufflenet_v2_x0_5":
            model = models.shufflenet_v2_x0_5(weights=True)
            model.fc = nn.Linear(model.fc.in_features, len(class_names))

        elif model_name == "shufflenet_v2_x1_0":
            model = models.shufflenet_v2_x1_0(weights=True)
            model.fc = nn.Linear(model.fc.in_features, len(class_names))
        
        elif model_name == "squeezenet1_0":
            model = models.squeezenet1_0(weights=True)
            model.classifier[1] = nn.Conv2d(512, len(class_names), kernel_size=(1, 1))

        elif model_name == "squeezenet1_1":
            model = models.squeezenet1_1(weights=True)
            model.classifier[1] = nn.Conv2d(512, len(class_names), kernel_size=(1, 1))
        
        elif model_name == "googlenet":
            model = models.googlenet(weights=True)
            model.fc = nn.Linear(model.fc.in_features, len(class_names))

        else:
            raise ValueError(f"Model {model_name} not supported!")

        model.load_state_dict(torch.load(f"{model_save_path}/{model_name}_model.pth", map_location=device))
        model.to(device)
        model.eval()
        models_list.append(model)
    return models_list

# Ensemble prediction function
def ensemble_predict(models, image_tensor):
    predictions = torch.zeros((1, len(class_names)), device=device)
    for model in models:
        outputs = model(image_tensor)
        predictions += torch.softmax(outputs, dim=1)
    return predictions

# Face detection function
def is_facial_image(image):
    image_np = np.array(image)
    boxes, _ = mtcnn.detect(image_np)
    return boxes is not None

# Prediction function
def predict_image(image, models, threshold=0.7):
    # Preprocess the image
    image_tensor = data_transforms(image).unsqueeze(0).to(device)

    # Aggregate predictions from ensemble models
    predictions = ensemble_predict(models, image_tensor)

    # Get confidence scores
    confidence_scores = predictions.squeeze(0).detach().cpu().numpy()
    max_confidence = np.max(confidence_scores)
    predicted_index = np.argmax(confidence_scores)

    predicted_class = class_names[predicted_index]
    return predicted_class, confidence_scores

# Streamlit app
def main():
    st.title("Severity Classification of Facial Paralysis")
    st.write("Upload an image to classify the severity of facial paralysis. The system will also detect if the uploaded image is not a facial image.")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Check if the uploaded image is facial
        st.write("Detecting if the uploaded image contains a face...")
        if not is_facial_image(image):
            st.error("The uploaded image is not a facial image.")
            return

        # Load saved models
        model_names = [
            "mobilenetv3_small_100",
            "shufflenet_v2_x0_5",
            "shufflenet_v2_x1_0",
            "squeezenet1_0",
            "squeezenet1_1",
            "googlenet",
        ]
        model_save_path = "./saved_models"
        saved_models = load_saved_models(model_names, model_save_path)

        # Perform prediction
        st.write("Classifying...")
        predicted_class, confidence_scores = predict_image(image, saved_models)

        st.success(f"Predicted Class: {predicted_class}")
        st.write("Confidence Scores:")
        for i, score in enumerate(confidence_scores):
            st.write(f"{class_names[i]}: {score * 100:.2f}%")

if __name__ == "__main__":
    main()