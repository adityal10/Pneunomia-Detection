import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F


class ChestXRayModel3(nn.Module):
    """
    Model replicates the architecture of TinyVGG adding another conv block layer.
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_l = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1), # values we can set our self are hyperparameters
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*28*28,
                    out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_l(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.conv_block_3(x)
        # print(x.shape)
        x = self.classifier(x)
        return x

# Load your trained model (make sure to provide the path to your model)
@st.cache_resource
def load_model():
    model = ChestXRayModel3(input_shape=1,
                         hidden_units=10,
                         output_shape=2) # Initialize the model
    model.load_state_dict(torch.load('ChestXRay-PNEUMONIADetection.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocess the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

# Predict pneumonia
def predict(image, model):
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        _, predicted_class = torch.max(probabilities, 1)
        return probabilities, predicted_class.item()

# Streamlit app
st.markdown("<h1 style='text-align: center; color: #003366;'>Pneumonia Detection from Chest X-Ray</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666666; font-size: 18px;'>Upload a chest X-ray image, and the model will analyze it to predict whether it indicates pneumonia.</p>", unsafe_allow_html=True)

# Upload image
uploaded_file = st.file_uploader("**Choose a chest X-ray image:**", type="jpg")


if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Load model
    model = load_model()

    # Preprocess and predict
    image_tensor = preprocess_image(image)
    probabilities, predicted_class = predict(image_tensor, model)

# Display result
    class_names = ['Normal', 'Pneumonia']  # Adjust based on your model's classes
    prediction_label = class_names[predicted_class]
    confidence_score = probabilities[0][predicted_class].item() * 100

    # Format the output nicely
    st.subheader("Prediction Results:")
    if prediction_label == 'Pneumonia':
        st.markdown(f"<h3 style='color: red;'>⚠️ {prediction_label}</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color: green;'>✔️ {prediction_label}</h3>", unsafe_allow_html=True)
    
    st.write(f"**Confidence Level:** {confidence_score:.2f}%")
    st.write("This confidence score represents the likelihood that the model's prediction is correct.")

    # Additional note for users
    st.info("**Note:** This tool is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.")