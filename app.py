import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torch import nn

# Streamlit App Title
st.title("🌙 CycleGAN Real-Time Video Transformation")
st.markdown("Convert day-to-night or night-to-day in real-time using your webcam! 🚀")

# Sidebar Settings
st.sidebar.header("Settings")
transform_type = st.sidebar.radio("Select Transformation", ("Day to Night", "Night to Day"))

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Definition (Must Match Training)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# Load Model with Caching
@st.cache_resource
def load_generator(model_path):
    model = Generator().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

# Ensure model files exist before loading
try:
    generator_g = load_generator("generator_g.pth")  # Day to Night
    generator_f = load_generator("generator_f.pth")  # Night to Day
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Image Processing Functions
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def preprocess_frame(image):
    """Convert image to model-compatible tensor."""
    return transform(image).unsqueeze(0).to(DEVICE)

def postprocess_frame(tensor):
    """Convert model output tensor back to image."""
    tensor = tensor.squeeze(0).detach().cpu().numpy()
    tensor = (tensor * 0.5 + 0.5) * 255  # Reverse normalization
    tensor = np.clip(tensor, 0, 255).astype(np.uint8)
    return Image.fromarray(np.transpose(tensor, (1, 2, 0)))

# Webcam Setup
FRAME_WINDOW = st.image([])
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("⚠️ Cannot access the webcam. Please enable camera permissions or try restarting your browser.")
    st.stop()

# Main Loop for Real-Time Transformation
while True:
    ret, frame = cap.read()
    if not ret:
        st.warning("⚠️ No frame captured. Check your camera settings.")
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR by default
    input_image = Image.fromarray(frame)
    processed_image = preprocess_frame(input_image)

    # Apply Model Transformation
    with torch.no_grad():
        generator = generator_g if transform_type == "Day to Night" else generator_f
        transformed_tensor = generator(processed_image)

    transformed_frame = postprocess_frame(transformed_tensor)

    # Display Output Frame
    FRAME_WINDOW.image(transformed_frame, use_column_width=True)

# Release Camera on Exit
cap.release()
st.write("🔴 Webcam feed stopped.")
