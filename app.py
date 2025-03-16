import os
os.environ["STREAMLIT_RUN_ONCE"] = "True"

import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load trained model
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

# Load Generator model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(DEVICE)

# Ensure correct model loading
try:
    state_dict = torch.load("generator_g.pth", map_location=DEVICE)
    generator.load_state_dict(state_dict)
    generator.eval()
except RuntimeError as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Function to process a frame
def transform_frame(frame):
    img = Image.fromarray(frame)  # Convert to PIL Image
    img = transform(img).unsqueeze(0).to(DEVICE)  # Apply transformations
    with torch.no_grad():
        output = generator(img)  # Run model
    output = output.squeeze().cpu().numpy().transpose(1, 2, 0)  # Convert tensor back to numpy
    output = (output + 1) / 2  # Denormalize to [0, 1]
    output = (output * 255).astype(np.uint8)  # Convert to uint8
    return output

# WebRTC Video Processing for Streamlit Cloud
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convert to NumPy array
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        transformed = transform_frame(img_rgb)  # Apply model
        
        # Resize to ensure both frames are the same height
        h, w, _ = img.shape
        transformed = cv2.resize(transformed, (w, h))

        # Stack images side by side (original on left, transformed on right)
        stacked_frame = np.hstack((img_rgb, transformed))
        
        return cv2.cvtColor(stacked_frame, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV

# Streamlit UI
st.title("Live Video Transformation using CycleGAN")
st.sidebar.header("Settings")

st.write("### Original (Left) | Transformed (Right)")

# WebRTC Streaming
webrtc_streamer(
    key="video",
    video_transformer_factory=VideoTransformer,
    async_transform=True,
)
