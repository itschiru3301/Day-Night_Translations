import os
import asyncio
import streamlit as st
import torch
import torch.nn as nn
import cv2

import numpy as np
from torchvision import transforms
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Ensure an event loop is running
if not asyncio.get_event_loop().is_running():
    asyncio.set_event_loop(asyncio.new_event_loop())

# Set environment variable for Streamlit
os.environ["STREAMLIT_RUN_ONCE"] = "True"

# Define the Generator model
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

# Load the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(DEVICE)

try:
    state_dict = torch.load("generator_g.pth", map_location=DEVICE)
    generator.load_state_dict(state_dict)
    generator.eval()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def transform_frame(frame):
    """Processes a frame through the model."""
    img = Image.fromarray(frame)
    img = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = generator(img)
    output = output.squeeze().cpu().numpy().transpose(1, 2, 0)
    output = (output + 1) / 2  # Denormalize
    output = (output * 255).astype(np.uint8)
    return output

# WebRTC video processing
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transformed = transform_frame(img_rgb)
        h, w, _ = img.shape
        transformed = cv2.resize(transformed, (w, h))
        stacked_frame = np.hstack((img_rgb, transformed))
        return cv2.cvtColor(stacked_frame, cv2.COLOR_RGB2BGR)

# Streamlit UI
st.title("Live Video Transformation using CycleGAN")
st.sidebar.header("Settings")
st.write("### Original (Left) | Transformed (Right)")

# WebRTC Streaming
webrtc_streamer(
    key="video",
    video_processor_factory=VideoTransformer,
    async_processing=True
)