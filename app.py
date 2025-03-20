# app.py
import asyncio
import sys

# Event loop fix must be FIRST
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "no current event loop" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()
            
get_or_create_eventloop()

# Torch classes workaround BEFORE other imports
import torch
torch.classes.__path__ = []  # Explicit empty path

# Now other imports
import streamlit as st
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import av
from streamlit_webrtc import webrtc_streamer

# Generator definition
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
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

@st.cache_resource
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def process_frame(frame: av.VideoFrame, model) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    pil_img = Image.fromarray(image[..., ::-1])  # BGR to RGB
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    with torch.no_grad():
        tensor = transform(pil_img).unsqueeze(0).to(next(model.parameters()).device)
        output = model(tensor)[0].cpu().numpy()
    
    output = (np.transpose(output, (1, 2, 0)) * 0.5 + 0.5) * 255
    return av.VideoFrame.from_ndarray(output[..., ::-1].astype(np.uint8), format="bgr24")

def main():
    st.title("Real-Time Day/Night Converter")
    
    # Model selection
    mode = st.radio("Conversion Mode:", 
                   ("Day → Night", "Night → Day"),
                   horizontal=True)
    
    # Load model
    model = load_model("generator_g.pth" if mode == "Day → Night" else "generator_f.pth")
    
    # WebRTC component
    webrtc_ctx = webrtc_streamer(
        key="cyclegan",
        video_frame_callback=lambda frame: process_frame(frame, model),
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True
    )
    
    if not webrtc_ctx.state.playing:
        st.info("Waiting for camera access...")
        st.warning("Please enable camera permissions")

if __name__ == "__main__":
    main()
