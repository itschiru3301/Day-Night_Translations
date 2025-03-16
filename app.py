import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

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
generator.load_state_dict(torch.load("generator_g.pth", map_location=DEVICE))
generator.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Function to process a frame
def transform_frame(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert to PIL Image
    img = transform(img).unsqueeze(0).to(DEVICE)  # Apply transformations
    with torch.no_grad():
        output = generator(img)  # Run model
    output = output.squeeze().cpu().numpy().transpose(1, 2, 0)  # Convert tensor back to numpy
    output = (output + 1) / 2  # Denormalize to [0, 1]
    output = (output * 255).astype(np.uint8)  # Convert to uint8
    return output

# Streamlit App
st.title("Live Video Transformation using CycleGAN")
st.sidebar.header("Settings")

# Select camera
camera_source = st.sidebar.selectbox("Select Camera", ["Default Camera"], index=0)

# Start video capture
cap = cv2.VideoCapture(0)  # 0 is default webcam
frame_placeholder = st.empty()  # Placeholder for video frames

st.sidebar.write("Press **Q** in the video window to stop.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.sidebar.write("Error: Unable to access camera")
        break

    frame_resized = cv2.resize(frame, (256, 256))  # Resize original frame
    transformed_frame = transform_frame(frame_resized)  # Apply model

    # Stack resized original and transformed images side-by-side
    stacked_frame = np.hstack((cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB), transformed_frame))

    frame_placeholder.image(stacked_frame, channels="RGB", use_column_width=True)

    # Stop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
