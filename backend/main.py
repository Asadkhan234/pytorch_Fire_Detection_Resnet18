from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os


# --------------------------------------------------
# 1. Create FastAPI app
# --------------------------------------------------

app = FastAPI(
    title="Fire Detection API",
    description="Fire Detection using ResNet18",
    version="1.0"
)


# --------------------------------------------------
# 2. CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# 3. Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# 4. Load ResNet18 architecture
# --------------------------------------------------

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)


# --------------------------------------------------
# 5. Load trained model
# --------------------------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "model",
    "Fire_Detection_Resnet.pth"
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


# --------------------------------------------------
# 6. Image preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# 7. Class names
# --------------------------------------------------

class_names = [
    "Fire",
    "No Fire"
]


# --------------------------------------------------
# 8. Home endpoint
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Fire Detection API is running",
        "model": "ResNet18",
        "classes": class_names
    }


# --------------------------------------------------
# 9. Health check
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device)
    }


# --------------------------------------------------
# 10. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read image
    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")


    # Preprocess
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Move to device
    image_tensor = image_tensor.to(device)


    # Prediction
    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )


    # Get result
    predicted_index = predicted_class.item()

    predicted_label = class_names[predicted_index]

    confidence_score = confidence.item()


    return {
        "prediction": predicted_label,
        "confidence": round(
            confidence_score * 100,
            2
        ),
        "class_index": predicted_index
    }