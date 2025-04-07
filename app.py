from flask import Flask, render_template, request
import torch
from torchvision import transforms, models
from PIL import Image
import os

app = Flask(__name__)

# Load the trained model
model = models.resnet50()
model.fc = torch.nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load("models/model.pth", map_location=torch.device('cpu')))
model.eval()

# Define image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Define class names
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']


@app.route('/')
def index():
    return render_template('MainPage.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    # Read image
    img = Image.open(file.stream).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Prediction
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        label = classes[predicted.item()]

    return render_template('pred.html', prediction=label)


if __name__ == '__main__':
   app.run(debug=True, port=8000)

