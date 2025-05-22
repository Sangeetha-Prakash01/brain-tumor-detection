from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
from models.bt_resnet50_model import get_resnet50_model
import os

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "models/model.pth"
model = get_resnet50_model(num_classes=4, pretrained=False, freeze=False)  # No pretrained, load weights only
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model weights not found at {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

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

    if not allowed_file(file.filename):
        return "Unsupported file type", 400

    try:
        img = Image.open(file.stream).convert('RGB')
    except Exception:
        return "Invalid image format", 400

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
        predicted_class = classes[predicted.item()]
        confidence = confidence.item() * 100

    if predicted_class == "no_tumor":
        result = f"You don't have tumor (Confidence: {confidence:.2f}%)"
    else:
        result = f"You have tumor: {predicted_class.capitalize()} (Confidence: {confidence:.2f}%)"

    return render_template('pred.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True, port=8000)

