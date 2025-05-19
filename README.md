# Brain Tumor Detection (End-to-End)

## Introduction

This project is a Flask web application for detecting brain tumors from MRI images using a deep learning model. Users can upload MRI scans through the app, and the model predicts whether the image indicates the presence of a tumor or not. The goal is to provide a simple and effective tool to help medical professionals quickly identify potential brain tumors.
### Dataset:
- The dataset contains MRI images, divided into two categories: **tumor** and **non-tumor**.
- Preprocessing techniques are applied to the dataset to ensure optimal model performance.

## Project Overview

* Collected and organized a brain MRI dataset into training and testing folders with two classes: **tumor** and **no tumor**.
* Preprocessed images by resizing them to 224x224 and converting to tensors for model input.
* Built a deep learning model using transfer learning with a pretrained ResNet-50, modifying the final layer for binary classification.
* Trained the model on the MRI dataset to accurately classify images as tumor or no tumor.
* Developed a Flask web application to provide a user-friendly interface for uploading MRI images.
* Integrated the trained model into the Flask app to make real-time predictions on uploaded images.
* Created HTML templates for different pages (upload, prediction result, error handling) with Bootstrap styling.
* Prepared scripts and environment files (like `train_model.py`, `app.py`, and `requirements.txt`) for easy reproducibility.

---

## Model Download and Directory Structure

### Pretrained Model:
### Dataset Source
This project uses MRI brain scan images from this public dataset:  
[Brain Tumor Detection Dataset on Kaggle](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)


### Directory Structure:
```
### Directory Structure:

brain-tumor-detection/
├── dataset/
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── no_tumor/
│   │   └── pituitary/
│   ├── Testing/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── no_tumor/
│   │   └── pituitary/
├── models/
│   └── bt_resnet50_model.py      # Model architecture definition
│   └── model.pth                 # Trained model weights
├── templates/
│   ├── Diseasedet.html           # Info page about brain tumors
│   ├── error.html                # Error page
│   ├── MainPage.html             # Main upload page
│   ├── pred.html                 # Prediction result page
│   └── uimg.html                 # Upload image page
├── static/
│   └── b.jpg                     # Static image used in templates
├── app.py                       # Flask application script
├── train_model.py               # Training script for the model
├── requirements.txt             # Project dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Git ignore file (optional)



## Setup Instructions

### Step 1: Create a Virtual Environment

Create a virtual environment to isolate the dependencies for this project.

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Required Libraries

Install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 3: Download the Pretrained Model

Download the pretrained model from [Brain Tumor Detection Dataset on Kaggle](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection) and place it in the `model/` directory as `brain_tumor_model.pth`.

### Step 4: Running the Flask App

To start the Flask web app, navigate to the `app/` directory and run the `app.py` file:

```bash
cd app/
python app.py
```

The app will be hosted at `http://127.0.0.1:5000/`. You can open the URL in your browser and upload MRI images to receive predictions.

## Flask Web Application Features

- **Image Upload**: Users can upload MRI images through the web interface.
- **Tumor Detection**: The uploaded image is fed into the model to predict whether a tumor is present.
- **Result Display**: The result is displayed on the same page with either a "Tumor" or "Non-Tumor" label.

## Model Architecture

The model usedThe model used in this project is a Convolutional Neural Network (CNN) based on the pretrained ResNet-50 architecture. Instead of building a CNN from scratch, transfer learning was applied by modifying the final fully connected layer to perform binary classification (tumor / no tumor) on brain MRI images.

### Key Layers:
- **Convolutional Layers**: For feature extraction from MRI images.
- **Max Pooling Layers**: For downsampling and reducing spatial dimensions.
- **Fully Connected Layers**: For classification.
- **Softmax Activation**: To produce the output probability of each class (Tumor/Non-Tumor).

## Data Preprocessing

To ensure the CNN model performs optimally, the following preprocessing steps are applied:
- **Grayscale Conversion**: All MRI images are converted to grayscale.
- **Resizing**: Images are resized to 64x64 pixels for uniformity.
- **Normalization**: Each pixel value is normalized to a range of [0, 1].
- **Data Augmentation**: Techniques like random rotation, flipping, and zooming are applied to expand the dataset and prevent overfitting.

## Conclusion

This Flask web app provides an end-to-end solution for detecting brain tumors using MRI images. With a simple user interface and a powerful backend, it can serve as a diagnostic tool for medical professionals. The project can be further enhanced by incorporating additional data, improving model accuracy, or deploying the app to a cloud platform like Heroku.

## Future Enhancements

- **Integration with Cloud Platforms**: Deploy the app on Heroku or AWS for wider accessibility.
- **Mobile Application**: Develop a mobile app to upload MRI images and get predictions on the go.
- **Transfer Learning**: Incorporate pre-trained models like ResNet to further improve accuracy.

---