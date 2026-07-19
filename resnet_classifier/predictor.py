import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

class ResNetPredictor:
    """
    Runs model inference on single MRI scans or image files to classify them.
    """
    def __init__(self, model_path, model_dir=None):
        self.model_path = model_path
        self.model_dir = model_dir
        self.model = None
        self._resolve_and_load_model()

    def _resolve_and_load_model(self):
        """Resolves the path to the model file and loads it."""
        # 1. Resolve relative filename if model_dir is provided
        if self.model_dir and not os.path.isabs(self.model_path):
            potential_path = os.path.join(self.model_dir, self.model_path)
            if os.path.exists(potential_path):
                self.model_path = potential_path

        # 2. Check if the resolved path is a directory (e.g. search for .keras or .h5 files)
        if os.path.isdir(self.model_path):
            model_files = [f for f in os.listdir(self.model_path) if f.endswith(('.keras', '.h5'))]
            if model_files:
                self.model_path = os.path.join(self.model_path, model_files[0])

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        print(f"Loading ResNet model from: {self.model_path}")
        self.model = load_model(self.model_path, compile=False)
        print("Model loaded successfully.")

    def predict(self, image_path):
        """
        Runs model inference on an image path.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found at {image_path}")

        # Read image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image file: {image_path}")

        # Convert BGR to RGB (Keras expects RGB format)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to (256, 256) matching the target_size of training
        img = cv2.resize(img, (256, 256))

        # Convert to float32 and apply ResNet50 preprocessing
        processed_img = img.astype(np.float32)
        processed_img = preprocess_input(processed_img)

        # Expand dimensions for batch execution (1, 256, 256, 3)
        batch_img = np.expand_dims(processed_img, axis=0)

        # Predict probabilities
        yhat = self.model.predict(batch_img, verbose=0)
        return yhat

    def get_predicted_class(self, image_path, class_names=None):
        """
        Predicts class probabilities and returns class index, class name, and list of probabilities.
        """
        yhat = self.predict(image_path)
        predicted_idx = int(np.argmax(yhat, axis=1)[0])
        
        probs = yhat[0].tolist()
        if class_names and predicted_idx < len(class_names):
            return predicted_idx, class_names[predicted_idx], probs
        return predicted_idx, None, probs
