import torch
from ultralytics import YOLO
from typing import List, Dict, Any, Union
from PIL import Image
import numpy as np
import cv2

# maps user-friendly versions to Hugging Face model IDs.

MODEL_REGISTRY = {
    '1.0': 'findingmllll/yolov11-brain-tumor-mri',
}

class BrainTumorClassifier:
    """
    A wrapper for the YOLOv11 Brain Tumor MRI classification model.
    Provides a simple API to get predictions from an image.
    """
    def __init__(self, model_version: str = '1.0'):
        """
        Initializes the classifier.

        Args:
            model_version (str): The version of the model to use.
        """
        if model_version not in MODEL_REGISTRY:
            raise ValueError(f"Model version '{model_version}' not found. "
                             f"Available versions: {list(MODEL_REGISTRY.keys())}")

        self.model_id = MODEL_REGISTRY[model_version]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Initializing classifier with model '{self.model_id}' on device '{self.device}'...")
        
        # YOLO handles automatic download from Hugging Face Hub.
        self.model = YOLO(self.model_id)
        self.model.to(self.device)
        print("Classifier initialized successfully.")

    def predict(self, image_source: Union[str, np.ndarray, Image.Image], top_k: int = 4) -> Dict[str, Any]:
        """
        Runs inference on a given image and returns class probabilities.

        Args:
            image_source (Union[str, np.ndarray, Image.Image]): Input image.
            top_k (int): The number of top predictions to return.

        Returns:
            Dict[str, Any]: A dictionary containing the top prediction and a list
                            of probabilities for all classes.
        """
        results = self.model(image_source, verbose=False)
        result = results[0]  # get the result for the first (and only) image
        
        # get top K predictions
        top_k_indices = result.probs.topk(k=top_k).indices.cpu().tolist()
        top_k_confidences = result.probs.topk(k=top_k).values.cpu().tolist()

        # format the output
        all_probabilities = []
        for i, class_name in result.names.items():
            all_probabilities.append({
                'class': class_name,
                'confidence': float(result.probs.data[i])
            })
        
        # sort by confidence
        all_probabilities.sort(key=lambda x: x['confidence'], reverse=True)

        return {
            "top_prediction": {
                "class": result.names[top_k_indices[0]],
                "confidence": round(top_k_confidences[0], 4)
            },
            "all_probabilities": all_probabilities
        }
        
    def visualize(self, image_path: str, save_path: str = 'output_classification.jpg') -> np.ndarray:
        """
        Runs inference and returns an image with the top prediction written on it.
        
        Args:
            image_path (str): Path to the input image.
            save_path (str): Path to save the output image. If None, image is not saved.
            
        Returns:
            np.ndarray: The image as a NumPy array with the prediction text.
        """
        prediction = self.predict(image_path)
        top_pred = prediction['top_prediction']
        
        # Load image with OpenCV to draw on it
        image = cv2.imread(image_path)
        
        # prepare text to display
        label = f"{top_pred['class']} ({top_pred['confidence']:.2f})"
        
        # set font and colors
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_color = (255, 255, 255)  # White
        bg_color = (0, 128, 0)      # Green background
        
        # get text size to draw a background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # draw the background rectangle and the text
        cv2.rectangle(image, (5, 5), (5 + text_width + 10, 5 + text_height + baseline + 5), bg_color, -1)
        cv2.putText(image, label, (10, 10 + text_height), font, font_scale, text_color, font_thickness)

        if save_path:
            cv2.imwrite(save_path, image)
            print(f"Visualization saved to {save_path}")
            
        return image
