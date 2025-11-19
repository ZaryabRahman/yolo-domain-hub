# YOLOv11 Brain Tumor MRI Classifier

This directory contains a YOLOv11-nano classification model trained to classify brain tumors from MRI images into four categories.

### Model Details

| Attribute | Value |
| :--- | :--- |
| **Model** | `yolov11n-cls` |
| **Type** | Classification |
| **Classes** | `glioma`, `meningioma`, `notumor`, `pituitary` |
| **Hugging Face**| [`Link to pre-trained model`](https://huggingface.co/findingmllll/yolov11-brain-tumor-mri) |

### Performance

The model was trained for 20 epochs and achieved the following performance on the validation set:

-   **Top-1 Accuracy:** `99.4%`
-   **Top-5 Accuracy:** `100.0%` (as there are only 4 classes)

### Dataset

This model was trained on the **Brain Tumor MRI Dataset** available on Kaggle.

-   **Link:** [https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
-   The dataset contains 7023 images of human brain MRI scans.

---

### How to Use

To use this model, instantiate the `BrainTumorClassifier` class and call the `predict` method. The model will be automatically downloaded from Hugging Face on the first run.

```python
from classifier import BrainTumorClassifier

# Initialize the classifier
classifier = BrainTumorClassifier(model_version='1.0')

# Run prediction
image_file = 'path/to/your/mri_image.jpg'
prediction_results = classifier.predict(image_file)

print(prediction_results)
```

### Example Output

```json
{
  "top_prediction": {
    "class": "glioma",
    "confidence": 0.9985
  },
  "all_probabilities": [
    { "class": "glioma", "confidence": 0.9985 },
    { "class": "pituitary", "confidence": 0.0010 },
    { "class": "meningioma", "confidence": 0.0003 },
    { "class": "notumor", "confidence": 0.0002 }
  ]
}
```
