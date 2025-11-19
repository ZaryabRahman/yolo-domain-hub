# a simple script to demonstrate the BrainTumorClassifier.

from models.medical.brain_tumor_classifier.classifier import BrainTumorClassifier
import json


def run_example():
    """
    Initializes the classifier and runs a prediction on a sample image.
    """
    print("--- Brain Tumor Classification Example ---")

    # IMPORTANT: Download a sample image from the dataset and place it
    # in this directory. For example, a 'glioma' image.
    # Rename it to 'sample_image.jpg'.
    sample_image_path = 'sample_image.jpg'

    try:
        # 1. Initialize the classifier
        # The model will be downloaded automatically on the first run.
        classifier = BrainTumorClassifier(model_version='1.0')

        # 2. Run prediction
        print(f"\nRunning prediction on '{sample_image_path}'...")
        results = classifier.predict(sample_image_path)
        
        print("\n--- Prediction Results ---")
        print(json.dumps(results, indent=2))
        
        # 3. Generate and save a visualization
        print("\n--- Generating Visualization ---")
        classifier.visualize(sample_image_path, save_path='result_visualization.jpg')
        
    except FileNotFoundError:
        print(f"\nERROR: Could not find '{sample_image_path}'.")
        print("Please download a sample image from the dataset, place it in this directory,")
        print("and name it 'sample_image.jpg' to run this example.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == '__main__':
    run_example()
