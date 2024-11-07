import os, sys
import cv2
import numpy as np
import logging
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CharacterRecognizeTF:
    def __init__(self, threshold=0.70, models=None, labels=None):
        self.model = models
        self.labels = labels
        self.threshold = threshold

    def load_model(self, model_path, weight_path):
        try:
            with open(model_path, 'r') as json_file:
                model_json = json_file.read()
            model = model_from_json(model_json)
            model.load_weights(weight_path)
            return model
        except Exception as e:
            logging.error(f'Could not load model: {e}')
            return None

    def load_labels(self, labels_path):
        try:
            labels = LabelEncoder()
            labels.classes_ = np.load(labels_path)
            return labels
        except Exception as e:
            logging.error(f'Could not load labels: {e}')
            return None

    def preprocess_image(self, image_path, resize=False):
        img = cv2.imread(image_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img / 255.0
        # if resize:
        #     img = cv2.resize(img, (224, 224))
        return img

    def predict_from_model(self, image):
        image = cv2.resize(image, (80, 80))

        if image.shape[-1] != 3:
            image = np.stack((image,) * 3, axis=-1)

        image = image.astype('float32')
        predictions = self.model.predict(image[np.newaxis, :], verbose=False)

        max_prob = np.max(predictions)
        predicted_class = np.argmax(predictions)

        if max_prob >= self.threshold:
            prediction = self.labels.inverse_transform([predicted_class])
            return prediction[0], max_prob
        else:
            return None, max_prob

    def predict_and_save(self, folder_path, output_folder):
        image_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path)]
        os.makedirs(output_folder, exist_ok=True)

        unknown_folder = os.path.join(output_folder, "unknown")
        os.makedirs(unknown_folder, exist_ok=True)

        for image_path in image_paths:
            image = self.preprocess_image(image_path, resize=True)
            pred_class, confidence = self.predict_from_model(image)

            if pred_class:
                class_folder = os.path.join(output_folder, pred_class)
                os.makedirs(class_folder, exist_ok=True)
                output_image_path = os.path.join(class_folder, os.path.basename(image_path))
            else:
                # Save in "unknown" folder if confidence is below threshold
                output_image_path = os.path.join(unknown_folder, os.path.basename(image_path))

            cv2.imwrite(output_image_path, cv2.imread(image_path))
            if pred_class:
                logging.info(f"Image: {os.path.basename(image_path)}, Prediction: {pred_class}, Confidence: {confidence}")
            else:
                logging.info(f"Image: {os.path.basename(image_path)} saved to 'unknown' folder due to low confidence ({confidence}).")


# Example usage
if __name__ == "__main__":
    base_model_path = r"C:\Users\DOT\Documents\febri\weights\ocr_model\new_model\20240925-11-01-14"

    model_path = os.path.join(base_model_path, "character_recognition.json")
    weight_path = os.path.join(base_model_path, "models_cnn.h5")
    labels_path = os.path.join(base_model_path, "character_classes.npy")
    input_folder = r"C:\Users\DOT\Documents\febri\github\ai-smartparking\output_chars_3"
    output_folder = r"C:\Users\DOT\Documents\febri\github\ai-smartparking\output_folder_tf"

    recognizer = CharacterRecognizeTF()
    recognizer.model = recognizer.load_model(model_path, weight_path)
    recognizer.labels = recognizer.load_labels(labels_path)
    
    if recognizer.model and recognizer.labels:
        recognizer.predict_and_save(input_folder, output_folder)