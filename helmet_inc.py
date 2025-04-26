import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tkinter import Tk, filedialog

def load_mobilenet_model():
    model = MobileNetV2(weights='imagenet')
    return model

def preprocess_image(img):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def classify_image(model, img_array):
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions)
    return decoded_predictions

def is_person_wearing_helmet(predictions):
    for _, label, confidence in predictions[0]:
        if 'helmet' in label.lower():
            return True, confidence
    return False, 0.0

def main():
    root = Tk()
    root.withdraw()

    try:
        # Ask user to choose an image file
        file_path = filedialog.askopenfilename(title="Select an Image File", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

        if not file_path:
            print("No file selected. Exiting.")
            return

        # Load MobileNetV2 model
        model = load_mobilenet_model()

        # Read the image
        img = cv2.imread(file_path)

        if img is None:
            print(f"Error: Unable to read the image from {file_path}")
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV loads images in BGR, convert to RGB

        # Resize the image to match the expected input shape
        img = cv2.resize(img, (224, 224))

        # Preprocess the image
        img_array = preprocess_image(img)

        # Classify the image
        predictions = classify_image(model, img_array)

        # Check if person is wearing a helmet
        is_wearing_helmet, confidence = is_person_wearing_helmet(predictions)

        # Display the result
        if is_wearing_helmet:
            print(f"The person is wearing a helmet with confidence: {confidence:.2f}")
        else:
            print("The person is not wearing a helmet.")
    except FileNotFoundError:
        print("Error: File not found.")
    except cv2.error as cv2_error:
        print(f"OpenCV Error: {cv2_error}")
    except Exception as e:
        print(f"Unexpected Error: {e}")
    finally:
        # Explicitly release Tkinter resources
        root.destroy()

if __name__ == "__main__":
    main()
