import os
# To stop GPU loading with tensorflow logs from printing 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or '2' to display warnings and errors
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from PIL import Image
from io import BytesIO
import random

# Preload background images
background_folder = "background_images/"
background_files = [os.path.join(background_folder, file) for file in os.listdir(background_folder)]
preloaded_background_images = [Image.open(file) for file in background_files]


def paste_rescaled_image_at_center(image_to_paste, scaling_factor=0.90):
    try:
        # Randomly select a preloaded background image
        background_image = random.choice(preloaded_background_images).copy()
        
        # Calculate scaling factor to fit the image within the background
        bg_width, bg_height = background_image.size
        img_width, img_height = image_to_paste.size
        scaled_factor = min(bg_width / img_width, bg_height / img_height) * scaling_factor

        # Calculate new dimensions after scaling
        new_width = int(img_width * scaled_factor)
        new_height = int(img_height * scaled_factor)

        # Resize image using Lanczos interpolation
        resized_image = image_to_paste.resize((new_width, new_height), Image.LANCZOS)

        # Calculate position to paste the resized image at the center
        position = (((bg_width - new_width) // 2), ((bg_height - new_height) // 2))

        # Paste resized image onto the background
        background_image.paste(resized_image, position)      
        return background_image

    except Exception as e:
        print(f"Error in paste_rescaled_image_at_center: {e}")
        return image_to_paste


# Function to load the document detection model
def load_document_detection_model():
    try:
        # Path to saved document detection model
        model_path = 'my_model_15/saved_model/'

        # Load TensorFlow saved model
        model = tf.saved_model.load(model_path)
        return model

    except Exception as e:
        print(f"Error in load_document_detection_model: {e}")
        return None


# Functon to detect document in image using document detection model
def detect_document(model, image):
    documentBox = []
    try:
        # Open image from response content
        #image = BytesIO(image_response.content)
        #image_to_paste = Image.open(image)

        # Paste and rescale image for better document detection
        image_to_test = np.array(paste_rescaled_image_at_center(image))
        
        # Image has incorrect shape
        if len(image_to_test.shape) != 3:
            #print("Error in detect_document: Image has incorrect shape")
            return documentBox, False, image_to_test
            
        # Convert image to tensor
        image_tensor = tf.convert_to_tensor(image_to_test, dtype=tf.uint8)[tf.newaxis, ...]

        # Run document detection model on image
        detections = model(image_tensor)
        scores = detections['detection_scores'].numpy()
        boxes = detections['detection_boxes'].numpy()
        for i, box in enumerate(boxes):
            max_score_index = scores[i].argmax()
            score = scores[i][max_score_index]

            # Check if detected document has a high confidence score
            if score > 0.5:
                h, w, _ = image_to_test.shape
                ymin, xmin, ymax, xmax = box[max_score_index]

                # Convert normalized coordinates to pixel coordinates
                xmin, xmax, ymin, ymax = int(xmin * w), int(xmax * w), int(ymin * h), int(ymax * h)
                documentBox = [xmin, xmax, ymin, ymax]

                return documentBox, True, image_to_test

        # Return empty documentBox and False if document is no detected
        return documentBox, False, image_to_test

    except Exception as e:
        print(f"There was an Error: {e}")

        # Return empty documentBox nad False in case of an error
        return documentBox, False, image_to_test