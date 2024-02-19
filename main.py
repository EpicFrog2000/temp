from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image, ImageDraw, ExifTags
import time
import hashlib
import numpy as np
import os
# To stop GPU loading with tensorflow logs from printing 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or '2' to display warnings and errors
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import random


# Define the measure_runtime decorator
def measure_runtime(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {execution_time:.6f} seconds")
        return result
    return wrapper


# Apply the measure_runtime decorator to all functions in the module
def enable_measure_runtime():
    for name, obj in globals().items():
        if callable(obj) and obj.__module__ == __name__:
            globals()[name] = measure_runtime(obj)


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
        image_to_test = np.array(paste_rescaled_image_at_center(image))
        
        if len(image_to_test.shape) != 3:
            #print("Error in detect_document: Image has incorrect shape")
            return documentBox, False, image_to_test
            
        image_tensor = tf.convert_to_tensor(image_to_test, dtype=tf.uint8)[tf.newaxis, ...]

        detections = model(image_tensor)
        scores = detections['detection_scores'].numpy()
        boxes = detections['detection_boxes'].numpy()
        for i, box in enumerate(boxes):
            max_score_index = scores[i].argmax()
            score = scores[i][max_score_index]

            if score > 0.5:
                h, w, _ = image_to_test.shape
                ymin, xmin, ymax, xmax = box[max_score_index]

                xmin, xmax, ymin, ymax = int(xmin * w), int(xmax * w), int(ymin * h), int(ymax * h)
                documentBox = [xmin, xmax, ymin, ymax]

                return documentBox, True, image_to_test

        return documentBox, False, image_to_test

    except Exception as e:
        print(f"There was an Error: {e}")

        return documentBox, False, image_to_test


# Make pytorch use GPU is possible
def check_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if "amd" in torch.cuda.get_device_name().lower():
            print("AMD GPU detected.")
            torch.cuda.set_device(torch.cuda.current_device())
        else:
            print("NVIDIA GPU detected.")
    else:
        device = torch.device("cpu")
        print("No GPU detected, using CPU.")
    return device


# Rotate the image based on EXIF orientation metadata
def rotate_image(img):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = img._getexif()
        if exif is not None:
            exif = dict(exif.items())
            if orientation in exif:
                if exif[orientation] == 3:
                    img = img.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    img = img.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    img = img.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError, TypeError):
        pass
    return img


# Draw boxes around found faces and show images with boxes
def draw_and_show_faces(stored_img, img):
    draw = ImageDraw.Draw(stored_img)
    boxes, _ = mtcnn.detect(stored_img)
    bounding_box_coordinates = [] 
    if boxes is not None:
        for box in boxes:
            draw.rectangle(box.tolist(), outline="green", width=4)
            bounding_box_coordinates.append(box.tolist())
    
    cropped_images = []
    for box_coords in bounding_box_coordinates:
        x_min, y_min, x_max, y_max = box_coords
        cropped_img = stored_img.crop((x_min, y_min, x_max, y_max))
        cropped_images.append(cropped_img)

    target_height = img.height
    for i in range(len(cropped_images)):
        ratio = target_height / cropped_images[i].height
        new_width = int(cropped_images[i].width * ratio)
        cropped_images[i] = cropped_images[i].resize((new_width, target_height))

    img = img.resize((int(img.width * target_height / img.height), target_height))
    stored_img = stored_img.resize((int(stored_img.width * target_height / stored_img.height), target_height))

    combined_width = img.width + stored_img.width
    max_height = max(img.height, stored_img.height)
    combined_img = Image.new('RGB', (combined_width, max_height))

    combined_img.paste(img, (0, 0))
    combined_img.paste(stored_img, (img.width, 0))

    combined_with_cropped = Image.new('RGB', (combined_img.width + sum([img.width for img in cropped_images]), max_height))
    combined_with_cropped.paste(combined_img, (0, 0))


    x_offset = combined_img.width
    for cropped_img in cropped_images:
        combined_with_cropped.paste(cropped_img, (x_offset, 0))
        x_offset += cropped_img.width

    combined_with_cropped.show()


# Compute embeddings for all stored images
def compute_embeddings(processedImgs):
    embeddings = {}
    for processedImg in processedImgs:
        stored_face, stored_prob = mtcnn(processedImg, return_prob=True)
        if stored_face is not None and stored_prob > 0.9:
            img_hash = hashlib.sha256(processedImg.tobytes()).hexdigest()
            embeddings[img_hash] = [resnet(stored_face.unsqueeze(0)).detach(), processedImg]
    return embeddings


# Takes photo of face and array with images, checks and returns if found matching face among stored_img_paths
def compare_faces(img_path, stored_img_embeddings, SHOW_RESULT, threshold=0.9):
    img = Image.open(img_path)
    img = rotate_image(img)
    face, prob = mtcnn(img, return_prob=True)

    if face is not None and prob > threshold:
        emb = resnet(face.unsqueeze(0)).detach()
        min_dist = float('inf')
        best_match_processed_img = None
        for img_hash, (stored_emb, processedImg) in stored_img_embeddings.items():
            dist = torch.dist(emb, stored_emb).item()
            if dist < threshold and dist < min_dist:
                min_dist = dist
                best_match_processed_img = processedImg

        if best_match_processed_img:
            if SHOW_RESULT:
                draw_and_show_faces(best_match_processed_img, img)
            return "Match found"
        else:
            return "No match found"
    else:
        return "No face detected or face probability less than 90%"


# Crop image
def crop_image(image, box):
    [xmin, xmax, ymin, ymax] = box
    height_increase = int((ymax - ymin) * 0.05)
    width_increase = int((xmax - xmin) * 0.05)
    ymin -= height_increase
    xmin -= width_increase
    ymax += height_increase
    xmax += width_increase
    ymin = max(0, ymin)
    xmin = max(0, xmin)
    ymax = min(image.shape[0], ymax)
    xmax = min(image.shape[1], xmax)
    cropped_image = image[ymin:ymax, xmin:xmax]
    return cropped_image


# Open and process and store opened images for later use
def preprocess(stored_img_paths):
    processedImgs = []
    for img_path in stored_img_paths:
        img = Image.open(img_path)
        img = rotate_image(img)
        documentBox, found, img = detect_document(model, img)
        
        #TODO: ewentualnie dodać wycianie Alex
        
        
        if found:
            img = crop_image(img, documentBox)
            img = Image.fromarray(img)
            img = img.convert("RGB")
            processedImgs.append(img)
    return processedImgs


if __name__ == "__main__":
    
    # Apply the measure_runtime decorator to all functions in this module to mesure execution time of functions
    enable_measure_runtime()
    
    # If True -> when program matches face, displays images of document and user face
    SHOW_RESULT = True
    
    # Try to use gpu
    device = check_gpu()
    print(device)

    # Initialize models
    mtcnn = MTCNN(image_size=160, margin=10, min_face_size=20, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    model = load_document_detection_model()
    
    # Paths to documents from which facenet will try to find user face
    stored_img_paths = ['images/dk1.jpg' ,'images/img3.jpg', 'images/img4.jpg', 'images/img5.jpg', 'images/img2.jpg', 'images/img1.jpg']
    
    # Open crop and process images
    processedImgs = preprocess(stored_img_paths)

    # Compute embeddings for stored images
    stored_img_embeddings = compute_embeddings(processedImgs)

    # Path to user photo (assuming there is a face on it)
    user_photo = 'images/mojamorda.jpg'
    
    # Checks if face from user_photo matches with any faces found in photos in stored_img_paths
    result = compare_faces(user_photo, stored_img_embeddings, SHOW_RESULT)
    print(result)