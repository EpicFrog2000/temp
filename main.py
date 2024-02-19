from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image, ImageDraw, ExifTags, ImageFont
import time
import model_operations
import hashlib
import numpy as np
import cv2

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

# rop image
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
        documentBox, found, img = model_operations.detect_document(model, img)
        
        #TODO: dodać wycianie Alex
        
        
        if found:
            img = crop_image(img, documentBox)
            img = Image.fromarray(img)
            img = img.convert("RGB")
            processedImgs.append(img)
    return processedImgs

if __name__ == "__main__":
    
    # Apply the measure_runtime decorator to all functions in this module to mesure execution time of functions
    enable_measure_runtime()
   
    # Try to use gpu
    device = check_gpu()
    print(device)
    
    # Initialize models
    mtcnn = MTCNN(image_size=160, margin=10, min_face_size=20, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    model = model_operations.load_document_detection_model()
    
    # If True -> when program matches face, displays images of document and user face
    SHOW_RESULT = True
    
    # Paths to documents
    stored_img_paths = ['images/dk1.jpg' ,'images/img3.jpg', 'images/img4.jpg', 'images/img5.jpg', 'images/img2.jpg', 'images/img1.jpg']

    # Open crop and process images
    processedImgs = preprocess(stored_img_paths)
    
    # Compute embeddings for stored images
    stored_img_embeddings = compute_embeddings(processedImgs)
    
    # Path to user photo
    user_photo = 'images/JA (2).jpg'
    
    # Checks if face from user_photo matches with any faces found in photos in stored_img_paths
    result = compare_faces(user_photo, stored_img_embeddings, SHOW_RESULT)
    print(result)