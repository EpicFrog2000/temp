from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image, ImageDraw, ExifTags
import time
import inspect

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
def apply_decorator_to_all_functions():
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
        exif = dict(img._getexif().items())
        if exif[orientation] == 3:
            img = img.rotate(180, expand=True)
        elif exif[orientation] == 6:
            img = img.rotate(270, expand=True)
        elif exif[orientation] == 8:
            img = img.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass
    return img

# Draw boxes around found faces and show images with boxes
def draw_and_show_faces(stored_img_path, img):
    stored_img = Image.open(stored_img_path)
    stored_img = rotate_image(stored_img)
    draw = ImageDraw.Draw(stored_img)
    boxes, _ = mtcnn.detect(stored_img) 
    if boxes is not None:
        for box in boxes:
            draw.rectangle(box.tolist(), outline="green", width=16)

    min_height = min(stored_img.height, img.height)
    img = img.resize((int(img.width * min_height / img.height), min_height))
    stored_img = stored_img.resize((int(stored_img.width * min_height / stored_img.height), min_height))

    combined_width = img.width + stored_img.width
    max_height = max(img.height, stored_img.height)
    combined_img = Image.new('RGB', (combined_width, max_height))

    combined_img.paste(img, (0, 0))
    combined_img.paste(stored_img, (img.width, 0))

    combined_img.show()

# Compute embeddings for all stored images
def compute_embeddings(stored_img_paths):
    embeddings = {}
    for stored_img_path in stored_img_paths:
        stored_img = Image.open(stored_img_path)
        stored_img = rotate_image(stored_img)
        stored_face, stored_prob = mtcnn(stored_img, return_prob=True)
        if stored_face is not None and stored_prob > 0.9:
            embeddings[stored_img_path] = resnet(stored_face.unsqueeze(0)).detach()
    return embeddings

# Takes photo of face and array with images, checks and returns if found matching face among stored_img_paths
def compare_faces(img_path, stored_img_embeddings, threshold=0.9):
    img = Image.open(img_path)
    img = rotate_image(img)
    face, prob = mtcnn(img, return_prob=True)

    if face is not None and prob > threshold:
        emb = resnet(face.unsqueeze(0)).detach()

        min_dist = float('inf')
        best_match_path = None

        for stored_img_path, stored_emb in stored_img_embeddings.items():
            dist = torch.dist(emb, stored_emb).item()
            if dist < threshold and dist < min_dist:
                min_dist = dist
                best_match_path = stored_img_path

        if best_match_path:
            #draw_and_show_faces(best_match_path, img)
            return "Match found: {}".format(best_match_path)
        else:
            return "No match found"
    else:
        return "No face detected or face probability less than 90%"




if __name__ == "__main__":
    
    # Apply the measure_runtime decorator to all functions in this module
    apply_decorator_to_all_functions()
    
    # Try to use gpu
    device = check_gpu()
    print(device)
    
    # Initialize models
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    # Compute embeddings for stored images
    stored_img_paths = ['images/img3.jpg', 'images/img4.jpg', 'images/img5.jpg', 'images/img2.jpg', 'images/img1.jpg']
    stored_img_embeddings = compute_embeddings(stored_img_paths)
    
    # Path to user photo
    user_photo = 'images/Danciu.jpg'
    
    # Checks if face from user_photo matches with any faces found in photos in stored_img_paths
    result = compare_faces(user_photo, stored_img_embeddings)
    print(result)
