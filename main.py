from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image, ImageDraw, ExifTags

# Initializing models
mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) 
resnet = InceptionResnetV1(pretrained='vggface2').eval()

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
def draw_and_show_faces(stored_img_path, stored_prob, prob, stored_img):
    print(stored_img_path)
    print("stored_prob: ", stored_prob)
    print("prob: ", prob)
    draw = ImageDraw.Draw(stored_img)
    boxes, _ = mtcnn.detect(stored_img)
    if boxes is not None:
        for box in boxes:
            draw.rectangle(box.tolist(), outline="red", width=6)
    stored_img.show()


# Takes photo of face and array with images, checks and returns if found matching face among stored_img_paths
def compare_faces(img_path, stored_img_paths, threshold=0.9): 
    img = Image.open(img_path)
    img = rotate_image(img)
    face, prob = mtcnn(img, return_prob=True) 
    
    if face is not None and prob > threshold:
        
        emb = resnet(face.unsqueeze(0)).detach()
        
        min_dist = float('inf')
        best_match_path = None
        
        for stored_img_path in stored_img_paths:
            
            stored_img = Image.open(stored_img_path)
            stored_img = rotate_image(stored_img)
            stored_face, stored_prob = mtcnn(stored_img, return_prob=True)
            
            draw_and_show_faces(stored_img_path, stored_prob, prob, stored_img)
            
            if stored_face is not None and stored_prob > threshold:
                
                stored_emb = resnet(stored_face.unsqueeze(0)).detach()
                dist = torch.dist(emb, stored_emb).item()
                
                if dist < threshold and dist < min_dist:
                    min_dist = dist
                    best_match_path = stored_img_path
        
        if best_match_path:
            return "Match found: {}".format(best_match_path)
        else:
            return "No match found"
    else:
        return "No face detected or face probability less than 90%"

stored_img_paths = ['images/img3.jpg', 'images/img4.jpg', 'images/img5.jpg', 'images/img2.jpg', 'images/img1.jpg']
#result = compare_faces('images/zuza.jpg', stored_img_paths)
#result = compare_faces('images/Ana.jpg', stored_img_paths)
#result = compare_faces('images/Danciu.jpg', stored_img_paths)
#result = compare_faces('images/Justyna.jpg', stored_img_paths)
result = compare_faces('images/Klaudia.jpg', stored_img_paths)

print(result)
