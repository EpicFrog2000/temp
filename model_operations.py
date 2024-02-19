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
    
# TODO: maybe fix below encounered errors:
# I am unable to replicate them at will even with given data so can't do mucha about it xd
# Conclusion: It is what it is, does not crash so git gut no? yes.

#ITEM:  {'user_id': '14640421', 'verification_id': '103909', 'identity_document_type': '3', 'identity_document_front': 'https://storage.googleapis.com/pw-com-private-bucket/identity-verification/kzx9gzookob7oorr7ng1jd30g.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=cloudsql%40pw-com.iam.gserviceaccount.com%2F20240213%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240213T190658Z&X-Goog-Expires=604800&X-Goog-SignedHeaders=host&X-Goog-Signature=3e40c33bb960419ac5cb7fffc2b0638de47f8fa490eed095cd322d04e1631e035643e9553c207a69d1e14b5278901cc20d260bc9dbe91f3915d4680031a0602463b3d3921d359ea9796a7b85ac4f11f1de0bf974dea01273515facc3bfbc10714d013b486a334e6be29393b8ac31dd7fce8ec609e235760faa50cf8313d657c30538dc87ccde6deaa12e6a032cfc5e0c0d8e8fb9c65ba6815823729fc2a2d2b1af6bafe3575cda4fc0640c0a78f9035108176d6dfd374aabc567b9cc2b07665c048ca28a3baaaf1f169ed4ccea20c8fd8272ae888ac795b842ee4b784834678bb69c5bfaf4c5e06a71982fa30b57916b6f2ade25388539ea8d0a3efb22c50ada', 'identity_document_back': 'https://storage.googleapis.com/pw-com-private-bucket/identity-verification/di439d00yazw9r1_7qg_sg2jh.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=cloudsql%40pw-com.iam.gserviceaccount.com%2F20240213%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240213T190658Z&X-Goog-Expires=604800&X-Goog-SignedHeaders=host&X-Goog-Signature=31536d5cc61efebee37b097fb014386302a56a897b933ec2bf0817d44d4ba99deaf90cfa5e0dfc08456e9260be24eac9eb32d4321044d7048cc36446a16acd264e6bb452c24a75ae3652e4b31d2491936fea8750d6e2a511b7d847e3c4e81ff511ffe8f2ae05ad7a718c1a372d550011a32d9e596ee0419261444d1e7589f3175cfd4fe651794861f8a27c5e27bca41036606b9381cd2c72f1823f4a339076215fe64641714d78a0f2b3ed8bd5f201abb24b9d230fadb212c6f2d7008d61ead1ed9a12fde6cebd9ac2dab43e6fbed980127f03eb89651ce1ed6d29ecb09bb42e262537a6a251b784578e0f5a9507bef36ca17087b30579f63a3497e8949aec36'}
#Error in image validation: (PreconditionNotMet) Tensor holds no memory. Call Tensor::mutable_data firstly.
#[Hint: holder_ should not be null.] (at ..\paddle\phi\core\dense_tensor_impl.cc:44)

#ITEM:  {'user_id': '14573796', 'verification_id': '103904', 'identity_document_type': '2', 'identity_document_front': 'https://storage.googleapis.com/pw-com-private-bucket/identity-verification/ptsday36p9ciuty0vi28x9cwh.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=cloudsql%40pw-com.iam.gserviceaccount.com%2F20240213%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240213T182450Z&X-Goog-Expires=604800&X-Goog-SignedHeaders=host&X-Goog-Signature=41f0a3b9b921b099bc895b32f7806340c90451500684a0104f372f7bd0edf4038be6efe127c4e2f507ab4c58e9db9e940250f044a2cb87cc962dd1f4bdfe80f58bb13a3f568687de9e278c8dd39a72bcd99c8e900e54dc753c862f31c0ed1b80c66e4bf6aa28dd442101b0ea8dc7e18e57d9269ef130bbb65bf95814f03a807aeaf72868abc08de5119ab8ede4dd59cb2a5804d1057789d9e67bbb2b105663ae44277ea8792f0643c7071e5b1796a339757cac88911082e5e698ef7f9dd0d0dcb2ecac400ca38d3e80ac6cc6142356d4740a92339324c9e0a744069fdbe6d21b46bd90cdf3052fe3b05d44ed2a4bbda835e48cc8adfd440ea8aa2091d49be0bc', 'identity_document_back': 'https://storage.googleapis.com/pw-com-private-bucket/identity-verification/lqoeo7sw8c2smz4i75s13m1kv.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=cloudsql%40pw-com.iam.gserviceaccount.com%2F20240213%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240213T182450Z&X-Goog-Expires=604800&X-Goog-SignedHeaders=host&X-Goog-Signature=60bc65c26751323cb78eb70ce2a05f682a881fcf9955443d05c5683bfef52898b540e1ea5c22285ec3de0cb2cdd6552bd70bd4717090a7f1db577f68a75113b150975f278d24a3f8eae81a10ea076a67bb966f04547b7b249eaef149709936b6bd7f826a6860d0fa7e4a02af0ff36e357ac92152b6efc0b6fba0099e28f1ea34017e13e6440889b0ad826f4ff481294c7cab92912de18ab05d1c484ce56be82c9bdf6757d37f1bf9d2f34c7a8fc7ac423f5ce45f9afe9cc9d7fcdee0ea1bd7e3d0a4ef5e283b375dece65bc9a6e8ea46fedcbe4b0b12a1e4d3c6b14baa31da91096e97d401b2820bc193105fe02a40158139f9123661357fe04d468654cf3cdf'}
#Error in paste_rescaled_image_at_center: broken data stream when reading image file
#Error in paste_rescaled_image_at_center: I/O operation on closed file

#ITEM:  {'user_id': '14634859', 'verification_id': '103918', 'identity_document_type': '2', 'identity_document_front': 'https://storage.googleapis.com/pw-com-private-bucket/identity-verification/wzlsv6e4ta98nwat7wrwzlf99.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=cloudsql%40pw-com.iam.gserviceaccount.com%2F20240213%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240213T201508Z&X-Goog-Expires=604800&X-Goog-SignedHeaders=host&X-Goog-Signature=58e0d92eca7344c36f2ab6a1b2d88815693359a54b96477b81dbebb22ae0bcf4d8c8fc5578d689b60e29f58875d301a19a20c25e71f0beec0b0ec47b2e343714540534f31dee2ce641f0f8c3c44ac05a1d72e92405ddd679f69a31ac8f8cb9c52cca7dd110c39177af9270de544b64d99fd0bc34aca23fa12ff25adee43eaf937953a01dc51786fb590cd447629510b27dc60b996f3b3224e3f1adbbfc056aa7c7c247eeb24476c3ac7ec72fea327484c84ecfcc352014ae1807a2a79bdd041352d82a3de67c9628c23ef324c200cccdfc3de78e1f05fd4f22193cafda40fe868b9a6a60067030609274d2e45db9ea2acf6c094c2549129945da2616907e72a2', 'identity_document_back': 'https://storage.googleapis.com/pw-com-private-bucket/identity-verification/sd3puqzjnv2ey_0syerlnt34j.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=cloudsql%40pw-com.iam.gserviceaccount.com%2F20240213%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240213T201508Z&X-Goog-Expires=604800&X-Goog-SignedHeaders=host&X-Goog-Signature=7b75cf90469641466465dce93d2e8542ffe11d36ffda6b0c01df41ab22d1d7a6545d11b1cfa4c1c47933b2c4f22701ad34b4d8f98e251210b0d7983e0a3400b9688f2d140968edc031534fd959e31bc16902d5a52d5c86e20591959b6df11e58f09c1e27f59dca79a193a3b5e34f15964a99f0269cb1f00d4245796af36a8aa7f1f6d7c56cfa6a1a5607969a630637d1227028cc533f2f43e826d65c4eec6d4a0ed6cc796ccb14cdd6c18e16d626b6862c8419d2f255faf4ab30615fb92e93a32577642ba6ae4386526aef58ed18ac4018b58493a8d4123014b8dcc8bae86eca90225d2cfaf2cbdd85002822b38ac334048167dd6cd08a6383cae45506e6e4a5'}
#Error in paste_rescaled_image_at_center: image file is truncated (26 bytes not processed)

#ITEM:  {'user_id': '13727577', 'verification_id': '103924', 'identity_document_type': '3', 'identity_document_front': 'https://storage.googleapis.com/pw-com-private-bucket/identity-verification/g78zalxhcaahirq2lhk1_4uk1.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=cloudsql%40pw-com.iam.gserviceaccount.com%2F20240213%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240213T204323Z&X-Goog-Expires=604800&X-Goog-SignedHeaders=host&X-Goog-Signature=1eb47dac67a46af6675c2e2f9417c3b9da72efed465df25bd8fac7d88a67acf4e2137995fda3f6bec9d1d9fdf924227abbcee2ac0006aff42ed375f2c49b16a361f4d7b5e0da557fa02e258731c342dc44d32427cbcc8b18a02411e651b1cd2328956f59e24e1304e966d2e7983a649c5c2dab0156383474fe3481b4efae19adc77386de74720a4acfb424004b988642618f4130cbc482c8aa380019249fcae3feda12399b40e5ba6c69cc956cac2c48d719a4c77ebacac2c93d45f7d5da8a508546f2010d34ed98b96071c6d913f49fd8c8bc27ae8e20b8fb81b544166f1010c0d53d47150918272a4221187d8d4d7a682fe7bbc72e790f3fda50f00e15db2a', 'identity_document_back': 'https://storage.googleapis.com/pw-com-private-bucket/identity-verification/wp00s3tvghw6cowmd4q_5d8xl.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=cloudsql%40pw-com.iam.gserviceaccount.com%2F20240213%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240213T204323Z&X-Goog-Expires=604800&X-Goog-SignedHeaders=host&X-Goog-Signature=6fd84d9499791e37f79a13d1580e632af3c0889180bf27401c7e4ab4d2703dab04d5399db63b52b5727e03b3ae0f39b9e495b6adc06f0e67b409a5b15c1336b8d40d0e3978420520ef6402f381f0d13fae26706028347da1c86d788639478722647f4ef7c7d06c6fac08f0280a6506fc07ab41de6835ae1386aed7e987475bda83332983d50c9e20778e826fad43e571174068cc14048d4b40300ee396c30cfe40fb9b2b016d322142946d48deb8e6e777240ba52198ddc648814d3e0d21655225b8e53f92005e48e7b2e80ff883946f219b05338f87b2507327c45d5bc8f4da39e3ce01d86cb315c2aae04c1a0c4e1aa549031cdca441ee56df33319cc1f28f'}
#Error in image validation: 'NoneType' object is not iterable

#ITEM:  {'user_id': '294599', 'verification_id': '103932', 'identity_document_type': '3', 'identity_document_front': 'https://storage.googleapis.com/pw-com-private-bucket/identity-verification/lt2s4tjn69ec5w01x7h9jahzd.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=cloudsql%40pw-com.iam.gserviceaccount.com%2F20240213%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240213T212603Z&X-Goog-Expires=604800&X-Goog-SignedHeaders=host&X-Goog-Signature=23ae38f6d933c9a209a8b746e655c57dde9c75032b6c874b474dba315a94d4c12325c619b99c2589b45676bb3d1400330fa5555cb4acf86755cfccbbef5779c09eff1c0245afdfbff9e438be57afa01f8715dd5d7762cec45996fa15ab98ded967e2d8d6340296ac1ffb06a9d5b2b67340faf1886866b812393e637c5fe1ccef938aed2382296981e1956b760fde1a06d45b6c4616781e98ca3a4175e2a9ea4d5fba92f4bbc1920c8ef7c8c8288a3824616db3d604973453b8935260063cce69e2a095b4543975585d181aa2c2314fe72bb3359684d53351dd1515661c0e6f1900281dc3a7c4223597f5cf0e5407254dee94c4bf7bf9496ae5d224089614960c', 'identity_document_back': 'https://storage.googleapis.com/pw-com-private-bucket/identity-verification/8dqnxf0r3nobeyo87lm7h2gfa.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=cloudsql%40pw-com.iam.gserviceaccount.com%2F20240213%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240213T212603Z&X-Goog-Expires=604800&X-Goog-SignedHeaders=host&X-Goog-Signature=62b89a46235ab807202403776ff1d600e61adfafd5da0bc2cbc1c8e397d69d66fc46165757edc54aae80095094ddc81932f76280130d8ca3edd365fd4ce9b4e796a0aa5671f84cb3f0cba98b542dfd9dfe59cbddea337d4aaab76d26cd0913e472943595e40e312f811103afa78ad8862f23b6260fea602631fbb0e2ea41a19f36c243b3659ef289a1efd5179111f5d07cdf64f24e55eba9e6c9c3798257800f7d66b2dc07f27e8d24e727276a8424a5f5fb81a9c7c95a3e06c6bbf44ce7dd86814f8b650642faf23968eb452f0bc4f66342848c34c609ba96e48443af2aeb9cf5fc3568fd09d0e25971a0b4373ae073ef943e258f563f03aed697cf55549291'}
#Error in detect_document: Image has incorrect shape

#ITEM:  {'user_id': '4408449', 'verification_id': '103934', 'identity_document_type': '2', 'identity_document_front': 'https://storage.googleapis.com/pw-com-private-bucket/identity-verification/aabm92m7mbv0p4su1p5k_6jqa.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=cloudsql%40pw-com.iam.gserviceaccount.com%2F20240213%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240213T212839Z&X-Goog-Expires=604800&X-Goog-SignedHeaders=host&X-Goog-Signature=85d150145334a65e1e7dd0b4fe9edc747dc5f73f56d1f144d81438327876046d6248646efa2eabe5ed3ad04a95e89c1b4531515a2e5a468e6f8d3a0133961b6458aed447bf6577c7915858d0a28ffe8db77f07548c15e09e69787783fd9ec21a61dfbb4d83e34482481c0249e168add9b0244649f116f6b605b254cfda01b18c46bc83449b3566803396c7d74a19d142a63c511b49c03f79b7a8abc94e9ee4f038b38abcde5ff0dbaf04471e6c6f3d57e0f6c6d949d4e261106ae29c890b570b0a3deee03b4a9c26b5a1f544e0f76d81bd2b9474b13cf4dd414089c45ccb61b0f6e07a1eb2416eda8d3eecc06e22fd614a14e46151093dcd666dda326267c2d5', 'identity_document_back': 'https://storage.googleapis.com/pw-com-private-bucket/identity-verification/a42c5t84qocnqoh1no48z_90g.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=cloudsql%40pw-com.iam.gserviceaccount.com%2F20240213%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240213T212839Z&X-Goog-Expires=604800&X-Goog-SignedHeaders=host&X-Goog-Signature=40d38e78f4ba890b14075cea0abbb982e867b6fec2752209c369254b0cfe77188a378ff5b5819b1bd1395ea546f59a5fda6ac628471af49591f859827dc06344449f42751d92dfe81235d6e20b2abd207ee74e8c78e8b5668ae7830fa9194b627ae5f3318855accde02f904a56a4d1d2bc4c9b14aca6a94d809fc45a95677519d44524828e6f8b4e266effa29d75810be5a09950df08959b22c9869ba50e5eeed2291b289a503b825e1097d0952402a6c1458cfa2d011dd905a03844699ec66380326a21c19a77421c8ccf6ca0ebfc755bb6c269a07e4aea7689c49087b00d10f01e7b3737fdd204d7727a6d01f7d845180a7f2fc3ff4a57c06e4553a17cd959'}
#Error in paste_rescaled_image_at_center: image file is truncated (9 bytes not processed)