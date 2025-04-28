from PIL import Image
import numpy as np

# def process_image(img_path, target_size=(224, 224)):
#     img = Image.open(img_path)
#     img = img.resize(target_size)

#     img_array = np.array(img)

#     img_array = np.expand_dims(img_array, axis=0)  

#     return img_array

def process_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    return np.expand_dims(img_array, axis=0)
