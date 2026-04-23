import numpy as np
from PIL import Image

def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """
    Preprocess the uploaded image for model prediction.
    
    Args:
        image (PIL.Image): The uploaded image.
        target_size (tuple): Target size for resizing (width, height). Default matches model training size (224, 224).
        
    Returns:
        np.ndarray: Preprocessed image array with shape (1, height, width, 3).
    """
    # 1. Convert to RGB (ensure 3 channels)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # 2. Resize image to match training size
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # 3. Convert to numpy array
    img_array = np.array(image)
    
    # 4. Normalize pixel values to range [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    # 5. Expand dimensions to match model input shape (batch_size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array
