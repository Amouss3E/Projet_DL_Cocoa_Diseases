import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.models import Model
from skimage.feature import graycomatrix, graycoprops

def load_cnn_models():
    cnn_models = {
        "MobileNetV2": MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
        "ResNet50": ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
        "EfficientNetB0": EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    }
    return {name: Model(inputs=model.input, outputs=model.layers[-2].output) for name, model in cnn_models.items()}

def extract_texture(image):
    gray = image.convert("L")  # Convertir en niveaux de gris
    gray_np = np.array(gray)

    distances = [1, 3, 5, 0, 0]
    angles = [0, 0, 0, np.pi/4, np.pi/2]
    features = []

    for d, a in zip(distances, angles):
        glcm = graycomatrix(gray_np, distances=[d], angles=[a], levels=256, symmetric=True, normed=True)
        features.extend([
            graycoprops(glcm, 'contrast')[0, 0],
            graycoprops(glcm, 'correlation')[0, 0],
            graycoprops(glcm, 'energy')[0, 0],
            graycoprops(glcm, 'homogeneity')[0, 0],
            graycoprops(glcm, 'dissimilarity')[0, 0]
        ])
    return np.array(features)

def extract_color_histogram(image):
    image_np = np.array(image)
    hist = np.histogram(image_np.flatten(), bins=256, range=[0, 256])[0]
    return hist.flatten()

def extract_hu_moments(image):
    gray = image.convert("L")  # Convertir en niveaux de gris
    gray_np = np.array(gray)
    moments = tf.image.moments(gray_np, axes=[0, 1])
    hu_moments = moments.central_moments.numpy().flatten()
    return hu_moments

def extract_all_features(image, cnn_models):
    image_resized = image.resize((224, 224))
    image_np = np.array(image_resized) / 255.0
    image_np = np.expand_dims(image_np, axis=0)

    cnn_features = np.hstack([model.predict(image_np).flatten() for model in cnn_models.values()])

    texture_features = extract_texture(image)
    color_features = extract_color_histogram(image)
    shape_features = extract_hu_moments(image)

    return np.concatenate([cnn_features, texture_features, color_features, shape_features])

def display_image_with_boxes(image_path, boxes):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for (x, y, w, h) in boxes:
        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
    return image
