import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.models import Model
from skimage.feature import graycomatrix, graycoprops
import joblib

def load_cnn_models():
    cnn_models = {
        "MobileNetV2": MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
        "ResNet50": ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
        "EfficientNetB0": EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    }
    return {name: Model(inputs=model.input, outputs=model.layers[-2].output) for name, model in cnn_models.items()}

def extract_texture(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    distances = [1, 3, 5, 0, 0]
    angles = [0, 0, 0, np.pi/4, np.pi/2]
    features = []
    for d, a in zip(distances, angles):
        glcm = graycomatrix(gray, distances=[d], angles=[a], levels=256, symmetric=True, normed=True)
        features.extend([
            graycoprops(glcm, 'contrast')[0, 0],
            graycoprops(glcm, 'correlation')[0, 0],
            graycoprops(glcm, 'energy')[0, 0],
            graycoprops(glcm, 'homogeneity')[0, 0],
            graycoprops(glcm, 'dissimilarity')[0, 0]
        ])
    return np.array(features)

def extract_color_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return hist.flatten()

def extract_hu_moments(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments)
    return hu_moments.flatten()

def extract_all_features(image, cnn_models):
    image_resized = cv2.resize(image, (224, 224)) / 255.0
    image_resized = np.expand_dims(image_resized, axis=0)
    
    cnn_features = np.hstack([model.predict(image_resized).flatten() for model in cnn_models.values()])
    
    texture_features = extract_texture(image)
    color_features = extract_color_histogram(image)
    shape_features = extract_hu_moments(image)
    
    return np.concatenate([cnn_features, texture_features, color_features, shape_features])

def display_image_with_boxes(image_path, boxes):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for (x, y, w, h) in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return image

