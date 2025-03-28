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




import streamlit as st
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
from utils import load_cnn_models, extract_all_features

# Chargement des modèles CNN
cnn_models = load_cnn_models()

# Initialisation de l'état de session
if 'segmented_pods' not in st.session_state:
    st.session_state.segmented_pods = {}

st.title("Détection et classification des maladies du cacaoyer")

# 1. Affichage des images sous forme de grille
image_files = [f for f in os.listdir("Images") if f.endswith(".jpg")]
st.subheader("Images disponibles")
cols = st.columns(4)
for i, img_file in enumerate(image_files):
    img = Image.open(os.path.join("Images", img_file))
    cols[i % 4].image(img, caption=img_file, use_container_width=True)

# 2. Sélection d'une image
selected_image = st.selectbox("Choisissez une image", image_files)
if selected_image:
    image_path = os.path.join("Images", selected_image)
    txt_path = image_path.replace(".jpg", ".txt")
    
    with open(txt_path, "r") as f:
        boxes = [list(map(lambda x: int(float(x)), line.strip().split())) for line in f]
    
    # Affichage de l'image avec boîtes
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    for box in boxes:
        x, y, w, h = box
        draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
    st.image(img, caption="Détection des cabosses", use_container_width=True)
    
    # 3. Segmentation de toutes les cabosses
    if st.button("Segmenter toutes les cabosses"):
        segmented_pods = []
        for i, (x, y, w, h) in enumerate(boxes, start=1):
            pod = img.crop((x, y, x + w, y + h))
            segmented_pods.append(pod)
        st.session_state.segmented_pods[selected_image] = segmented_pods
    
    # Affichage des cabosses segmentées
    if selected_image in st.session_state.segmented_pods:
        st.subheader("Cabosses segmentées")
        pod_cols = st.columns(4)
        for i, pod in enumerate(st.session_state.segmented_pods[selected_image]):
            pod_cols[i % 4].image(pod, caption=f"Cabosse {i+1}", use_container_width=True)
    
    # 4. Prédiction
    if selected_image in st.session_state.segmented_pods:
        selected_pod_idx = st.selectbox("Sélectionnez une cabosse", range(1, len(st.session_state.segmented_pods[selected_image]) + 1))
        if st.button("Prédire la maladie"):
            pod = st.session_state.segmented_pods[selected_image][selected_pod_idx - 1]
            features = extract_all_features(np.array(pod), cnn_models)
            pca = PCA(n_components=0.99)
            reduced_features = pca.fit_transform(features.reshape(1, -1))
            model = joblib.load("disease_classifier.pkl")
            prediction = model.predict(reduced_features)[0]
            st.write(f"**Maladie prédite :** {prediction}")

