
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
import numpy as np
from PIL import Image
import os
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA

# Chargement des modèles CNN
cnn_models = load_cnn_models()

# Interface Streamlit
st.title("Détection et classification des maladies du cacaoyer")

# Sélection de l'image
image_files = [f for f in os.listdir("Images") if f.endswith(".jpg")]
selected_image = st.selectbox("Choisissez une image", image_files)

if selected_image:
    image_path = os.path.join("Images", selected_image)
    txt_path = image_path.replace(".jpg", ".txt")
    
    # Chargement des boîtes englobantes
    with open(txt_path, "r") as f:
        boxes = [list(map(int, line.strip().split())) for line in f]
    
    # Affichage de l'image avec boîtes
    image_with_boxes = display_image_with_boxes(image_path, boxes)
    st.image(image_with_boxes, caption="Détection des cabosses", use_column_width=True)
    
    # Sélection d'une cabosse
    selected_box = st.selectbox("Sélectionnez une cabosse", range(len(boxes)))
    
    if st.button("Segmenter la cabosse"):
        x, y, w, h = boxes[selected_box]
        
        # Ouvrir l'image avec PIL
        image = Image.open(image_path)
        image = image.convert("RGB")
        image_array = np.array(image)  # Convertir en tableau numpy
        
        # Extraire la cabosse
        cabosse = image_array[y:y+h, x:x+w]
        
        # Extraction des caractéristiques
        all_features = extract_all_features(cabosse, cnn_models)
        
        # Réduction avec ACP
        pca = PCA(n_components=0.99)
        reduced_features = pca.fit_transform(all_features.reshape(1, -1))
        
        # Prédiction
        model = joblib.load("modele_svm.pkl")  # Charger le modèle de classification
        prediction = model.predict(reduced_features)[0]
        
        # Affichage du résultat
        st.image(cabosse, caption="Cabosse segmentée", use_column_width=True)
        st.write(f"**Maladie prédite :** {prediction}")
