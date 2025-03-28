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
from PIL import Image, ImageDraw
import joblib
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA

if "segmented_pods" not in st.session_state:
    st.session_state.segmented_pods = {}
    
# Chemin du dossier contenant les images et annotations
IMAGE_DIR = "Images"

# Fonction pour charger les images et leurs annotations
def load_data(directory):
    images, bboxes, filenames = [], [], []
    for file in os.listdir(directory):
        if file.endswith(".jpg"):
            img_path = os.path.join(directory, file)
            txt_path = img_path.replace(".jpg", ".txt")

            # Charger l'image
            image = Image.open(img_path).convert("RGB")
            images.append(image)

            # Charger les annotations (boîtes englobantes)
            if os.path.exists(txt_path):
                with open(txt_path, "r") as f:
                    annotations = [list(map(float, line.split())) for line in f.readlines()]
                    bboxes.append(annotations)
            else:
                bboxes.append([])  # Pas d'annotations pour cette image

            filenames.append(file)

    return images, bboxes, filenames

# Charger les images et annotations
images, bboxes, filenames = load_data(IMAGE_DIR)

# Interface Streamlit
st.title("Détection et classification des maladies du cacaoyer")

### **1. Affichage en grille des images**
st.subheader("Images disponibles")
cols = st.columns(4)
for i, file in enumerate(filenames):
    with cols[i % 4]:
        st.image(images[i], caption=file, use_container_width=True)

### **2. Sélection d'une image et affichage des boîtes**
st.subheader("Détection des cabosses")

# Ajout d'une option vide au selectbox
selected_image = st.selectbox("Choisissez une image", ["Images"] + filenames)

# N'afficher l'image que si une vraie image a été choisie
if selected_image != "Sélectionnez une image":
    index = filenames.index(selected_image)
    image = images[index].copy()  # Copie pour éviter de modifier l'originale
    draw = ImageDraw.Draw(image)

    # Dessiner les boîtes englobantes
    for bbox in bboxes[index]:
        x, y, w, h = bbox
        x1, y1 = (x - w/2) * image.width, (y - h/2) * image.height
        x2, y2 = (x + w/2) * image.width, (y + h/2) * image.height
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    # Affichage de l'image avec boîtes
    st.image(image, use_column_width=True)

### **3. Segmentation des cabosses**
if st.button("Segmenter toutes les cabosses"):
    segmented_pods = []
    
    for bbox in bboxes[index]:
        x, y, w, h = bbox
        x1, y1 = int((x - w/2) * image.width), int((y - h/2) * image.height)
        x2, y2 = int((x + w/2) * image.width), int((y + h/2) * image.height)
        pod = images[index].crop((x1, y1, x2, y2))
        segmented_pods.append(pod)

    st.session_state.segmented_pods = {selected_image: segmented_pods}

# Affichage des cabosses segmentées en grille
if selected_image in st.session_state.get("segmented_pods", {}):
    st.subheader("Cabosses segmentées")
    pod_cols = st.columns(4)
    for i, pod in enumerate(st.session_state.segmented_pods[selected_image]):
        pod_cols[i % 4].image(pod, caption=f"Cabosse {i+1}", use_container_width=True)

### **4. Prédiction**
if selected_image in st.session_state.segmented_pods:
    st.subheader("Prédiction de la maladie")
    selected_pod = st.selectbox("Choisissez une cabosse", range(1, len(st.session_state.segmented_pods[selected_image]) + 1))

    if st.button("Prédire la maladie"):
        # Chargement des modèles CNN
        cnn_models = load_cnn_models()
        
        # Extraction des caractéristiques
        pod = st.session_state.segmented_pods[selected_image][selected_pod - 1]
        features = extract_all_features(pod, cnn_models)
        
        # Réduction avec ACP
        pca = PCA(n_components=0.99)
        reduced_features = pca.fit_transform(features.reshape(1, -1))
        
        # Prédiction avec le modèle SVM
        model = joblib.load("disease_classifier.pkl")
        prediction = model.predict(reduced_features)[0]

        # Affichage du résultat
        st.image(pod, caption=f"Cabosse {selected_pod}", use_container_width=True)
        st.write(f"**Maladie prédite :** {prediction}")
