import streamlit as st
import numpy as np
import cv2
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"  # Désactive OpenEXR qui peut causer des problèmes
import cv2
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
from utils import load_cnn_models, extract_all_features, display_image_with_boxes

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
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cabosse = image[y:y+h, x:x+w]
        
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
