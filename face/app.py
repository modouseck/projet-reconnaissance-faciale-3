# app.py
# Application Streamlit simple : uploader une image -> prédiction -> affichage
# Règle 'unknown' si probabilité max < threshold

import streamlit as st
import numpy as np
import joblib
from PIL import Image
from mtcnn import MTCNN
from keras_facenet import FaceNet
from tensorflow.keras.preprocessing.image import img_to_array
import tempfile
from pathlib import Path
import os
import io

# Use the requested folder path for models/images as configured default
FACE_DIR = "/Users/User/Desktop/face/face mouhamed:ami.py"

st.set_page_config(page_title="Face Recognition (Mouhamed & Friends)", layout="centered")
st.title("🔎 Reconnaissance Faciale (Mouhamed & Friends)")

# Charger ressources (detector + embedder only)
@st.cache_resource
def load_models():
    detector = MTCNN()
    embedder = FaceNet()
    return detector, embedder

detector, embedder = load_models()

# Helper functions for model discovery/loading
def find_model_files(models_dir: Path):
    patterns = ["*.joblib", "*.pkl", "*.sav"]
    files = []
    for pat in patterns:
        files.extend(sorted(models_dir.glob(pat)))
    return files

def load_classifier_from_path(path: Path):
    try:
        clf = joblib.load(str(path))
        return clf, None
    except Exception as e:
        return None, str(e)

def load_classifier_from_uploaded(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp.flush()
            tmp_path = tmp.name
        clf = joblib.load(tmp_path)
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return clf, None
    except Exception as e:
        return None, str(e)

def extract_face_from_pil(pil_image, required_size=(160,160)):
    pixels = np.asarray(pil_image.convert('RGB'))
    results = detector.detect_faces(pixels)
    if not results:
        return None
    x, y, w, h = results[0]['box']
    x, y = abs(x), abs(y)
    face = pixels[y:y+h, x:x+w]
    face_img = Image.fromarray(face).resize(required_size)
    return img_to_array(face_img)

def get_embedding_from_array(face_array):
    img = np.expand_dims(face_array, axis=0)
    img = (img - 127.5) / 128.0
    return embedder.embeddings(img)[0]

# Model selection UI (use FACE_DIR as default folder)
st.sidebar.header("Modèle de classification")
uploaded_model = st.sidebar.file_uploader("Charger un modèle (.pkl, .joblib, .sav)", type=["pkl","joblib","sav"])
models_dir = Path(FACE_DIR)
available_models = []
selected_model_path = None
if uploaded_model is None:
    if models_dir.exists() and models_dir.is_dir():
        available_models = find_model_files(models_dir)
    # if FACE_DIR is not a directory, no files will be found
    if available_models:
        model_names = [p.name for p in available_models]
        sel = st.sidebar.selectbox("Choisir un modèle trouvé", ["(aucun)"] + model_names)
        if sel != "(aucun)":
            selected_model_path = models_dir / sel
    else:
        st.sidebar.info(f"Aucun modèle trouvé dans {models_dir}. Tu peux téléverser un modèle via le menu.")
else:
    # will load uploaded_model later
    pass

# Try to load classifier
clf = None
load_error = None
if uploaded_model is not None:
    clf, load_error = load_classifier_from_uploaded(uploaded_model)
elif selected_model_path is not None:
    clf, load_error = load_classifier_from_path(selected_model_path)

if clf is None:
    st.sidebar.error("Aucun modèle chargé." + (f" Erreur: {load_error}" if load_error else ""))
else:
    st.sidebar.success("Modèle chargé.")

uploaded_file = st.file_uploader("Téléverse une image (jpg, png)", type=["jpg","jpeg","png"])

threshold = st.slider("Seuil de confiance pour 'inconnu' (probabilité max)", 0.0, 1.0, 0.6)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    caption = getattr(uploaded_file, "name", "Image chargée")
    st.image(image, caption=caption, use_column_width=True)
    if clf is None:
        st.error("Aucun modèle de classification disponible. Téléverse ou sélectionne un modèle via la barre latérale.")
    else:
        face_array = extract_face_from_pil(image)
        if face_array is None:
            st.error("Aucun visage détecté. Essaie une autre photo (visage visible).")
        else:
            emb = get_embedding_from_array(face_array)
            try:
                # some classifiers may not implement predict_proba
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba([emb])[0]
                    best_idx = np.argmax(probs)
                    best_prob = probs[best_idx]
                    pred_label = clf.classes_[best_idx]
                else:
                    pred_label = clf.predict([emb])[0]
                    best_prob = 1.0
                if best_prob < threshold:
                    st.warning(f"Visage inconnu (confiance max={best_prob:.2f}).")
                else:
                    st.success(f"Prédit : {pred_label} (confiance {best_prob:.2f})")
                # afficher top 3 prédictions si disponibles
                if hasattr(clf, "predict_proba"):
                    top3_idx = np.argsort(probs)[-3:][::-1]
                    st.write("Top 3 prédictions :")
                    for i in top3_idx:
                        st.write(f"- {clf.classes_[i]} : {probs[i]:.3f}")
            except Exception as e:
                st.error(f"Erreur lors de la prédiction: {e}")
