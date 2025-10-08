# app.py
# 🚀 Reconnaissance faciale avec DeepFace (compatible Streamlit Cloud)
# Auteur : Cathy 🌸

import streamlit as st
from deepface import DeepFace
import os
from PIL import Image

st.set_page_config(page_title="Reconnaissance Faciale - mouhamed:ami", layout="centered")
st.title("🔍 Application de Reconnaissance Faciale (Mouhamed & Friends)")

# === Étape 1 : Upload d'une image de référence ===
st.header("📸 Étape 1 : Téléverse les photos de référence")
uploaded_folder = "/Users/User/Desktop/face/face mouhamed:ami"
os.makedirs(uploaded_folder, exist_ok=True)

uploaded_files = st.file_uploader(
    "Téléverse plusieurs images (nomme-les ex : mouhamed_1.jpg, ami_1.jpg...)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for f in uploaded_files:
        with open(os.path.join(uploaded_folder, f.name), "wb") as file:
            file.write(f.read())
    st.success(f"{len(uploaded_files)} image(s) enregistrée(s) dans le dossier `faces/` ✅")

# === Étape 2 : Upload d'une image à tester ===
st.header("🤖 Étape 2 : Téléverse une image à reconnaître")
test_image = st.file_uploader("Choisis une image de test", type=["jpg", "jpeg", "png"])

if test_image:
    test_path = "test_image.jpg"
    with open(test_path, "wb") as file:
        file.write(test_image.read())
    image = Image.open(test_path)
    st.image(image, caption="🧠 Image test à reconnaître", use_column_width=True)

    # === Étape 3 : Exécution du modèle ===
    st.header("⚙️ Étape 3 : Résultat de la reconnaissance")
    try:
        result = DeepFace.find(img_path=test_path, db_path=uploaded_folder, model_name=["Facenet", "VGG-Face", "OpenFace", "DeepFace"])
        if len(result) > 0 and not result[0].empty:
            matched_img = result[0].iloc[0]['identity']
            distance = result[0].iloc[0]['Facenet_cosine']
            st.success(f"✅ Visage reconnu : **{os.path.basename(matched_img)}** (distance : {distance:.4f})")
        else:
            st.warning("Aucun visage correspondant trouvé 😕")
    except Exception as e:
        st.error(f"Erreur lors de la reconnaissance : {e}")