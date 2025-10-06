# embeddings.py
# Génère embeddings FaceNet pour chaque image dans le dossier "faces/"
# Utilise MTCNN pour détecter / recadrer le visage avant embedding

import os
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from keras_facenet import FaceNet
from tensorflow.keras.preprocessing.image import img_to_array

# Use the requested path exactly as given by the user (FACE_REF).
# If it's a file, the script will use its parent directory as the images folder.
FACE_REF = "/Users/User/Desktop/face/face mouhamed:ami.py"

def _resolve_folder(path):
    if os.path.isdir(path):
        return path
    return os.path.dirname(path) or path

IMAGES_DIR = _resolve_folder(FACE_REF)
EMBEDDINGS_OUT = os.path.join(IMAGES_DIR, "faces_embeddings.npz")
MODEL_OUT = os.path.join(IMAGES_DIR, "best_model.pkl")

detector = MTCNN()
embedder = FaceNet()

def extract_face(image_path, required_size=(160,160)):
    """Detecte le visage avec MTCNN, recadre et redimensionne."""
    image = Image.open(image_path).convert('RGB')
    pixels = np.asarray(image)
    results = detector.detect_faces(pixels)
    if not results:
        raise ValueError(f"Aucun visage détecté dans {image_path}")
    # prendre le premier visage détecté
    x, y, w, h = results[0]['box']
    x, y = abs(x), abs(y)
    face = pixels[y:y+h, x:x+w]
    face_image = Image.fromarray(face).resize(required_size)
    face_array = img_to_array(face_image)
    return face_array

def get_embedding_from_array(face_array):
    """Prend un array (160,160,3) et renvoie l'embedding 512D."""
    img = np.expand_dims(face_array, axis=0)
    img = (img - 127.5) / 128.0
    return embedder.embeddings(img)[0]

def get_embedding(image_path):
    """Pipeline complet : detect -> crop -> embed"""
    face = extract_face(image_path)
    return get_embedding_from_array(face)

def build_dataset(folder=IMAGES_DIR, out_file=EMBEDDINGS_OUT):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Dossier d'images introuvable: {folder}")
    X, y, names = [], [], []
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder, fname)
            try:
                emb = get_embedding(path)
                label = fname.split("_")[0]   # convention : label_index.jpg
                X.append(emb)
                y.append(label)
                names.append(fname)
                print(f"✅ Embedding pour {fname} (label={label})")
            except Exception as e:
                print(f"⚠️ Skipped {fname} : {e}")
    X = np.array(X)
    y = np.array(y)
    np.savez(out_file, X=X, y=y, names=np.array(names))
    print(f"\n📂 Embeddings sauvegardés dans '{out_file}' -> X:{X.shape}, labels:{len(y)}")
    return X, y

# ...existing code (training / evaluation)...
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def load_embeddings(file=EMBEDDINGS_OUT):
    if not os.path.exists(file):
        raise FileNotFoundError(f"Fichier d'embeddings introuvable: {file}")
    data = np.load(file, allow_pickle=True)
    return data["X"], data["y"], data.get("names")

def evaluate_and_save_best(X, y, save_path=MODEL_OUT):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    models = {
        "SVM": SVC(kernel="linear", probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
    }

    best_model = None
    best_score = -1
    best_name = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"🔸 {name} accuracy: {acc:.4f}")
        if acc > best_score:
            best_score = acc
            best_model = model
            best_name = name

    # Evaluation détaillée du meilleur
    y_pred = best_model.predict(X_test)
    print(f"\n🎯 Meilleur modèle: {best_name} (accuracy={best_score:.4f})")
    print("\n--- Classification report ---")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print("\n--- Matrice de confusion ---")
    print(cm)

    # Sauvegarder le modèle
    joblib.dump(best_model, save_path)
    print(f"\n✅ Modèle sauvegardé dans {save_path}")

    # Sauvegarder figure matrice de confusion
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title(f"Matrice de confusion - {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "confusion_matrix.png"))
    print("📊 Matrice de confusion sauvegardée : confusion_matrix.png")

if __name__ == "__main__":
    # 1) Générer embeddings depuis le dossier d'images (IMAGES_DIR)
    X, y = build_dataset(folder=IMAGES_DIR, out_file=EMBEDDINGS_OUT)

    # 2) Charger embeddings et entraîner/évaluer / sauvegarder le meilleur modèle
    X_loaded, y_loaded, names = load_embeddings(EMBEDDINGS_OUT)
    evaluate_and_save_best(X_loaded, y_loaded, save_path=MODEL_OUT)
