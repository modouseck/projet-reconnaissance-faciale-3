#!/usr/bin/env python3
"""face.py - extract embeddings with keras-facenet, train an SVC, and save model.

Minimal, single-file clean replacement.
Usage: python face.py [folder]
"""

import os
import re
import sys
import numpy as np
import joblib
import argparse
from keras_facenet import FaceNet
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import load_img

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

from sklearn.svm import SVC

embedder = FaceNet()


def _preprocess_img_path(path):
    img = load_img(path, target_size=(160, 160))
    arr = img_to_array(img).astype('float32')
    arr = (arr - 127.5) / 128.0
    return arr


def load_image_paths_and_labels(folder):
    # Previously this listed only the top-level folder. Now walk recursively and
    # use the containing subfolder name (relative to the root folder) as label
    # when available. This handles folders with spaces and nested per-person dirs.
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.heic', '.gif'}
    items = []
    folder = os.path.abspath(folder)
    for dirpath, _, filenames in os.walk(folder):
        rel = os.path.relpath(dirpath, start=folder)
        # if images are directly in the root folder rel == '.' -> use filename base
        for fname in sorted(filenames):
            _, ext = os.path.splitext(fname)
            if ext.lower() not in exts:
                continue
            full = os.path.join(dirpath, fname)
            if rel == '.':
                base = os.path.splitext(fname)[0].lower()
                label_parts = re.sub(r'[^\w\s]', ' ', base).strip().split()
                label = label_parts[0] if label_parts else 'unknown'
            else:
                # use the last component of the relative path as the label
                dir_label = os.path.basename(rel).lower()
                label_parts = re.sub(r'[^\w\s]', ' ', dir_label).strip().split()
                label = label_parts[0] if label_parts else 'unknown'
            items.append((full, label))
    return items


def extract_embeddings(items, batch_size=16):
    paths = [p for p, _ in items]
    labels = [l for _, l in items]
    embeddings = []
    final_labels = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i+batch_size]
        imgs = []
        keep_labels = []
        for p, l in zip(batch_paths, labels[i:i+batch_size]):
            try:
                arr = _preprocess_img_path(p)
                imgs.append(arr)
                keep_labels.append(l)
            except Exception as e:
                print('skip', p, e)
        if not imgs:
            continue
        batch_array = np.stack(imgs, axis=0)
        emb = embedder.embeddings(batch_array)
        for k in range(len(keep_labels)):
            embeddings.append(emb[k])
            final_labels.append(keep_labels[k])
    if not embeddings:
        return np.array([]), np.array([])
    return np.vstack(embeddings), np.array(final_labels)


def train_and_save(folder):
    # Normalize the folder path and load items recursively.
    folder = os.path.abspath(os.path.expanduser(folder))
    items = load_image_paths_and_labels(folder)
    if not items:
        print('No images found in', folder)
        return
    X, y = extract_embeddings(items)
    if X.size == 0:
        print('No embeddings could be extracted')
        return
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2:
        print('Need at least 2 classes to train (found {})'.format(len(unique)))
        return
    # show simple counts
    print('Found {} images in {} classes'.format(len(y), len(unique)))
    for u, c in zip(unique, counts):
        print(' - {}: {}'.format(u, c))
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X, y)
    out = os.path.join(folder, 'face_svc.joblib')
    joblib.dump({'model': clf, 'labels': list(np.unique(y))}, out)
    print('Saved model to', out)


# new helper to normalize user-provided folder strings
def normalize_folder_string(folder_arg):
    # strip surrounding quotes, unescape backslash-escaped chars, expand ~, make absolute
    s = folder_arg.strip().strip('\'"')
    s = re.sub(r'\\(.)', r'\1', s)
    s = os.path.expanduser(s)
    return os.path.abspath(s)


if __name__ == '__main__':
    # Use argparse so paths with spaces are handled clearly.
    parser = argparse.ArgumentParser(description='Train face SVC from images folder')
    parser.add_argument('folder', nargs='?',
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help='Images folder (default: script directory)')
    args = parser.parse_args()
    folder = normalize_folder_string(args.folder)
    if not os.path.isdir(folder):
        print('Folder not found or not a directory:', folder)
        sys.exit(1)
    print('Using folder:', folder)
    try:
        train_and_save(folder)
    except Exception as e:
        print('Error training model:', e)
        sys.exit(1)
