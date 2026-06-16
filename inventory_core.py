#Classes e funções centrais para detecção e classificação de produtos

from ultralytics import YOLO
from collections import Counter

import cv2
import numpy as np
import os

import torch
import open_clip

from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

CATALOGO_FOLDER = "catalogo"
catalog_embeddings = {}


# ==========================
# MODELOS
# ==========================

model = YOLO("best.pt")

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

clip_model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = clip_model.to(device)



# ==========================
# FNC
# ==========================

def load_catalog():

    arquivos = [

        f for f in os.listdir(
            CATALOGO_FOLDER
        )

        if f.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png"
            )
        )
    ]

    for arquivo in arquivos:

        path = os.path.join(
            CATALOGO_FOLDER,
            arquivo
        )

        img = cv2.imread(path)

        if img is None:
            continue

        nome = os.path.splitext(
            arquivo
        )[0]

        emb = create_embedding(img)

        catalog_embeddings[nome] = emb

    print(
        f"Catálogo carregado: {len(catalog_embeddings)} produtos"
    )


# ==========================
# FEATURES
# ==========================

def create_embedding(crop):
    if crop.shape[0] < 10 or crop.shape[1] < 10:
        raise Exception(
            "Crop muito pequeno"
        )

    rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(rgb)

    image = clip_preprocess(
        image
    ).unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        features = clip_model.encode_image(
            image
        )

        features /= features.norm(
            dim=-1,
            keepdim=True
        )

    return features.cpu().numpy().flatten()

# ==========================
# DETECÇÃO
# ==========================

def detect_products(frame):

    crops = []
    boxes = []

    results = model.predict(
        frame,
        imgsz=640,
        conf=0.15,
        iou=0.5,
        verbose=False
    )

    total = 0

    for r in results:

        total += len(r.boxes)

        for box in r.boxes:

            conf = float(box.conf)

            if conf < 0.15:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            x1, y1, x2, y2 = map(
                int,
                [x1, y1, x2, y2]
            )

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            h = y2 - y1
            w = x2 - x1

            if h < 15 or w < 15:
                continue

            area = h * w

            img_area = (
                frame.shape[0] *
                frame.shape[1]
            )

            # ignora caixas gigantes
            if area > img_area * 0.20:
                continue

            crops.append(crop)

            boxes.append(
                (
                    x1,
                    y1,
                    x2,
                    y2
                )
            )

    print(f"YOLO detectou {len(crops)} objetos")

    return crops, boxes

def classify_product(crop):

    emb = create_embedding(crop)

    melhor_nome = None
    melhor_score = -1

    for nome, catalog_emb in catalog_embeddings.items():

        score = np.dot(
            emb,
            catalog_emb
        )

        if score > melhor_score:

            melhor_score = score
            melhor_nome = nome

    return melhor_nome, melhor_score


def get_color(label):

    seed = abs(hash(label))

    return (
        seed % 255,
        (seed // 255) % 255,
        (seed // 65025) % 255
    )