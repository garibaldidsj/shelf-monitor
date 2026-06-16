### Contagem em tempo real usando webcam
from inventory_core import *

import time

# ==========================
# CONFIG
# ==========================


CAMERA_SOURCE = 0

PROCESS_INTERVAL = 10

CONFIDENCE_THRESHOLD = 0.40

# ==========================
# INICIALIZA
# ==========================

print("Carregando catálogo...")

load_catalog()

cap = cv2.VideoCapture(
    CAMERA_SOURCE
)

if not cap.isOpened():

    raise Exception(
        "Não foi possível abrir a câmera"
    )

last_process = 0

last_boxes = []
last_labels = []
last_counts = Counter()

# ==========================
# LOOP
# ==========================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    now = time.time()

    # ==========================
    # PROCESSA A CADA X SEGUNDOS
    # ==========================

    if now - last_process > PROCESS_INTERVAL:

        crops, boxes = detect_products(
            frame
        )

        labels = []

        counts = Counter()

        for crop in crops:

            try:

                nome, score = classify_product(
                    crop
                )

                if score >= CONFIDENCE_THRESHOLD:

                    labels.append(nome)

                    counts[nome] += 1

                else:

                    labels.append(
                        "desconhecido"
                    )

                    counts[
                        "desconhecido"
                    ] += 1

            except Exception as e:

                print(
                    "Erro:",
                    e
                )

                labels.append(
                    "erro"
                )

                counts[
                    "erro"
                ] += 1

        last_boxes = boxes
        last_labels = labels
        last_counts = counts

        last_process = now

        print("\n====================")

        for produto, qtd in counts.items():

            print(
                f"{produto}: {qtd}"
            )

    # ==========================
    # DESENHA RESULTADOS
    # ==========================

    for idx, box in enumerate(
        last_boxes
    ):

        if idx >= len(last_labels):
            continue

        x1, y1, x2, y2 = box

        label = last_labels[idx]

        color = get_color(label)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    y = 30

    for produto, qtd in last_counts.items():

        cv2.putText(
            frame,
            f"{produto}: {qtd}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        y += 30

    cv2.imshow(
        "Inventory Monitor",
        frame
    )

    key = cv2.waitKey(1)

    if key == 27:
        break

# ==========================
# FINALIZA
# ==========================

cap.release()

cv2.destroyAllWindows()