#Contagem de produtos em imagens estáticas
#processa pasta de teste e salva resultados em resultado/

from inventory_core import *

# ==========================
# CONFIGURAÇÕES
# ==========================
CATALOGO_FOLDER = "catalogo"
catalog_embeddings = {}

INPUT_FOLDER = "teste"
OUTPUT_FOLDER = "resultado"
DEBUG_CROPS = False

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================
# PROCESSAMENTO
# ==========================

def process_image(path):
    
    frame = cv2.imread(path)

    if frame is None:
        return None, None

    crops, boxes = detect_products(frame)

    if DEBUG_CROPS:
        os.makedirs(
            "resultado/crops",
            exist_ok=True
        )
        for idx, crop in enumerate(crops):
            cv2.imwrite(
                f"resultado/crops/{idx}.jpg",
                crop
            )

    print(
        f"Objetos detectados: {len(crops)}"
    )

    if len(crops) == 0:

        return frame, {}

    counts = Counter()

    
    labels = []
    for crop in crops:

        try:

            nome, score = classify_product(
                crop
            )
            print(
             f"{nome} -> {score:.3f}"
            )
            # Ajuste esse valor conforme os testes
            if score > 0.35:

                counts[nome] += 1

                labels.append(nome)

            else:

                labels.append(
                    "desconhecido"
                )

                counts[
                    "desconhecido"
                ] += 1

        except Exception as e:

            print(
                "Erro classificação:",
                e
            )

            labels.append(
                "erro"
            )

            counts[
                "erro"
            ] += 1

    print("\nResultado:")

    for produto, qtd in counts.items():

        print(
            f"{produto}: {qtd}"
        )

    colors = {}
    for produto in counts.keys():
        colors[produto] = get_color(produto)
                

    for idx, box in enumerate(boxes):

        if idx >= len(labels):
            continue

        x1, y1, x2, y2 = box

        label = labels[idx]

        color = colors[label]

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

    for produto, qtd in counts.items():

        cv2.putText(
            frame,
            f"{produto}: {qtd}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        y += 30

    return frame, counts

# ==========================
# PROCESSAR PASTA
# ==========================

def process_folder():

    resultados_csv = []

    extensoes = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    )

    arquivos = [

        f for f in os.listdir(
            INPUT_FOLDER
        )

        if f.lower().endswith(
            extensoes
        )
    ]

    print(
        f"\nEncontradas {len(arquivos)} imagens"
    )

    for arquivo in arquivos:

        print(
            "\n===================="
        )

        print(
            f"Processando: {arquivo}"
        )

        caminho = os.path.join(
            INPUT_FOLDER,
            arquivo
        )

        try:

            imagem_resultado, counts = process_image(
                caminho
            )

            if imagem_resultado is None:
                continue

            saida = os.path.join(
                OUTPUT_FOLDER,
                arquivo
            )

            cv2.imwrite(
                saida,
                imagem_resultado
            )

            for grupo, qtd in counts.items():

                resultados_csv.append(
                    {
                        "arquivo": arquivo,
                        "grupo": grupo,
                        "quantidade": qtd
                    }
                )

        except Exception as e:

            print(
                f"Erro em {arquivo}: {e}"
            )

    if len(resultados_csv):

        df = pd.DataFrame(
            resultados_csv
        )

        df.to_csv(
            os.path.join(
                OUTPUT_FOLDER,
                "contagem.csv"
            ),
            index=False
        )

        print(
            "\nCSV salvo com sucesso."
        )

# ==========================
# EXECUÇÃO
# ==========================
load_catalog()

print("Produtos carregados:")
print(list(catalog_embeddings.keys()))

process_folder()