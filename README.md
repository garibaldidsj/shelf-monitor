# Shelf Inventory Monitor

Sistema de monitoramento de estoque em prateleiras utilizando:

* YOLO para detecção de produtos
* CLIP para classificação baseada em catálogo
* OpenCV para processamento de imagem
* Python

## Funcionalidades

* Contagem automática de produtos
* Classificação por catálogo de imagens
* Processamento de imagens estáticas
* Monitoramento em vídeo ao vivo
* Detecção de produtos desconhecidos

## Estrutura

catalogo/

* imagens de referência dos produtos

teste/

* imagens de teste

resultado/

* resultados processados

## Execução

Imagem:

python image_counter.py

Vídeo:

python live_counter.py
