# `gs-wildfire-vision: Deteccao de Incendios por Computer Vision`

> EfficientNetB0 fine-tuned para deteccao de incendios florestais em imagens de satelite. Transfer learning com ImageNet, data augmentation, analise completa com curvas ROC e matriz de confusao. GS 2026.1, FIAP.

---

## `Tecnologias`

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00)
![EfficientNet](https://img.shields.io/badge/EfficientNetB0-transfer%20learning-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-vision-green)
![Colab](https://img.shields.io/badge/Google%20Colab-GPU-F9AB00)
![License](https://img.shields.io/badge/license-MIT-green)

---

## `O que faz`

Classifica imagens de satelite em `wildfire` (incendio ativo) e `nowildfire` (sem incendio):

- **Transfer learning**: EfficientNetB0 pre-treinado em ImageNet, camadas superiores congeladas
- **Fine-tuning**: desbloqueio progressivo das camadas finais do backbone para adaptacao ao dominio
- **Data augmentation**: rotacao, flip, zoom, ajuste de brilho para robustez
- **Avaliacao**: curva ROC, AUC, matriz de confusao, curvas de aprendizado

---

## `Pipeline`

```
Dataset Kaggle (imagens RGB 256x256)
    |
    Data augmentation: ImageDataGenerator
    |
    EfficientNetB0 (backbone ImageNet, congelado)
        +-- GlobalAveragePooling2D
        +-- Dense(256, relu) + Dropout(0.5)
        +-- Dense(1, sigmoid)  → P(wildfire)
    |
    Phase 1: treina apenas head (50 epochs)
    Phase 2: fine-tune backbone (desbloqueio parcial, lr reduzido)
    |
    Avaliacao: ROC · AUC · Confusion Matrix · Classification Report
```

---

## `Resultados`

| Metrica | Fase 1 | Fase 2 (fine-tune) |
|---------|--------|--------------------|
| Acuracia | ver notebook | ver notebook |
| AUC-ROC | ver notebook | ver notebook |

Graficos completos em `curvas_treinamento.png`, `curva_roc.png` e `matriz_confusao.png`.

---

## `Arquivos`

| Arquivo | Descricao |
|---------|---------|
| `wildfire_detection_efficientnetb0.ipynb` | Notebook completo com todas as fases executadas |
| `curvas_treinamento.png` | Loss e accuracy por epoch |
| `curva_roc.png` | Curva ROC com AUC |
| `matriz_confusao.png` | Matriz de confusao normalizada |
| `distribuicao_classes.png` | Balanceamento do dataset |
| `exemplos_imagens.png` | Exemplos de imagens do dataset |
| `erros_modelo.png` | Falsos positivos e falsos negativos |
| `resumo_metricas.png` | Dashboard de metricas finais |

> Os modelos treinados (.keras) nao estao incluidos neste repositorio pelo tamanho (84MB).
> Para reproduzir, execute o notebook no Colab com GPU T4.

---

## `Como executar`

```python
# 1. Abra wildfire_detection_efficientnetb0.ipynb no Google Colab
# 2. Configure o Kaggle API token para download do dataset
# 3. Runtime: Change runtime type → GPU (T4)
# 4. Run all
```

---

## `Conceitos aplicados`

- **`Transfer learning`**: reutiliza representacoes visuais de ImageNet para classificacao de satelite
- **`Fine-tuning progressivo`**: congela backbone na fase 1, descongela parcialmente na fase 2 com lr menor
- **`EfficientNetB0`**: arquitetura compound-scaled com compound coefficient phi=0, eficiencia parametros/acuracia
- **`AUC-ROC`**: metrica threshold-independente para avaliacao de classificadores binarios

---

## `Licenca`

Distribuido sob a licenca MIT.

---

## `Autor`

**Arthur Baptista dos Santos**
RM 565346 · Inteligencia Artificial · FIAP 2025-2026

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Arthur%20Baptista-0077B5?logo=linkedin)](https://linkedin.com/in/arthur-baptista-dos-santos)
[![GitHub](https://img.shields.io/badge/GitHub-Arthur--Baptista--dos--Santos-181717?logo=github)](https://github.com/Arthur-Baptista-dos-Santos)
