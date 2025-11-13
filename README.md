# SISTEMA DE ANALISIS DE EMOCIONES

Sistema de clasificacion de emociones en tiempo real con HSEmotion.

## Componentes

- HSEmotion: Clasificacion de emociones (EfficientNet-B0)
- Face Detection: Haar Cascade OpenCV

## Estados detectados

1. FRUSTRADO - Enojo, tristeza, disgusto
2. DISTRAIDO - Miedo, sorpresa
3. CONCENTRADO - Neutral, atento
4. ENTENDIENDO - Feliz, comprendiendo

## Instalacion

```bash
pip install -r requirements.txt
```

## Ejecucion

```bash
python main.py
```

## Controles

- q: Salir
- d: Detalles emociones

## Metricas

- Precision: 90-95%
- FPS: 30-40
- RAM: 500MB

## Arquitectura

```
Camara
  |
Haar Cascade (deteccion rostro)
  |
HSEmotion (clasificacion)
  |
Suavizado temporal (15 frames)
  |
Estado cognitivo
```

## Estructura

```
.
├── main.py
├── face_detector.py
├── emotion_classifier.py
└── requirements.txt
```

## Requisitos

- Python 3.8-3.11
- Webcam
- 4GB RAM minimo
- Windows/Linux/Mac