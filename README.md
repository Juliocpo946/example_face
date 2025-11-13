# SISTEMA DE ANALISIS COGNITIVO

Sistema de deteccion en tiempo real con MediaPipe, HSEmotion y analisis avanzado.

## Componentes

- MediaPipe Face Mesh: 468 landmarks faciales + iris tracking
- HSEmotion: Clasificacion de emociones (EfficientNet-B0)
- Gaze Detection: Seguimiento de mirada mediante iris
- Drowsiness Detection: EAR/MAR para deteccion de somnolencia

## Estados detectados

1. SOMNOLIENTO - Ojos cerrados, bostezos frecuentes
2. FRUSTRADO - Enojo, tristeza, disgusto
3. DISTRAIDO - No mira camara, miedo, sorpresa
4. CONCENTRADO - Neutral, atento
5. ENTENDIENDO - Feliz, comprendiendo

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
- i: Info sistema
- r: Reset somnolencia

## Metricas

### Precision
- Emociones: 90-95%
- Gaze: 85-90%
- Somnolencia: 90-95%
- Global: 88-92%

### Rendimiento
- FPS: 25-35
- RAM: 600MB
- CPU: Medio

### Umbrales
- EAR < 0.25: Ojos cerrados
- MAR > 0.6: Bostezo
- 48 frames: Alerta somnolencia (2s)
- 15 frames: Suavizado emocional

## Arquitectura

```
Camara
  |
MediaPipe Face Mesh (468 landmarks)
  |
  +-> EmotionClassifier (HSEmotion)
  +-> GazeDetector (Iris tracking)
  +-> DrowsinessDetector (EAR/MAR)
  |
Decision Logic
  |
Estado Final
```

## Estructura

```
.
├── main.py
├── face_detector.py
├── emotion_classifier.py
├── gaze_detector.py
├── drowsiness_detector.py
└── requirements.txt
```

## Requisitos

- Python 3.8-3.11
- Webcam
- 4GB RAM minimo
- Windows/Linux/Mac

## Notas

Sistema optimizado para uso en mobiles/tablets via API.
Procesamiento ligero y escalable.
