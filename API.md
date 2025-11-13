# REFERENCIA API

## FaceDetector

### Inicializacion

```python
from face_detector import FaceDetector

detector = FaceDetector()
```

### Metodos

#### detect(image)

Detecta rostro y retorna landmarks de MediaPipe.

**Parametros:**
- image: np.ndarray (BGR)

**Retorna:**
- landmarks: MediaPipe NormalizedLandmarkList o None

**Ejemplo:**
```python
landmarks = detector.detect(frame)
if landmarks:
    print(f"Detectados {len(landmarks.landmark)} landmarks")
```

#### get_bbox_from_landmarks(landmarks, image_shape)

Calcula bounding box desde landmarks.

**Parametros:**
- landmarks: MediaPipe landmarks
- image_shape: tuple (height, width, channels)

**Retorna:**
- bbox: tuple (x, y, w, h)

#### crop_face(image, bbox)

Extrae region facial.

**Parametros:**
- image: np.ndarray (BGR)
- bbox: tuple (x, y, w, h)

**Retorna:**
- face_crop: np.ndarray (BGR)

#### draw_bbox(image, bbox, color)

Dibuja rectangulo en rostro.

**Parametros:**
- image: np.ndarray (BGR)
- bbox: tuple (x, y, w, h)
- color: tuple (B, G, R), default (0, 255, 0)

**Retorna:**
- result: np.ndarray (BGR)

---

## EmotionClassifier

### Inicializacion

```python
from emotion_classifier import EmotionClassifier

classifier = EmotionClassifier()
```

### Metodos

#### predict(face_crop)

Predice estado cognitivo desde imagen facial.

**Parametros:**
- face_crop: np.ndarray (BGR)

**Retorna:**
- cognitive_state: str
- confidence: float (0.0-1.0)
- emotion: str
- emotion_dict: dict {emotion: probability%}

**Estados cognitivos:**
- 'concentrado'
- 'entendiendo'
- 'distraido'
- 'frustrado'

**Emociones base:**
- 'Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise'

**Ejemplo:**
```python
state, conf, emo, all_emo = classifier.predict(face_crop)
print(f"Estado: {state}, Confianza: {conf:.2%}")
print(f"Emocion: {emo}")
print(f"Todas: {all_emo}")
```

**Output ejemplo:**
```
Estado: concentrado, Confianza: 87%
Emocion: Neutral
Todas: {'Neutral': 87.3, 'Happiness': 8.2, 'Surprise': 2.1, ...}
```

---

## GazeDetector

### Inicializacion

```python
from gaze_detector import GazeDetector

gaze = GazeDetector()
```

### Metodos

#### analyze_gaze(landmarks, image_shape)

Analiza direccion de mirada.

**Parametros:**
- landmarks: MediaPipe landmarks
- image_shape: tuple (height, width, channels)

**Retorna:**
- looking_at_camera: bool
- gaze_direction: str

**Direcciones:**
- 'centro': Mirando camara
- 'lateral': Ligeramente desviado
- 'desviado': Completamente fuera
- 'sin_deteccion': No se pudo calcular

**Ejemplo:**
```python
looking, direction = gaze.analyze_gaze(landmarks, frame.shape)
print(f"Mirando: {looking}, Direccion: {direction}")
```

#### is_looking_at_camera(landmarks, image_shape)

Determina si mira camara con metricas detalladas.

**Parametros:**
- landmarks: MediaPipe landmarks
- image_shape: tuple (height, width)

**Retorna:**
- looking: bool
- left_ratio: float
- right_ratio: float

---

## DrowsinessDetector

### Inicializacion

```python
from drowsiness_detector import DrowsinessDetector

drowsy = DrowsinessDetector()
```

### Atributos publicos

- total_blinks: int
- total_yawns: int
- is_drowsy: bool
- drowsiness_level: int (0-3)

### Metodos

#### detect(landmarks, image_shape)

Detecta somnolencia.

**Parametros:**
- landmarks: MediaPipe landmarks
- image_shape: tuple (height, width)

**Retorna:**
- is_drowsy: bool
- drowsiness_level: int
- stats: dict

**Niveles somnolencia:**
- 0: Alerta
- 1: Cansado
- 2: Somnoliento
- 3: Dormido

**stats contiene:**
```python
{
    'ear': float,              # Eye Aspect Ratio
    'mar': float,              # Mouth Aspect Ratio
    'blinks': int,             # Total parpadeos
    'yawns': int,              # Total bostezos
    'eye_closed_frames': int,  # Frames ojos cerrados
    'recent_blinks': int,      # Parpadeos ultimos 10s
    'recent_yawns': int        # Bostezos ultimos 10s
}
```

**Ejemplo:**
```python
is_drowsy, level, stats = drowsy.detect(landmarks, frame.shape)
print(f"Dormido: {is_drowsy}, Nivel: {level}")
print(f"EAR: {stats['ear']:.2f}, MAR: {stats['mar']:.2f}")
```

#### reset_drowsy_state()

Reinicia contador de somnolencia critica.

**Ejemplo:**
```python
drowsy.reset_drowsy_state()
```

---

## Uso completo

```python
import cv2
from face_detector import FaceDetector
from emotion_classifier import EmotionClassifier
from gaze_detector import GazeDetector
from drowsiness_detector import DrowsinessDetector

face_det = FaceDetector()
emotion_clf = EmotionClassifier()
gaze_det = GazeDetector()
drowsy_det = DrowsinessDetector()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    landmarks = face_det.detect(frame)
    
    if landmarks:
        bbox = face_det.get_bbox_from_landmarks(landmarks, frame.shape)
        face_crop = face_det.crop_face(frame, bbox)
        
        state, conf, emo, all_emo = emotion_clf.predict(face_crop)
        looking, direction = gaze_det.analyze_gaze(landmarks, frame.shape)
        is_drowsy, level, stats = drowsy_det.detect(landmarks, frame.shape)
        
        if is_drowsy or level >= 2:
            final_state = 'somnoliento'
        elif not looking and state != 'frustrado':
            final_state = 'distraido'
        else:
            final_state = state
        
        print(f"Estado: {final_state}, EAR: {stats['ear']:.2f}")
        
        frame_display = face_det.draw_bbox(frame, bbox)
        cv2.imshow('Sistema', frame_display)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Tipos de datos

```python
from typing import Tuple, Dict, Optional
import numpy as np

BBox = Tuple[int, int, int, int]  # (x, y, w, h)
ImageShape = Tuple[int, int, int]  # (height, width, channels)
CognitiveState = str  # 'concentrado' | 'entendiendo' | 'distraido' | 'frustrado' | 'somnoliento'
EmotionDict = Dict[str, float]  # {'Anger': 2.1, 'Neutral': 85.3, ...}
DrowsyStats = Dict[str, Union[float, int]]
```

---

## Constantes configurables

### EmotionClassifier

```python
window_size = 15  # Frames para suavizado
```

### GazeDetector

```python
gaze_threshold = 0.15  # Sensibilidad deteccion
```

### DrowsinessDetector

```python
EAR_THRESHOLD = 0.25      # Umbral ojos cerrados
MAR_THRESHOLD = 0.6       # Umbral bostezo
EYE_CLOSED_FRAMES = 48    # Frames alerta
YAWN_FRAMES = 20          # Frames bostezo
```

---

## Indices landmarks MediaPipe

### Ojos
- Izquierdo: 33, 160, 158, 133, 153, 144
- Derecho: 362, 385, 387, 263, 373, 380

### Iris
- Izquierdo: 474, 475, 476, 477
- Derecho: 469, 470, 471, 472

### Boca
- 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308

---

## Formulas matematicas

### EAR (Eye Aspect Ratio)

```
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
```

Donde p1-p6 son puntos del ojo.

### MAR (Mouth Aspect Ratio)

```
MAR = (||p2-p8|| + ||p3-p7||) / (2 * ||p1-p5||)
```

Donde p1-p8 son puntos de la boca.

### Gaze Ratio

```
gaze_ratio = ||iris_center - eye_center||
```

---

## Manejo de errores

Todos los metodos manejan casos de error retornando valores seguros:

- FaceDetector.detect(): None si no detecta rostro
- EmotionClassifier.predict(): Estado 'concentrado' por defecto
- GazeDetector.analyze_gaze(): (False, 'sin_deteccion')
- DrowsinessDetector.detect(): (False, 0, {}) si no hay landmarks
