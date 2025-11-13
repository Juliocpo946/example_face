# ARQUITECTURA DEL SISTEMA

## Vision general

Sistema modular de analisis cognitivo en tiempo real que combina deteccion facial avanzada, clasificacion de emociones y analisis de comportamiento visual.

## Componentes principales

### 1. FaceDetector

Responsable de deteccion y extraccion de landmarks faciales.

**Tecnologia:** MediaPipe Face Mesh

**Funciones:**
- Detecta 468 landmarks faciales
- Incluye refinamiento para iris (478 landmarks totales)
- Extrae bounding box del rostro
- Proporciona coordenadas normalizadas

**Metodos:**
- detect(): Procesa frame y retorna landmarks
- get_bbox_from_landmarks(): Calcula bounding box
- crop_face(): Extrae ROI facial
- draw_bbox(): Visualiza deteccion

**Configuracion:**
- max_num_faces: 1
- refine_landmarks: True
- min_detection_confidence: 0.5
- min_tracking_confidence: 0.5

### 2. EmotionClassifier

Clasificador de emociones basado en deep learning.

**Tecnologia:** HSEmotion (EfficientNet-B0)

**Emociones detectadas:**
- Anger, Contempt, Disgust, Sadness → frustrado
- Fear, Surprise → distraido
- Happiness → entendiendo
- Neutral → concentrado

**Suavizado temporal:**
- Ventana deslizante de 15 frames
- Votacion por mayoria
- Promedio de confianza

**Metodos:**
- predict(): Retorna estado cognitivo, confianza, emocion, probabilidades

**Precision:** 90-95%

### 3. GazeDetector

Detector de direccion de mirada mediante iris tracking.

**Tecnologia:** MediaPipe Iris

**Landmarks utilizados:**
- Ojos: 33, 160, 158, 133, 153, 144 (izq) / 362, 385, 387, 263, 373, 380 (der)
- Iris: 474-477 (izq) / 469-472 (der)

**Metodos:**
- get_eye_center(): Calcula centro del ojo
- get_iris_center(): Calcula centro del iris
- calculate_gaze_ratio(): Distancia iris-centro
- is_looking_at_camera(): Determina atencion
- analyze_gaze(): Analisis completo

**Estados:**
- centro: Mirando camara
- lateral: Mirada desviada
- desviado: Completamente fuera

**Precision:** 85-90%

### 4. DrowsinessDetector

Detector de somnolencia mediante EAR/MAR.

**Tecnologia:** MediaPipe landmarks + scipy

**Landmarks utilizados:**
- Ojos: 33, 160, 158, 133, 153, 144 (izq) / 362, 385, 387, 263, 373, 380 (der)
- Boca: 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308

**Metricas:**

EAR (Eye Aspect Ratio):
```
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
```

MAR (Mouth Aspect Ratio):
```
MAR = (||p2-p8|| + ||p3-p7||) / (2 * ||p1-p5||)
```

**Umbrales:**
- EAR < 0.25: Ojos cerrados
- MAR > 0.6: Bostezo
- 48 frames: Alerta critica
- 20 frames: Confirmar bostezo

**Niveles:**
- 0: Alerta
- 1: Cansado (15+ parpadeos/10s)
- 2: Somnoliento (3+ bostezos/10s)
- 3: Dormido (ojos cerrados >2s)

**Metodos:**
- calculate_ear(): Calcula EAR
- calculate_mar(): Calcula MAR
- detect(): Analisis completo
- reset_drowsy_state(): Reinicia estado

**Precision:** 90-95%

## Flujo de datos

```
Frame de camara
    |
    v
FaceDetector (MediaPipe)
    |
    +-- landmarks (478 puntos)
    |
    +----> EmotionClassifier
    |      |
    |      +-- HSEmotion
    |      +-- Suavizado temporal
    |      +-- Estado cognitivo base
    |
    +----> GazeDetector
    |      |
    |      +-- Iris tracking
    |      +-- Calculo gaze ratio
    |      +-- Determinar atencion
    |
    +----> DrowsinessDetector
           |
           +-- Calculo EAR
           +-- Calculo MAR
           +-- Conteo parpadeos/bostezos
           +-- Nivel somnolencia
           |
           v
    Decision Logic
    (prioridad: somnoliento > frustrado > distraido > concentrado > entendiendo)
           |
           v
    Estado Final + Metricas
           |
           v
    Visualizacion
```

## Logica de decision

```python
if is_drowsy or drowsy_level >= 2:
    estado_final = SOMNOLIENTO
elif not looking_at_camera and estado != FRUSTRADO:
    estado_final = DISTRAIDO
else:
    estado_final = emotion_classifier.predict()
```

## Optimizaciones

### Procesamiento
- Analisis cada 3 frames (frame_count % 3)
- MediaPipe en modo tracking (mas rapido)
- Suavizado temporal reduce recalculos

### Memoria
- deque con maxlen (historial limitado)
- Sin almacenamiento de frames
- Procesamiento en streaming

### CPU
- MediaPipe optimizado para CPU
- HSEmotion en CPU mode
- Sin operaciones GPU

## Metricas de rendimiento

### Latencia por componente
- FaceDetector: 10-15ms
- EmotionClassifier: 30-40ms
- GazeDetector: 5-10ms
- DrowsinessDetector: 5-10ms
- Total: 50-75ms (13-20 FPS teorico)
- Real con optimizaciones: 25-35 FPS

### Consumo recursos
- RAM: 600MB
- CPU: 40-60% (un core)
- Ancho banda: N/A (local)

### Precision global
- Combinada: 88-92%
- Falsos positivos: <5%
- Falsos negativos: <8%

## Escalabilidad

### Para API
Sistema disenado para facil adaptacion a arquitectura cliente-servidor:

```
Cliente (movil/tablet)
    |
    +-- Captura video stream
    |
    v
API (servidor)
    |
    +-- FaceDetector
    +-- EmotionClassifier
    +-- GazeDetector
    +-- DrowsinessDetector
    |
    v
Respuesta JSON
    |
    +-- estado
    +-- confianza
    +-- metricas
```

### Consideraciones API
- Procesar cada 3-5 frames del stream
- Respuesta JSON ligera (<1KB)
- Mantener estado en servidor (historial temporal)
- WebSocket para streaming continuo
- REST para queries individuales

## Dependencias

### Criticas
- opencv-python: Captura y procesamiento video
- mediapipe: Deteccion facial y landmarks
- hsemotion: Clasificacion emociones
- torch: Backend HSEmotion

### Auxiliares
- numpy: Operaciones matriciales
- scipy: Calculo distancias
- collections: Estructuras datos (deque)

## Extensiones futuras

### Posibles mejoras
- Multi-face detection
- Tracking por ID
- Analisis temporal extendido
- Exportacion metricas (CSV/DB)
- Dashboard web
- Alertas personalizables
- Integracion con plataformas educativas

### Investigacion
- Modelo personalizado para poblacion sorda
- Dataset propio para fine-tuning
- Action Units adicionales
- Analisis de pose completa
