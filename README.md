# Sistema de Analisis de Emociones y Atencion

Sistema de clasificacion de emociones, deteccion de somnolencia y seguimiento de atencion en tiempo real.

## Componentes

| Componente | Tecnologia | Funcion |
|------------|------------|---------|
| Emociones | HSEmotion (EfficientNet-B0) | Clasificacion de 8 emociones basicas |
| Somnolencia | MediaPipe Face Mesh + EAR/MAR | Deteccion de ojos cerrados y bostezos |
| Atencion | MediaPipe + Head Pose | Deteccion si mira a la pantalla |
| Rostros | Haar Cascade OpenCV | Deteccion facial |

## Estados Detectados

| Estado | Descripcion | Prioridad |
|--------|-------------|-----------|
| DURMIENDO | Ojos cerrados por tiempo prolongado | 1 (maxima) |
| NO MIRA PANTALLA | Cabeza girada fuera del rango | 2 |
| FRUSTRADO | Enojo, tristeza, disgusto | 3 |
| DISTRAIDO | Miedo, sorpresa, bostezando | 4 |
| CONCENTRADO | Neutral, atento | 5 |
| ENTENDIENDO | Feliz, comprendiendo | 6 |

## Requisitos del Sistema

- Python 3.8 - 3.11
- Webcam
- 4GB RAM minimo
- Windows / Linux / Mac

## Instalacion

### 1. Clonar o copiar el proyecto

```bash
cd emotion_system
```

### 2. Crear entorno virtual

```bash
python -m venv venv
```

### 3. Activar entorno virtual

**Windows (CMD):**
```bash
venv\Scripts\activate
```

**Windows (PowerShell):**
```bash
venv\Scripts\Activate.ps1
```

**Linux / Mac:**
```bash
source venv/bin/activate
```

### 4. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Verificar instalacion

```bash
python -c "import cv2; import mediapipe; import hsemotion; print('OK')"
```

## Ejecucion

### Activar entorno virtual (si no esta activo)

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / Mac:**
```bash
source venv/bin/activate
```

### Ejecutar aplicacion

```bash
python main.py
```

## Controles

| Tecla | Accion |
|-------|--------|
| `q` | Salir |
| `d` | Mostrar/ocultar detalles de emociones |

## Estructura del Proyecto

```
emotion_system/
├── main.py                  # Aplicacion principal
├── config.py                # Configuraciones y parametros
├── interfaces.py            # Interfaces abstractas (SOLID)
├── face_detector.py         # Deteccion de rostros (Haar Cascade)
├── landmark_extractor.py    # Extraccion de landmarks (MediaPipe)
├── drowsiness_analyzer.py   # Analisis de somnolencia (EAR/MAR)
├── attention_analyzer.py    # Analisis de atencion (Head Pose)
├── emotion_classifier.py    # Clasificacion de emociones (HSEmotion)
├── state_aggregator.py      # Agregacion de estados
├── display_renderer.py      # Renderizado visual
├── video_capture.py         # Captura de video
├── analysis_pipeline.py     # Pipeline de procesamiento
├── requirements.txt         # Dependencias
└── README.md                # Documentacion
```

## Arquitectura

```
Camara
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│                    Video Capture                         │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│                 Face Detector (Haar Cascade)             │
└─────────────────────────────────────────────────────────┘
  │
  ├──────────────────────┬────────────────────────────────┐
  ▼                      ▼                                ▼
┌──────────────┐  ┌─────────────────┐  ┌─────────────────────┐
│  Landmark    │  │   HSEmotion     │  │                     │
│  Extractor   │  │   Classifier    │  │                     │
│ (MediaPipe)  │  │ (EfficientNet)  │  │                     │
└──────────────┘  └─────────────────┘  │                     │
  │                      │             │                     │
  ├──────────┬───────────┘             │                     │
  ▼          ▼                         │                     │
┌──────────────────────────┐           │                     │
│   Drowsiness Analyzer    │           │   State Aggregator  │
│   (EAR + MAR)            │──────────▶│                     │
└──────────────────────────┘           │                     │
  │                                    │                     │
  ▼                                    │                     │
┌──────────────────────────┐           │                     │
│   Attention Analyzer     │──────────▶│                     │
│   (Head Pose)            │           │                     │
└──────────────────────────┘           └─────────────────────┘
                                                 │
                                                 ▼
                                       ┌─────────────────────┐
                                       │  Display Renderer   │
                                       └─────────────────────┘
                                                 │
                                                 ▼
                                              Pantalla
```

## Metricas Utilizadas

### EAR (Eye Aspect Ratio)

Detecta si los ojos estan cerrados.

```
EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
```

- Ojos abiertos: EAR ≈ 0.25 - 0.35
- Ojos cerrados: EAR < 0.22

### MAR (Mouth Aspect Ratio)

Detecta bostezos.

```
MAR = |p_top - p_bottom| / |p_left - p_right|
```

- Normal: MAR < 0.6
- Bostezando: MAR > 0.6

### Head Pose

Detecta orientacion de la cabeza.

- Pitch (inclinacion vertical): ±15°
- Yaw (giro horizontal): ±20°

Fuera de estos rangos = No mira a la pantalla.

## Configuracion

Editar `config.py` para ajustar parametros:

```python
@dataclass
class DrowsinessConfig:
    ear_threshold: float = 0.22          # Umbral EAR
    mar_threshold: float = 0.6           # Umbral MAR
    drowsy_frames_threshold: int = 20    # Frames para confirmar somnolencia
    yawn_frames_threshold: int = 15      # Frames para confirmar bostezo


@dataclass
class AttentionConfig:
    pitch_threshold: float = 15.0        # Grados verticales
    yaw_threshold: float = 20.0          # Grados horizontales
    not_looking_frames_threshold: int = 10
```

## Compatibilidad con Lentes

| Tipo de Lente | Funciona | Precision |
|---------------|----------|-----------|
| Sin lentes | Si | ~95-97% |
| Lentes transparentes | Si | ~93-95% |
| Lentes con antireflejante | Si | ~90-93% |
| Lentes de sol | No | N/A |

## Solucion de Problemas

### Error: No se pudo acceder a la camara

1. Verificar que la webcam esta conectada
2. Cerrar otras aplicaciones que usen la camara
3. En Linux, verificar permisos: `sudo chmod 666 /dev/video0`

### Error: ModuleNotFoundError

1. Verificar que el entorno virtual esta activo
2. Reinstalar dependencias: `pip install -r requirements.txt`

### Rendimiento lento

1. Reducir resolucion de camara
2. Aumentar `process_every_n_frames` en `config.py`
3. Usar GPU si esta disponible (cambiar `device` a `cuda` en `EmotionConfig`)

### MediaPipe no detecta landmarks

1. Mejorar iluminacion
2. Acercarse a la camara
3. Evitar contraluz

## Desactivar Entorno Virtual

```bash
deactivate
```

## Rendimiento Esperado

| Metrica | Valor |
|---------|-------|
| FPS | 25-35 |
| RAM | ~600MB |
| CPU | 15-25% |
| Latencia | <100ms |

## Licencia

MIT License