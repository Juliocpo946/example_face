# INSTALACION

## Requisitos

- Python 3.8 - 3.11
- Webcam
- 4GB RAM minimo
- Windows/Linux/Mac

## Pasos

### 1. Clonar o descargar proyecto

```bash
git clone <repository>
cd proyecto
```

### 2. Crear entorno virtual

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

Linux/Mac:
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar

```bash
python main.py
```

## Verificacion

Si todo funciona correctamente veras:

```
============================================================
SISTEMA DE ANALISIS COGNITIVO
MediaPipe + HSEmotion + Gaze + Drowsiness
============================================================
[INFO] Inicializando componentes...
[INFO] Sistema iniciado
```

## Solucion de problemas

### Error: No se pudo acceder a la camara

Verificar que:
- Camara conectada y funcionando
- Ningun otro programa usa la camara
- Permisos de camara habilitados

### Error: ModuleNotFoundError

```bash
pip install -r requirements.txt
```

### Sistema lento

- Reducir procesamiento: Cambiar frame_count % 3 a % 5 en main.py
- Cerrar otros programas
- Verificar RAM disponible

### MediaPipe no detecta rostro

- Mejorar iluminacion
- Verificar distancia de camara (50-100cm)
- Rostro completamente visible

## Parametros ajustables

### emotion_classifier.py
```python
window_size = 15  # Suavizado temporal
```

### gaze_detector.py
```python
gaze_threshold = 0.15  # Sensibilidad gaze
```

### drowsiness_detector.py
```python
EAR_THRESHOLD = 0.25      # Umbral ojos cerrados
MAR_THRESHOLD = 0.6       # Umbral bostezo
EYE_CLOSED_FRAMES = 48    # Frames para alerta
YAWN_FRAMES = 20          # Frames bostezo
```

## Rendimiento esperado

Hardware minimo:
- CPU: Intel i5 8va gen
- RAM: 4GB
- FPS: 20-25

Hardware recomendado:
- CPU: Intel i7 8va gen
- RAM: 8GB
- FPS: 30-35
