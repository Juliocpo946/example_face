GUÍA COMPLETA DE DETECTORES DE EMOCIONES Y MIRADA
===================================================

DETECTORES DE EMOCIONES DISPONIBLES
------------------------------------

1. FER (Facial Emotion Recognition)
   Archivo: main_fer.py
   Requisitos: requirements_fer.txt
   Ventajas:
   - Instalación simple y rápida
   - Bajo consumo de recursos
   - CNN preentrenado
   Desventajas:
   - Precisión: ~59-66%
   - Solo 7 emociones básicas
   Recomendado para: Prototipos rápidos, equipos limitados

2. DeepFace
   Archivo: main_deepface.py
   Requisitos: requirements_deepface.txt
   Ventajas:
   - 97%+ precisión en rostros
   - Múltiples modelos (VGG-Face, FaceNet)
   - Bien documentado y usado en producción
   Desventajas:
   - Descarga ~100MB en primera ejecución
   - Menos preciso en emociones espontáneas
   Recomendado para: Balancear precisión y velocidad

3. HSEmotion ⭐ RECOMENDADO
   Archivo: main_hsemotion.py
   Requisitos: requirements_hsemotion.txt
   Ventajas:
   - 🏆 Ganador ABAW 2022-2024 (state-of-the-art)
   - Velocidad optimizada (EfficientNet)
   - 8 emociones (incluye Contempt)
   - Mejor balance precisión/velocidad
   Desventajas:
   - Requiere PyTorch (~500MB primera vez)
   Recomendado para: Producción, máxima precisión

4. Py-Feat (Suite completa)
   Archivo: main_pyfeat.py
   Requisitos: requirements_pyfeat.txt
   Ventajas:
   - Suite completa (emociones + action units + landmarks)
   - Análisis científico detallado
   - 68 landmarks faciales
   - Orientación de cabeza
   Desventajas:
   - Mayor consumo de RAM
   - Descarga ~500MB
   - Más lento que otros
   Recomendado para: Investigación, análisis detallado


DETECTOR DE MIRADA
------------------

5. Gaze Tracking (Seguimiento de pupilas)
   Archivo: main_gaze.py
   Requisitos: requirements_gaze.txt
   Funciones:
   - Detecta si el usuario mira a la cámara
   - Dirección de mirada (centro, izquierda, derecha, arriba, abajo)
   - Detección de pupilas en tiempo real
   - Se combina con cualquier detector de emociones
   Ventajas:
   - Solo requiere OpenCV (ya instalado)
   - No descarga modelos adicionales
   - Ligero y rápido
   Limitaciones:
   - Requiere buena iluminación
   - Sensible a reflejos en ojos
   Recomendado para: Detectar atención real del usuario


RECOMENDACIÓN DE USO
--------------------

CASO 1: Máxima Precisión + Análisis de Atención
   python main_gaze.py
   Combina HSEmotion (state-of-the-art) + Gaze Tracking

CASO 2: Análisis Científico Completo
   python main_pyfeat.py
   Action units, landmarks, emociones detalladas

CASO 3: Producción Balanceada
   python main_hsemotion.py
   Mejor relación precisión/velocidad

CASO 4: Prototipo Rápido
   python main_fer.py
   Instalación más rápida


COMPARACIÓN DE PRECISIÓN
-------------------------
HSEmotion:  ████████████████████ 95%+ (ganador ABAW)
Py-Feat:    ███████████████████░ 90-95% (científico)
DeepFace:   ██████████████████░░ 85-90% (rostros)
FER:        █████████████░░░░░░░ 59-66% (básico)


COMPARACIÓN DE VELOCIDAD
-------------------------
HSEmotion:  ████████████████████ Muy rápido (optimizado)
FER:        ███████████████████░ Rápido
DeepFace:   ████████████████░░░░ Medio
Py-Feat:    ██████████░░░░░░░░░░ Lento (análisis completo)


INSTALACIÓN RECOMENDADA
------------------------

PASO 1: Instala HSEmotion (mejor opción)
   pip install -r requirements_hsemotion.txt

PASO 2: Gaze Tracking ya funciona (solo usa OpenCV)
   pip install -r requirements_gaze.txt

PASO 3: Ejecuta el sistema completo
   python main_gaze.py


CONTROLES UNIVERSALES
----------------------
'q' - Salir del programa
'd' - Mostrar/ocultar detalles de emociones
'g' - Activar/desactivar análisis de mirada (main_gaze.py)


ESTADOS COGNITIVOS DETECTADOS
------------------------------
CONCENTRADO (verde): Neutral, atento
ENTENDIENDO (amarillo): Happy, comprendiendo
DISTRAÍDO (naranja): Fear, Surprise, desenfocado
FRUSTRADO (rojo): Anger, Disgust, Sad, bloqueado


ARQUITECTURA DEL SISTEMA
-------------------------

face_detector.py          → Detecta rostros (Haar Cascade)
                          ↓
[EMOCIÓN]                 → hsemotion_classifier.py (RECOMENDADO)
                          → deepface_classifier.py
                          → fer_classifier.py
                          → pyfeat_classifier.py
                          ↓
[MIRADA]                  → gaze_detector.py
                          ↓
[INTEGRACIÓN]             → main_gaze.py (emociones + mirada)
                          → main_hsemotion.py (solo emociones)
                          → main_deepface.py (solo emociones)
                          → main_fer.py (solo emociones)
                          → main_pyfeat.py (análisis completo)


SOLUCIÓN DE PROBLEMAS
----------------------

Problema: "No se detectan emociones correctamente"
  - Asegura buena iluminación frontal
  - Rostro completamente visible
  - Cámara a la altura de los ojos

Problema: "No detecta si miro la cámara"
  - Ajusta threshold en gaze_detector.py (línea 9)
  - Ilumina uniformemente tu rostro
  - Evita reflejos directos en gafas/ojos

Problema: "Sistema lento"
  - Usa main_fer.py (más ligero)
  - Cierra otros programas
  - Reduce resolución de cámara

Problema: "Error al instalar PyTorch"
  - Windows: Descarga wheel desde pytorch.org
  - Linux: sudo apt-get install python3-torch


NOTAS DEL PROFESOR
-------------------
- ResNet: Redes neuronales residuales
- MLflow: Tracking de experimentos ML
- Vectorización de imágenes: Convertir a embeddings
- U-Net + CLIP: Segmentación + embeddings visuales


PRÓXIMOS PASOS
--------------
1. Prueba main_gaze.py primero (combina todo)
2. Si es lento, cambia a main_hsemotion.py
3. Para investigación, usa main_pyfeat.py
4. Ajusta thresholds según tu ambiente


CONTACTO Y SOPORTE
------------------
HSEmotion: https://github.com/sb-ai-lab/EmotiEffLib
Py-Feat: https://py-feat.org/
DeepFace: https://github.com/serengil/deepface
FER: https://github.com/justinshenk/fer
