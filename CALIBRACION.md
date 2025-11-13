# AJUSTES REALIZADOS

## Cambios en gaze_detector.py

### Umbral de deteccion de mirada
- Anterior: `avg_ratio < 5.0`
- Nuevo: `avg_ratio < 10.0`
- Razon: Mas tolerante a variaciones de distancia/iluminacion

### Umbral desviacion severa
- Anterior: `> 7`
- Nuevo: `> 12`
- Razon: Distingue mejor entre mirada lateral y completamente desviada

## Cambios en drowsiness_detector.py

### EAR threshold
- Anterior: `0.25`
- Nuevo: `0.23`
- Razon: Mas estricto para detectar ojos cerrados

### MAR threshold
- Anterior: `0.6`
- Nuevo: `0.65`
- Razon: Evita falsos positivos de bostezos

### Frames para alerta
- Anterior: `48` (2 segundos)
- Nuevo: `60` (2.5 segundos)
- Razon: Mas tiempo antes de alerta critica

### Frames para bostezo
- Anterior: `20`
- Nuevo: `25`
- Razon: Confirmar bostezo real vs movimiento boca

### Parpadeos para nivel 1
- Anterior: `15`
- Nuevo: `25`
- Razon: Parpadeo normal no es cansancio

### Parpadeos para nivel 2
- Anterior: `25`
- Nuevo: `35`
- Razon: Menos falsos positivos

### Bostezos para nivel 2
- Anterior: `3`
- Nuevo: `4`
- Razon: Mas evidencia antes de alertar

### Ventana historial
- Anterior: `300 frames` (10 segundos)
- Nuevo: `200 frames` (6-7 segundos)
- Razon: Respuesta mas rapida a cambios

## Cambios en main.py

### Output debug
Agregado contador de parpadeos recientes para diagnostico

## Como calibrar manualmente

### Si gaze siempre dice NO

Ajustar en gaze_detector.py linea 49:
```python
looking = avg_ratio < X
```
- Valores menores (5-8): Mas estricto
- Valores mayores (10-15): Mas tolerante

### Si detecta somnolencia cuando estas despierto

Ajustar en drowsiness_detector.py:
```python
EAR_THRESHOLD = X  # Valores: 0.20-0.28
```
- Menor valor: Mas estricto (solo muy cerrados)
- Mayor valor: Mas tolerante

### Si no detecta bostezos reales

Ajustar en drowsiness_detector.py:
```python
MAR_THRESHOLD = X  # Valores: 0.55-0.70
```
- Menor valor: Mas sensible
- Mayor valor: Mas estricto

## Modo debug

Para ver valores reales y calibrar:

En main.py agregar despues de linea 70:
```python
print(f"[DEBUG] Left ratio: {gaze_detector.is_looking_at_camera(landmarks, frame.shape)[1]:.2f}, "
      f"Right ratio: {gaze_detector.is_looking_at_camera(landmarks, frame.shape)[2]:.2f}")
```

Mira la camara directamente y anota los valores.
Ajusta umbral 20% arriba del valor maximo que veas.

## Condiciones optimas

- Iluminacion frontal uniforme
- Distancia: 60-80cm
- Camara a altura de ojos
- Fondo simple
- Sin gafas con mucho reflejo