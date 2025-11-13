# USO DE HERRAMIENTA DE CALIBRACION

## Problema

El sistema siempre detecta "Mirando: NO" incluso cuando miras la camara.

## Solucion

Usar herramienta de calibracion automatica para tu camara/iluminacion especifica.

## Pasos

### 1. Ejecutar herramienta

```bash
python calibrate_gaze.py
```

### 2. Preparacion

- Posicionate a 60-80cm de la camara
- Iluminacion frontal uniforme
- Sin gafas con mucho reflejo

### 3. Calibrar

- Presiona 'c' para iniciar
- Mira fijamente al centro de la camara durante 3 segundos
- NO muevas la cabeza

### 4. Resultado

Veras en consola algo como:

```
[RESULTADO]
  Ratio promedio: 8.45
  Ratio maximo: 12.32
  Threshold actual: 10.0
  Threshold recomendado: 16.0

En gaze_detector.py linea 12, cambiar a:
  self.looking_threshold = 16.0
```

### 5. Aplicar cambio

Abrir `gaze_detector.py` y en linea 12 cambiar:

```python
self.looking_threshold = 16.0
```

### 6. Probar

```bash
python main.py
```

Ahora deberia detectar correctamente cuando miras.

## Explicacion de valores

La herramienta muestra en pantalla:

- **Left/Right**: Distancia iris-centro de cada ojo
- **Avg**: Promedio de ambos ojos
- **Threshold**: Umbral actual (si Avg < Threshold = mirando)

Si Avg siempre es mayor que Threshold cuando miras, necesitas calibrar.

## Si sigue sin funcionar

### Opcion 1: Aumentar threshold manualmente

En `gaze_detector.py` linea 12:

```python
self.looking_threshold = 20.0  # Aumentar de 10 a 20
```

### Opcion 2: Verificar iluminacion

- Luz frontal directa al rostro
- Sin luz trasera fuerte (contraluz)
- Evitar reflejos en ojos

### Opcion 3: Distancia camara

- Prueba 50cm
- Prueba 70cm
- Prueba 90cm

Cada distancia da diferentes ratios.

## Valores tipicos

- Mirando directamente: 3-10
- Mirando ligeramente al lado: 10-15
- Mirando completamente al lado: 15-25

Tu threshold debe ser 30% mayor que tu valor maximo cuando miras directamente.
