import torch
import torch.nn as nn
from hsemotion.facial_emotions import HSEmotionRecognizer

print("=== Paso 1: Exportando PyTorch a ONNX (MODELO COMPLETO) ===")

# 1. Cargar el wrapper
recognizer = HSEmotionRecognizer(model_name='enet_b0_8_best_afew', device='cpu')

# [CORRECCIÓN] Acceder explícitamente al modelo real, no buscarlo a ciegas.
# En hsemotion v0.3+, el modelo suele estar en 'recognizer.model' o 'recognizer.net'
# Vamos a probar ambos o inspeccionar.
model = None

if hasattr(recognizer, 'model'):
    model = recognizer.model
elif hasattr(recognizer, 'net'):
    model = recognizer.net
else:
    # Búsqueda de emergencia mejorada: buscamos el que tenga capa lineal final
    print("Buscando modelo con clasificador...")
    for attr_name in dir(recognizer):
        try:
            attr_value = getattr(recognizer, attr_name)
            if isinstance(attr_value, nn.Module):
                # Verificamos si tiene capa de salida (classifier)
                # EfficientNet suele tener '_fc' o 'classifier'
                if hasattr(attr_value, 'classifier') or hasattr(attr_value, '_fc'):
                    model = attr_value
                    print(f"-> Encontrado candidato: {attr_name}")
                    break
        except:
            pass

if model is None:
    print("ERROR CRÍTICO: No se encontró el modelo clasificador.")
    exit(1)

model.eval()

# 2. Validación de Salida (Para asegurarnos que son 8 emociones)
dummy_input = torch.randn(1, 3, 224, 224)
try:
    output = model(dummy_input)
    print(f"Forma de salida detectada: {output.shape}")
    if output.shape[1] != 8:
        print(f"¡ALERTA! La salida es {output.shape[1]}, pero esperábamos 8 (emociones).")
        print("Es posible que estemos exportando solo el backbone.")
        # Si sale 1280, necesitamos añadir la capa final manualmente si la librería la separó
        exit(1)
    else:
        print("¡CORRECTO! El modelo exportará 8 clases.")
except Exception as e:
    print(f"Error al probar el modelo: {e}")
    exit(1)

# 3. Exportar a ONNX (Opset 11)
onnx_path = "model_float32.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=['input'],
    output_names=['output'],
    opset_version=11
)

print(f"LISTO: Archivo '{onnx_path}' creado correctamente.")