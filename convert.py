import os
import torch
import torch.nn as nn
from hsemotion.facial_emotions import HSEmotionRecognizer
import onnx

print("=== Conversión Definitiva (Reparando Clasificador) ===")

# 1. Cargar librería
print("[1/4] Cargando modelo HSEmotion...")
recognizer = HSEmotionRecognizer(model_name='enet_b0_8_best_afew', device='cpu')

# Buscar el modelo base
model = None
if hasattr(recognizer, 'net'):
    model = recognizer.net
elif hasattr(recognizer, 'model'):
    model = recognizer.model
else:
    # Búsqueda manual
    for attr in dir(recognizer):
        val = getattr(recognizer, attr)
        if isinstance(val, nn.Module):
            model = val
            break

if model is None:
    print("ERROR: No se encontró el modelo.")
    exit(1)

model.eval()

# 2. Validar y Reparar el Modelo
dummy_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = model(dummy_input)

print(f"   -> Salida original detectada: {output.shape}")

final_model = model

# Si la salida es 1280 (Features), le falta el clasificador. Lo buscamos y lo unimos.
if output.shape[1] == 1280:
    print("   -> ¡AVISO! Detectado solo el backbone (1280 features).")
    print("   -> Buscando y uniendo el clasificador faltante...")
    
    # Buscamos la capa lineal (classifier/fc) dentro del modelo original
    classifier = None
    # Nombres comunes de la capa final en EfficientNet
    possible_names = ['classifier', '_fc', 'fc', 'last_linear']
    
    for name in possible_names:
        if hasattr(model, name):
            classifier = getattr(model, name)
            print(f"   -> Clasificador encontrado en: '{name}'")
            break
            
    if classifier is not None:
        # Creamos una clase que ejecute Backbone + Clasificador
        class CompleteModel(nn.Module):
            def __init__(self, backbone, head):
                super().__init__()
                self.backbone = backbone
                self.head = head
            
            def forward(self, x):
                # EfficientNet a veces devuelve features en .forward()
                # y necesita pasar por .head() o .classifier()
                x = self.backbone(x) 
                x = self.head(x)
                return x

        # IMPORTANTE: Para evitar recursión infinita, necesitamos 
        # separar la parte de extracción de la parte de clasificación si están en el mismo objeto
        # En timm/efficientnet, .forward() suele hacer todo si no se ha modificado.
        # Si model(x) da 1280, es probable que 'model' sea solo el extractor o forward() esté truncado.
        
        # Truco: Forzamos el modelo wrapper
        final_model = CompleteModel(model, classifier)
        
        # Validamos de nuevo
        out2 = final_model(dummy_input)
        print(f"   -> Nueva salida corregida: {out2.shape}")
        
        if out2.shape[1] != 8:
            print("ERROR: Aún no logramos obtener 8 salidas. Revisa la arquitectura.")
            # Fallback de emergencia: Si todo falla, exportamos lo que hay, 
            # pero tu app Flutter tendrá que manejar 1280 vectores (no recomendado).
            final_model = model 
    else:
        print("   -> NO se encontró clasificador. Se exportará el backbone (1280).")

# 3. Exportar a ONNX
print("[2/4] Exportando a ONNX (Opset 11)...")
onnx_path = "model_float32.onnx"
torch.onnx.export(
    final_model,
    dummy_input,
    onnx_path,
    input_names=['input'],
    output_names=['output'],
    opset_version=11
)

# 4. Ejecutar conversión automática
print("[3/4] Ejecutando onnx2tf...")
os.system(f"onnx2tf -i {onnx_path} -o saved_model_tflite")

print("\n" + "="*50)
print("PROCESO TERMINADO. Revisa la carpeta 'saved_model_tflite'")
print("Copia 'model_float32.tflite' -> 'emotion_model.tflite' en Flutter.")
print("="*50)