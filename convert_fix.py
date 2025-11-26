import os
import torch
import torch.nn as nn
# Importamos para asegurar que las clases estén disponibles para el pickle
from hsemotion.facial_emotions import HSEmotionRecognizer 
import onnx

print("=== Conversión Definitiva (Corrección de Carga) ===")

# 1. Localizar el archivo de pesos .pt
user_home = os.path.expanduser("~")
model_path = os.path.join(user_home, ".hsemotion", "enet_b0_8_best_afew.pt")

print(f"[1/5] Buscando archivo: {model_path}")

if not os.path.exists(model_path):
    if os.path.exists("enet_b0_8_best_afew.pt"):
        model_path = "enet_b0_8_best_afew.pt"
    else:
        print("ERROR: No se encuentra el archivo .pt")
        exit(1)

# 2. Cargar el archivo (Manejo robusto de tipos)
print("[2/5] Cargando archivo...")
try:
    loaded_obj = torch.load(model_path, map_location='cpu')
    print(f"   -> Objeto cargado tipo: {type(loaded_obj).__name__}")
except Exception as e:
    print(f"ERROR carga: {e}")
    exit(1)

# Normalizar a diccionario de pesos (state_dict)
state_dict = None
backbone_model = None

if isinstance(loaded_obj, dict):
    state_dict = loaded_obj
    print("   -> Es un diccionario de pesos.")
elif hasattr(loaded_obj, 'state_dict'):
    state_dict = loaded_obj.state_dict()
    backbone_model = loaded_obj # Guardamos el modelo cargado para usarlo de backbone
    print("   -> Es un Modelo completo. Pesos extraídos.")
else:
    print("ERROR: Formato desconocido.")
    exit(1)

# 3. Buscar los pesos del clasificador [8, 1280]
print("[3/5] Buscando matriz 'secreta' de 8 emociones...")

classifier_weight = None
classifier_bias = None

for key, value in state_dict.items():
    if isinstance(value, torch.Tensor):
        if value.shape == torch.Size([8, 1280]):
            print(f"   -> ¡ENCONTRADO! Pesos (8x1280) en: '{key}'")
            classifier_weight = value
        elif value.shape == torch.Size([8]):
            # Usamos heurística: si encontramos el peso, el bias suele tener nombre similar
            if classifier_weight is not None: 
                # Verificamos si el nombre coincide parcialmente (ej: classifier.weight y classifier.bias)
                # O simplemente tomamos el último vector de 8 encontrado
                print(f"   -> Candidato a Bias (8) en: '{key}'")
                classifier_bias = value

if classifier_weight is None:
    print("ERROR: No se encontró la matriz [8, 1280].")
    exit(1)

# 4. Reconstruir el modelo Frankenstein
print("[4/5] Ensamblando modelo final...")

# Si no tenemos backbone del archivo, intentamos cargarlo de la librería
if backbone_model is None:
    try:
        recognizer = HSEmotionRecognizer(model_name='enet_b0_8_best_afew', device='cpu')
        if hasattr(recognizer, 'net'): backbone_model = recognizer.net
        elif hasattr(recognizer, 'model'): backbone_model = recognizer.model
    except:
        print("No se pudo instanciar backbone de librería.")

if backbone_model is None:
    print("ERROR: Falta el backbone.")
    exit(1)

backbone_model.eval()

class FinalModel(nn.Module):
    def __init__(self, backbone, weight, bias):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(1280, 8)
        # Forzamos los pesos encontrados
        self.head.weight.data = weight
        if bias is not None:
            self.head.bias.data = bias
        else:
            self.head.bias.data.fill_(0.0)
            
    def forward(self, x):
        # Intentamos extraer features puras (sin clasificar)
        if hasattr(self.backbone, 'forward_features'):
            features = self.backbone.forward_features(x)
            # Global Average Pooling si es 4D [B, C, H, W] -> [B, C]
            if features.dim() == 4:
                features = features.mean(dim=(2, 3))
        else:
            features = self.backbone(x)
            
        # Si el backbone por error ya devuelve 8, lo devolvemos directo
        if features.shape[1] == 8:
            return features
            
        # Si devuelve features (1280), pasamos por nuestra cabeza corregida
        if features.shape[1] == 1280:
            return self.head(features)
            
        return features # Fallback

final_model = FinalModel(backbone_model, classifier_weight, classifier_bias)
final_model.eval()

# 5. Exportar
print("[5/5] Exportando...")
dummy_input = torch.randn(1, 3, 224, 224)

# Validar antes de exportar
out = final_model(dummy_input)
print(f"   -> Validación de salida: {out.shape}")
if out.shape[1] != 8:
    print("ERROR: La salida sigue sin ser 8.")
    exit(1)

onnx_path = "emotion_fixed.onnx"
torch.onnx.export(
    final_model,
    dummy_input,
    onnx_path,
    input_names=['input'],
    output_names=['output'],
    opset_version=11
)

os.system(f"onnx2tf -i {onnx_path} -o saved_model_fixed")

print("\n" + "="*60)
print("¡LISTO! Archivo: saved_model_fixed/emotion_fixed_float32.tflite")
print("Renómbralo a 'emotion_model.tflite' y cópialo a tu app.")
print("="*60)