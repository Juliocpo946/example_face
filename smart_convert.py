# --- INICIO DEL PARCHE DE COMPATIBILIDAD ---
import onnx
if not hasattr(onnx, "mapping"):
    from onnx import TensorProto
    import numpy as np
    class Mapping:
        TENSOR_TYPE_TO_NP_TYPE = {
            TensorProto.FLOAT: np.dtype('float32'),
            TensorProto.DOUBLE: np.dtype('float64'),
            TensorProto.INT32: np.dtype('int32'),
            TensorProto.INT64: np.dtype('int64'),
            TensorProto.STRING: np.dtype('object'),
            TensorProto.BOOL: np.dtype('bool'),
            TensorProto.UINT8: np.dtype('uint8'),
            TensorProto.INT8: np.dtype('int8'),
            TensorProto.UINT16: np.dtype('uint16'),
            TensorProto.INT16: np.dtype('int16'),
            TensorProto.UINT32: np.dtype('uint32'),
            TensorProto.UINT64: np.dtype('uint64'),
        }
    onnx.mapping = Mapping()
# --- FIN DEL PARCHE ---

import torch
import torch.nn as nn
from onnx_tf.backend import prepare
import tensorflow as tf
from hsemotion.facial_emotions import HSEmotionRecognizer

print("=== Iniciando Conversión Inteligente (con Flex Ops) ===")

# 1. Cargar librería
print("[1/5] Cargando librería HSEmotion...")
recognizer = HSEmotionRecognizer(model_name='enet_b0_8_best_afew', device='cpu')

# 2. Buscar modelo
print("[2/5] Buscando modelo interno...")
model = None
for attr_name in dir(recognizer):
    try:
        attr_value = getattr(recognizer, attr_name)
        if isinstance(attr_value, nn.Module):
            print(f"   -> ¡Modelo encontrado en el atributo: '{attr_name}'!")
            model = attr_value
            break
    except:
        pass

if model is None:
    print("ERROR: No se encontró el modelo.")
    exit(1)

model.eval()

# 3. Dummy Input
dummy_input = torch.randn(1, 3, 224, 224)

# 4. Exportar a ONNX
print("[3/5] Exportando a ONNX...")
onnx_path = "emotion_model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=['input'],
    output_names=['output'],
    opset_version=11
)

# 5. Convertir a TensorFlow
print("[4/5] Convirtiendo a TensorFlow...")
onnx_model = onnx.load(onnx_path)
tf_rep = prepare(onnx_model)
tf_rep.export_graph("emotion_model_tf")

# 6. Convertir a TFLite (¡CON LA CORRECCIÓN!)
print("[5/5] Convirtiendo a TFLite (Habilitando TF Select Ops)...")
converter = tf.lite.TFLiteConverter.from_saved_model("emotion_model_tf")

# --- ESTAS SON LAS LÍNEAS MÁGICAS QUE ARREGLAN TU ERROR ---
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # Ops normales de móvil
  tf.lite.OpsSet.SELECT_TF_OPS    # Ops complejas de TensorFlow (DepthwiseConv2dNative)
]
# ----------------------------------------------------------

converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# 7. Guardar
tflite_path = "emotion_model.tflite"
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"\n¡EXITO TOTAL! Archivo generado: {tflite_path}")