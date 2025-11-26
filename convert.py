# Conversión con Flex Ops - Warnings corregidos
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suprimir warnings de TF
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desactivar oneDNN warnings

import warnings
warnings.filterwarnings('ignore')

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

import torch
import torch.nn as nn
from onnx_tf.backend import prepare
import tensorflow as tf
from hsemotion.facial_emotions import HSEmotionRecognizer

print("=== Conversion HSEmotion a TFLite (Flex Ops) ===")
print()

# 1. Cargar modelo
print("[1/5] Cargando modelo HSEmotion...")
recognizer = HSEmotionRecognizer(model_name='enet_b0_8_best_afew', device='cpu')

model = None
for attr_name in dir(recognizer):
    try:
        attr_value = getattr(recognizer, attr_name)
        if isinstance(attr_value, nn.Module):
            model = attr_value
            break
    except:
        pass

if model is None:
    print("ERROR: No se encontro el modelo.")
    exit(1)

model.eval()
print("      Modelo cargado correctamente")

# 2. Exportar a ONNX
print("[2/5] Exportando a ONNX...")
dummy_input = torch.randn(1, 3, 224, 224)
onnx_path = "emotion_model.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=['input'],
    output_names=['output'],
    opset_version=11,
    verbose=False
)
print("      ONNX exportado")

# 3. Convertir a TensorFlow
print("[3/5] Convirtiendo a TensorFlow SavedModel...")
onnx_model = onnx.load(onnx_path)
tf_rep = prepare(onnx_model)
tf_rep.export_graph("emotion_model_tf")
print("      SavedModel creado")

# 4. Convertir a TFLite con Flex Ops
print("[4/5] Convirtiendo a TFLite con Flex Ops...")
converter = tf.lite.TFLiteConverter.from_saved_model("emotion_model_tf")

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()
print("      Conversion completada")

# 5. Guardar
print("[5/5] Guardando modelo...")
tflite_path = "emotion_model.tflite"
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

size_mb = len(tflite_model) / 1024 / 1024
print()
print("=" * 50)
print(f"EXITO: {tflite_path}")
print(f"Tamano: {size_mb:.2f} MB")
print()
print("IMPORTANTE: Este modelo requiere Flex Ops en Android.")
print("Agrega esta dependencia en android/app/build.gradle.kts:")
print('  implementation("org.tensorflow:tensorflow-lite-select-tf-ops:2.14.0")')
print("=" * 50)