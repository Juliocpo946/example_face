# --- INICIO DEL PARCHE DE COMPATIBILIDAD ---
import onnx
# El parche: Re-creamos 'mapping' que fue eliminado en versiones modernas de ONNX
# Esto engaña a onnx-tf para que funcione con el ONNX moderno que acabamos de instalar.
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
from onnx_tf.backend import prepare
import tensorflow as tf
from hsemotion.facial_emotions import HSEmotionRecognizer

print("=== Iniciando Conversión con Parche de Compatibilidad ===")

# 1. Cargar tu modelo de PyTorch
print("[1/4] Cargando modelo PyTorch...")
recognizer = HSEmotionRecognizer(model_name='enet_b0_8_best_afew', device='cpu')
model = recognizer.net
model.eval()

# 2. Definir entrada de ejemplo (224x224 RGB)
dummy_input = torch.randn(1, 3, 224, 224)

# 3. Exportar a ONNX
print("[2/4] Exportando a ONNX...")
onnx_path = "emotion_model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=['input'],
    output_names=['output'],
    opset_version=11
)

# 4. Convertir ONNX a TensorFlow
print("[3/4] Convirtiendo a TensorFlow...")
onnx_model = onnx.load(onnx_path)
tf_rep = prepare(onnx_model)
tf_rep.export_graph("emotion_model_tf")

# 5. Convertir a TFLite
print("[4/4] Convirtiendo a TFLite...")
converter = tf.lite.TFLiteConverter.from_saved_model("emotion_model_tf")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# 6. Guardar
tflite_path = "emotion_model.tflite"
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"\n¡EXITO! Archivo generado: {tflite_path}")