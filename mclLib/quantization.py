from onnxruntime.quantization import quantize_dynamic, QuantType
from onnx import load

from mclLib.modelpostfix import *

def quantize_onnx_model(onnx_model_path, quantized_model_path):
    onnx_opt_model = load(onnx_model_path)
    quantize_dynamic( onnx_model_path,
                     quantized_model_path,
                     weight_type=QuantType.QInt8)
    print(f"quantized model saved to: {quantized_model_path}")

def quantize_onnx_model_list(model_list):
    for mn in model_list:
        onnx_change = onnx_path + mn + onnx_
        quantize_onnx_model(onnx_change, onnx_change)


