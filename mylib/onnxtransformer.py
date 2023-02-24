from mylib.modelpostfix import *
import tensorflow as tf
import os

def transformToOnnx(model_name_list):
    for mn in model_name_list:
        # h5 to pb
        model_convert = tf.keras.models.load_model(h5_path + mn + h5, compile=False)
        model_convert.save(tmp_model_path, save_format="tf")

        # pb to onnx 
        os.system('python -m tf2onnx.convert --saved-model ' +  tmp_model_path + ' --output ' + onnx_path + mn + onnx + ' --opset 13')

        print(f'onnx transorm complete {mn + onnx}')

