import tensorflow as tf
import os
import shutil

from mclLib.modelpostfix import *

def transformToOnnx(model_name_list):
    for mn in model_name_list:
        # h5 to pb
        model_convert = tf.keras.models.load_model(h5_path + mn + h5_, compile=False)
        model_convert.save(tmp_model_path, save_format="tf")

        # pb to onnx 
        os.system('python -m tf2onnx.convert --saved-model ' +  tmp_model_path + ' --output ' + onnx_path + mn + onnx_ + ' --opset 17')

        print(f'onnx transorm complete {mn + onnx_}')
        shutil.rmtree(tmp_model_path)


