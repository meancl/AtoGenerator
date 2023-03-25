import onnxruntime as rt
import onnxmltools
import numpy as np
from sklearn.decomposition import PCA
from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType

from mclLib.cleaner import *
from mclLib.modelpostfix import *
from mclLib.ftploader import *

def decomposite(n, x_set, x_set2=None, onnx_model_name=None):
    pca = PCA(n_components=n)
    x_len = x_set.shape[1]
    pca_model = pca.fit(x_set)
    x1_pca = pca_model.transform(x_set)
    x2_pca = None
    if x_set2 is not None:
        x2_pca = pca_model.transform(x_set2)
    
    if onnx_model_name is not None:
        onnx_model = onnxmltools.convert_sklearn(pca_model, "PCA", [("input", DoubleTensorType([None, x_len]))])
        with open(onnx_path + onnx_model_name + onnx_, "wb") as f:
            f.write(onnx_model.SerializeToString())
        cleanOnnx()

    return pca_model, x1_pca, x2_pca

def getOnnxPcaModel(model_name):
    ftp = FtpLoader(ftp_ip, ftp_port, ftp_id, ftp_pw)
    ftp.download(model_name+onnx_, onnx_path, just_onnx)
    sess = rt.InferenceSession(onnx_path + model_name + onnx_)
    return sess
