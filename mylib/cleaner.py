import os.path
import os
from mylib.modelpostfix import *
from mylib.ftploader import *
        
def clean(data_list=None):
    if data_list is None:
        h5_dir = os.listdir(h5_path)
        onnx_dir = os.listdir(onnx_path)

        ftp= FtpLoader("221.149.119.60", 2021, "ftp_user", "jin9409")

        for filename in h5_dir:
            if filename == '_':
                continue
            ftp.upload(filename, h5_path, '/h5/')
            os.remove(h5_path + filename)

        for filename in onnx_dir:
            if filename == '_':
                continue
            ftp.upload(filename, onnx_path, '/onnx/')
            os.remove(onnx_path + filename)
    else:
        for filename in data_list:
            os.remove(h5_path + filename + h5_)
