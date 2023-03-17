import os.path
import os

from mclLib.modelpostfix import *
from mclLib.ftploader import *
from mclLib.server_info import *

def removeTestData(h5_list:list=None, onnx_list:list=None):
    if h5_list is not None:
        for filename in h5_list:
            os.remove(h5_path + filename + h5_)

    if onnx_list is not None:
        for filename in onnx_list:
            os.remove(onnx_path + filename + onnx_)

def cleanOnnx(ftp=None):
    if ftp is None:
        ftp= FtpLoader(ftp_ip, ftp_port, ftp_id, ftp_pw)
    onnx_dir = os.listdir(onnx_path)

    for filename in onnx_dir:
        if filename == '_':
            continue
        ftp.upload(filename, onnx_path, '/onnx/')
        os.remove(onnx_path + filename)

def cleanH5(ftp=None):
    if ftp is None:
        ftp= FtpLoader(ftp_ip, ftp_port, ftp_id, ftp_pw)
        
    h5_dir = os.listdir(h5_path)
    for filename in h5_dir:
        if filename == '_':
            continue
        ftp.upload(filename, h5_path, '/h5/')
        os.remove(h5_path + filename)

def cleanAll():
    ftp= FtpLoader(ftp_ip, ftp_port, ftp_id, ftp_pw)
    cleanH5(ftp)
    cleanOnnx(ftp)