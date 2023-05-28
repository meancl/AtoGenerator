import os.path
import os

from mclLib.modelpostfix import *
from mclLib.ftploader import *
from mclLib.server_info import *

def removeTestData(onnx_list:list=None):
    if onnx_list is not None:
        for filename in onnx_list:
            os.remove(onnx_path + filename + onnx_)

    if onnx_list is None :
        onnx_dir = os.listdir(onnx_path) 

        for filename in onnx_dir:
            if filename == '_':
                continue
            os.remove(onnx_path + filename)

def cleanOnnx(ftp=None):
    if ftp is None:
        ftp= FtpLoader(ftp_ip, ftp_port, ftp_id, ftp_pw)
    onnx_dir = os.listdir(onnx_path)

    for filename in onnx_dir:
        if filename == '_':
            continue
        ftp.upload(filename, onnx_path, '/onnx/')
        os.remove(onnx_path + filename)


def cleanAll():
    ftp= FtpLoader(ftp_ip, ftp_port, ftp_id, ftp_pw)
    cleanOnnx(ftp)