import os.path
import os

from mclLib.modelpostfix import *
from mclLib.ftploader import *
from mclLib.server_info import *
from mclLib.chartpostfix import *

def removeTestData(h5_list:list=None, onnx_list:list=None):
    if h5_list is not None:
        for filename in h5_list:
            os.remove(h5_path + filename + h5_)

    if onnx_list is not None:
        for filename in onnx_list:
            os.remove(onnx_path + filename + onnx_)

    if h5_list is None and onnx_list is None :
        onnx_dir = os.listdir(onnx_path) 
        h5_dir = os.listdir(h5_path)

        for filename in onnx_dir:
            if filename == '_':
                continue
            os.remove(onnx_path + filename)
        
        for filename in h5_dir:
            if filename == '_':
                continue
            os.remove(h5_path + filename)

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

def removeChart():
    every1m_dir = os.listdir(Every1min_path)
    for filename in every1m_dir:
        if filename == '_':
            continue
        os.remove(Every1min_path + filename)
    
    afterBuy10sec_dir = os.listdir(AfterBuy10sec_path)
    for filename in afterBuy10sec_dir:
        if filename == '_':
            continue
        os.remove(AfterBuy10sec_path + filename)

def cleanAll():
    ftp= FtpLoader(ftp_ip, ftp_port, ftp_id, ftp_pw)
    cleanH5(ftp)
    cleanOnnx(ftp)