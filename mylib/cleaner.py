import os.path
import os
from mylib.modelpostfix import *
from mylib.ftploader import *
        
def clean(data_list=None):
    if data_list is None:
        h5_dir = os.listdir(project_absolute_path + '/h5')
        onnx_dir = os.listdir(project_absolute_path + '/onnx')

        ftp= FtpLoader("221.149.119.60", 2021, "ftp_user", "jin9409")

        for filename in h5_dir:
            if filename == '_':
                continue
            ftp.upload(filename, project_absolute_path + '/h5/', '/h5/')
            os.remove(project_absolute_path + '/h5/' + filename)

        for filename in onnx_dir:
            if filename == '_':
                continue
            ftp.upload(filename, project_absolute_path + '/onnx/', '/onnx/')
            os.remove(project_absolute_path + '/onnx/' + filename)
    else:
        for filename in data_list:
            os.remove(project_absolute_path + '/h5/' + filename + h5)
