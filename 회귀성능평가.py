from mylib.ftploader import *
from mylib.modelpostfix import *
import tensorflow as tf
from sqlalchemy import create_engine
from mylib.scaler import *
import sqlalchemy as db
import pandas as pd
from mylib.featurenames import *
import numpy as np
from mylib.cleaner import *
from tensorflow.python.client import device_lib
from keras.models import load_model

print(device_lib.list_local_devices())



engine = create_engine('mysql://meancl:1234@221.149.119.60:2023/mjtradierdb')
conn = engine.connect()

test_model_name = ['test_v11_Robust_acc_max', 'test_v11_Robust_lss_min', 'test_v11_Robust_recent']
''' get db data '''
br_full_data = pd.read_sql_table('buyreports', conn)

''' db data filter '''
get_filter = ( br_full_data['isAllBuyed'] == 1) & ( br_full_data['isAllSelled'] == 1) 
br = br_full_data[get_filter]

''' get features name'''
feature_names_102 =  f_name_102
feature_size = len(feature_names_102)

''' set X data '''
X = br[feature_names_102].to_numpy(dtype=np.float64)

''' set y data '''
y = br['fMaxPowerAfterBuyWhile10'].to_numpy(dytpe=np.float32)

x_datas = [] 
models = []
y_predict = []

ftp= FtpLoader("221.149.119.60", 2021, "ftp_user", "jin9409")
modelTester = ModelTester(engine, conn)
for idx, mn in enumerate(test_model_name):
    ftp.download(mn + h5_, h5_path, '/h5/')
    modelTester.setNpData(X)
    modelTester.matchOldScaler(mn + onnx_)
    modelTester.fitScale()
    x_datas.append(modelTester.np_data)

    model = load_model(h5_path + mn + h5_)
    models.append(model)

    pred = model.predict(x_datas[idx])
    y_predict.append(pred)


y_test =y

#################################
############## test #############

len_y = y_test.shape[0]

suc_ratio = 0.65
fail_ratio = 1.0

for i in range(len_y):
        
    sum = 0

    for pred in y_predict:
        sum += pred[i]
    evg = sum / len(y_predict) # 평균 구하기

    # y_test[i] 와 evg 값의 비교         
   

   


clean(test_model_name)