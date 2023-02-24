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


ftp= FtpLoader("221.149.119.60", 2021, "ftp_user", "jin9409")

engine = create_engine('mysql://meancl:1234@221.149.119.60:2023/mjtradierdb')
conn = engine.connect()
# ftp.upload("fMax30_5_v1_Robust_acc_max.h5", './h5/', '/h5/')
t_model_name = ['reuse_gpu_test_Robust_lss_min', 'reuse_gpu_test_Robust_acc_max', 'reuse_gpu_test_Robust_recent']
''' get db data '''
br_full_data = pd.read_sql_table('buyreports', conn)

''' db data filter '''
get_filter = ( br_full_data['isAllBuyed'] == 1) & ( br_full_data['isAllSelled'] == 1) 
# extract_filter = (br_full_data['dTradeTime'] >= datetime.datetime(2023, 2, 16))
br = br_full_data[get_filter]
# br_extract = br_full_data[extract_filter]

''' get features name'''
feature_names_102 =  f_name_102
feature_size = len(feature_names_102)

''' set X data '''
X = br[feature_names_102].to_numpy(dtype=np.float64)

''' set y data '''
y_condition = (br['fMaxPowerAfterBuyWhile10'] >= 0.04)
y = np.where(y_condition, 1, 0)

x_datas = [] 
models = []
y_predict = []

modelTester = ModelTester(engine, conn)
for i in t_model_name:
    ftp.download(i + h5, project_absolute_path + '/h5/', '/h5/')
    modelTester.setNpData(X)
    modelTester.matchOldScaler(i + onnx)
    modelTester.fitScale()
    x_datas.append(modelTester.np_data)
    model_tmp = tf.keras.models.load_model(h5_path + i + h5, compile=True)
    models.append(model_tmp)

y_test =y

for idx, md in enumerate(models):
    pred = md.predict(x_datas[idx])
    y_predict.append(pred)

# test
one = 0
zero = 0

ac = 0
fl = 0
d_ac = 0
d_fl = 0

suc_crit = 0.8 # 1이라 판정할 기준
fl_crit = 0.5 # 0이라 판정할 기준

len_y = y_test.shape[0]

suc_ratio = 0.65
suc_line = round(len(y_predict) * suc_ratio)

fail_ratio = 1.0
fail_line = round(len(y_predict) * fail_ratio)

for i in range(len_y):
    if(y_test[i] == 1.0):
        one += 1
    elif(y_test[i] == 0.0):
        zero += 1
        
    # PREDICT 0
    pass_0 = False
    pass_0_check = 0 
    for pred in y_predict:
        if pred[i][0] < fl_crit :
            pass_0_check += 1
            
    if pass_0_check >= fail_line:
        pass_0 = True
            
    if pass_0: 
        if(y_test[i] == 0.0):
            d_ac += 1
        else:
            d_fl += 1
    
    # PREDICT 1
    pass_1 = False
    pass_1_check = 0 
    for pred in y_predict:
        if pred[i][0] > suc_crit :
            pass_1_check += 1
            
    if pass_1_check >= suc_line:
        pass_1 = True
            
    if pass_1: 
        if(y_test[i] == 1.0):
            ac += 1
        else:
            fl += 1

   
    
print('총량 : ', one+zero)
print('0 : ', zero, ', 비율 : ', (zero / (1 if one+zero == 0 else one+zero)) * 100, '(%)')
print('1 : ', one, ', 비율 : ', (one / (1 if one+zero == 0 else one+zero)) * 100, '(%)', end='\n\n')

print('============ predict 0 =============')
print('총 횟수 : ', d_ac+ d_fl, ',  타겟기준 : ', fl_crit)
print('실제 0 : ', d_ac)
print('실제 1 : ', d_fl)
print('정답비율 : ', (d_ac / (1 if d_ac+d_fl == 0 else d_ac+d_fl)) * 100, '(%)', end='\n\n')
    
print('============ predict 1 =============')
print('총 횟수 : ', ac+ fl, ', 타겟기준 : ', suc_crit)
print('실제 1 : ', ac)
print('실제 0 : ', fl)
print('정답비율 : ', (ac / (1 if ac+fl == 0 else ac+fl)) * 100, '(%)', end='\n\n')

clean(t_model_name)