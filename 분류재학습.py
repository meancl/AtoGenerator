import sqlalchemy as db
import pandas as pd
import datetime
import numpy as np
from sqlalchemy import create_engine
import os
import random
from mylib.scaler import *
from mylib.featurenames import *
from mylib.modelpostfix import *
from mylib.onnxtransformer import *
from mylib.cleaner import *
# from mylib.quantization import *
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, Input, Dropout, BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

engine = create_engine('mysql://meancl:1234@221.149.119.60:2023/mjtradierdb')
conn = engine.connect()

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
y_condition = (br['fMaxPowerAfterBuyWhile10'] >= 0.04)
y = np.where(y_condition, 1, 0)

######### 필수 !!!! ###############
reusable_model_name = 'gpu_test_Robust_recent'
model_name = 'reuse_gpu_test'

''' set scaler '''

modelTester = ModelTester(engine, conn)
ftp = FtpLoader("221.149.119.60", 2021, "ftp_user", "jin9409")
ftp.download(reusable_model_name + h5_, h5_path, '/h5/')
modelTester.setNpData(X)
modelTester.matchOldScaler(reusable_model_name + onnx_)
modelTester.fitScale()
X = modelTester.np_data

''' set prefix model name '''

call_back_last_filename = model_name + '_cls_' + modelTester.scale_method + '_recent'
call_back_acc_max_filename = model_name + '_cls_' + modelTester.scale_method + '_acc_max' 
call_back_loss_min_filename = model_name + '_cls_' + modelTester.scale_method + '_lss_min'


''' write scale data to db '''
model_name_list = [call_back_last_filename, call_back_acc_max_filename, call_back_loss_min_filename]
for mn in model_name_list:
    modelTester.writeScalerToDB(
                    feature_names=feature_names_102,
                    model_name=mn+onnx_,
                    )
    
''' make random seed '''
random_seed = int(1 / (random.random() + 0.00000001) * 100)

''' split train test validation data '''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=random_seed)

print('X_train : ', X_train.shape)
print('y_train : ', y_train.shape)
print('X_train : ', X_valid.shape)
print('y_train : ', y_valid.shape)
print('X_test  : ', X_test.shape)
print('y_test  : ', y_test.shape)

# Reusable 
compile_pass = True
model = tf.keras.models.load_model(h5_path + reusable_model_name + h5_, compile=compile_pass) # compile=True면 별도의 compile 작업이 필요없다
model.summary()

''' set model compile method '''
if not compile_pass:
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


''' set epoch and batch size '''
EPOCH = 1
BATCH_SIZE = 64

''' make check points '''
checkpoint_acc = ModelCheckpoint(h5_path + call_back_acc_max_filename + h5_,
                            monitor='val_accuracy',
                            verbose=1,
                            save_best_only=True,
                            mode='max')

checkpoint_loss = ModelCheckpoint(h5_path + call_back_loss_min_filename + h5_,
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=True,
                            mode='min')

checkpoint_last = ModelCheckpoint(h5_path + call_back_last_filename + h5_,
                            verbose=1,
                            save_freq = 'epoch'
                            )

history = model.fit(X_train, y_train, 
      validation_data=(X_valid, y_valid),
      epochs=EPOCH, 
      batch_size=BATCH_SIZE, 
      callbacks=[checkpoint_acc, checkpoint_loss, checkpoint_last] # checkpoint 콜백
     )

''' convert h5 to onnx '''
transformToOnnx(model_name_list)
# quantize_onnx_model_list(model_name_list) # 양자화하면 성능이 쓰레기

clean()