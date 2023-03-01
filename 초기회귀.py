import sqlalchemy as db
import pandas as pd
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
import keras

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
y = br['fMaxPowerAfterBuyWhile30'].to_numpy(dtype=np.float32)

''' set scaler '''
scale_method = STANDARD

modelTester = ModelTester(engine, conn)
modelTester.setNpData(X)
modelTester.setScaler(scale_method)
modelTester.fitScale()
X = modelTester.np_data


''' set prefix model name '''
model_name = 'fMax30_005'
call_back_last_filename = model_name + '_reg_' + scale_method + '_recent'
call_back_mae_min_filename = model_name + '_reg_' + scale_method + '_mae_min' 
call_back_loss_min_filename = model_name + '_reg_' + scale_method + '_lss_min'

''' write scale data to db '''
model_name_list = [call_back_last_filename, call_back_mae_min_filename, call_back_loss_min_filename]
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

''' set dnn model '''
''' dropout example : x = Dropout(.1)(x) '''
nInputDim = feature_size
nOutputDim = 1
main_input = Input(shape=(nInputDim), name='input')
x = Dense(1024, activation='relu')(main_input)
x = Dropout(.1)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(.1)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(.1)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(.1)(x)
x = Dense(2048, activation='relu')(x)
x = Dropout(.1)(x)
x = Dense(2048, activation='relu')(x)
x = Dropout(.1)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(.1)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(.1)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(.1)(x)
main_output = Dense(nOutputDim, name='output')(x)
model = Model(inputs=main_input, outputs=main_output)
model.summary()

''' set epoch and batch size '''
EPOCH = 150
BATCH_SIZE = 100

''' set model compile method '''
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

''' make check points '''
checkpoint_mae = ModelCheckpoint(h5_path + call_back_mae_min_filename + h5_,
                            monitor='val_mae',
                            verbose=1,
                            save_best_only=True,
                            mode='min')

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
      callbacks=[checkpoint_mae, checkpoint_loss, checkpoint_last] # checkpoint 콜백
     )

''' convert h5 to onnx '''
transformToOnnx(model_name_list)
# quantize_onnx_model_list(model_name_list) # 양자화하면 성능이 쓰레기.. 

clean()