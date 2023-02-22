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
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, Input, Dropout, BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint

engine = create_engine('mysql://meancl:1234@221.149.119.60:2023/mjtradierdb')
conn = engine.connect()

''' get db data '''
br_full_data = pd.read_sql_table('buyreports', conn)
table = db.Table('scaledatasdict', db.MetaData(), autoload=True, autoload_with=engine)

''' db data filter '''
get_filter = ( br_full_data['isAllBuyed'] == 1) & ( br_full_data['isAllSelled'] == 1) & (br_full_data['nBuyVolume'] > 0)
# extract_filter = (br_full_data['dTradeTime'] >= datetime.datetime(2023, 2, 16))
br = br_full_data[get_filter]
# br_extract = br_full_data[extract_filter]

''' get features name'''
feature_names_102 =  f_name_102
feature_size = len(feature_names_102)

''' set X data '''
X = br[feature_names_102].to_numpy(dtype=np.float64)

''' set y data '''
y_condition = (br['fMaxPowerAfterBuyWhile30'] >= 0.05)
y = np.where(y_condition, 1, 0)



''' set scaler '''
scale_method = ROBUST

modelTester = ModelTester(engine, conn)
modelTester.setNpData(X)
modelTester.setScaler(scale_method)
modelTester.fitScale()


''' set prefix model name '''
model_name = 'test'
call_back_last_filename = model_name + '_' + scale_method + '_recent'
call_back_acc_max_filename = model_name + '_' + scale_method + '_acc_max' 
call_back_loss_min_filename = model_name + '_' + scale_method + '_lss_min'



''' write scale data to db '''
model_name_list = [call_back_last_filename, call_back_acc_max_filename, call_back_loss_min_filename]
for mn in model_name_list:
    modelTester.writeScalerToDB(
                    feature_names=feature_names_102,
                    model_name=mn+onnx,
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
nInputDim = feature_size
nOutputDim = 1
main_input = Input(shape=(nInputDim), name='input')
x = Dense(512, activation='relu')(main_input)
x = Dropout(.1)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(.1)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(.1)(x)
x = Dense(2048, activation='relu')(x)
x = Dropout(.2)(x)
x = Dense(2048, activation='relu')(x)
x = Dropout(.2)(x)
x = Dense(2048, activation='relu')(x)
x = Dropout(.2)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(.1)(x)
x = Dense(512, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x = Dense(512, activation='relu')(x)
main_output = Dense(nOutputDim, activation='sigmoid', name='output')(x)
model = Model(inputs=main_input, outputs=main_output)
model.summary()

''' set epoch and batch size '''
EPOCH = 1
BATCH_SIZE = 500

''' set model compile method '''
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

''' make check points '''
checkpoint_acc = ModelCheckpoint(h5_path + call_back_acc_max_filename + h5,
                            monitor='val_accuracy',
                            verbose=1,
                            save_best_only=True,
                            mode='max')

checkpoint_loss = ModelCheckpoint(h5_path + call_back_loss_min_filename + h5,
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=True,
                            mode='min')

checkpoint_last = ModelCheckpoint(h5_path + call_back_last_filename + h5,
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
for mn in model_name_list:
    # h5 to pb
    
    model_convert = tf.keras.models.load_model(h5_path + mn + h5, compile=False)
    model_convert.save(tmp_model_path, save_format="tf")

    # pb to onnx 
    os.system('python -m tf2onnx.convert --saved-model ' +  tmp_model_path + ' --output ' + onnx_path + mn + onnx + ' --opset 13')


