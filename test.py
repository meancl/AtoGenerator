# import pandas as pd
# import numpy as np
# from sqlalchemy import create_engine
# from mylib.featurenames import *

# def matchOldScaler(engine, model_name:str):
#         sql = f"select dTime from scaledatasdict where sModelName = '{model_name}' order by dTime desc"
#         dt = pd.read_sql(sql, con=engine)
#         if dt.shape[0] > 0:
#             date_ = dt.iloc[0]
#             sql2=  f"select * from scaledatasdict where sModelName = '{model_name}' and dTime = '{date_[0]}'"
#             al = pd.read_sql(sql2, con=engine)
#             al.sort_values(by=['nSeq'], inplace=True)
#             d0 = al['fD0'].to_numpy(dtype=np.float64)
#             d1 = al['fD1'].to_numpy(dtype=np.float64)
#             d2 = al['fD2'].to_numpy(dtype=np.float64)
#             scale_method = al['sScaleMethod'][0]

        


# engine = create_engine('mysql://meancl:1234@221.149.119.60:2023/mjtradierdb')
# conn = engine.connect()
# matchOldScaler(engine, "fProfit_10_Robust_c.onnx")


from mylib.scaler import *
from mylib.featurenames import *
from mylib.modelpostfix import *
import tensorflow as tf
import os

''' convert h5 to onnx '''
for mn in ['abcde_Robust_acc_max', 'abcde_Robust_lss_min', 'abcde_Robust_recent']:
    # h5 to pb
    model_convert = tf.keras.models.load_model(h5_path + mn + h5, compile=False)
    model_convert.save(tmp_model_path, save_format="tf")

    # pb to onnx 
    os.system('python -m tf2onnx.convert --saved-model ' +  tmp_model_path + ' --output ' + onnx_path + mn + onnx + ' --opset 13')


