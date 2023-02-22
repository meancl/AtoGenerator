import numpy as np
import pandas as pd
from datetime import datetime
import sqlalchemy as db
from sqlalchemy import select, insert, update
from sqlalchemy import desc

MINMAX = 'MinMax'
ROBUST = 'Robust'
STANDARD = 'Standard'


class ModelTester:
    def __init__(self, engine, conn):
        self.engine = engine
        self.conn = conn


    def setNpData(self, np_data):
        self.np_data = np_data.copy()

    def setScaler(self, scale_method:str):
            self.scale_method = scale_method

            col_num = self.np_data.shape[1]

            if self.scale_method == MINMAX:
                self.d0 = self.np_data.min(axis=0)
                self.d1 = self.np_data.max(axis=0)
                self.d2 = self.d0
            elif self.scale_method == STANDARD:
                self.d0 = self.np_data.mean(axis=0)
                self.d1 = self.np_data.std(axis=0)
                self.d2 = np.zeros(col_num, dtype=np.float64)
            elif self.scale_method == ROBUST:
                self.d0 = np.median(self.np_data, axis=0)
                self.d1 = np.quantile(self.np_data, q=0.75, axis=0)
                self.d2 = np.quantile(self.np_data, q=0.25, axis=0)
            else:
                raise Exception()
    
    '''
    스케일 변수들을 db에 삽입한다.
    '''
    def writeScalerToDB(self, feature_names:list, model_name:str):
        try:
            today = datetime.today()
            scaleMethod = self.scale_method
            sModel = model_name
            
            d0_s = self.d0
            d1_s = self.d1
            d2_s = self.d2
            
            for idx, col in enumerate(feature_names):
                sVar = col
                
                d0 = d0_s[idx]
                d1 = d1_s[idx]
                d2 = d2_s[idx]
                
                denom = d1 - d2
                if denom == 0:
                    d1 = 1
                    d2 = 0 
                
                table = db.Table('scaledatasdict', db.MetaData(), autoload=True, autoload_with=self.engine)
                query = db.insert(table).values( {'dTime': today, 'sScaleMethod':scaleMethod, 'sVariableName':sVar, 
                                'sModelName':sModel, 'fD0':d0, 'fD1':d1, 'fD2':d2, 'nSeq':idx})

                result_proxy = self.conn.execute(query)
                result_proxy.close()
            print('put scale to ', sModel, ' ends')
        except Exception as ex:
            print(ex)
            return

    def fitScale(self):
        row_num = self.np_data.shape[0]
        col_num = self.np_data.shape[1]
        
        d0_s = self.d0
        d1_s = self.d1
        d2_s = self.d2
        
        for i in range(col_num):
            d0 = d0_s[i]
            d1 = d1_s[i]
            d2 = d2_s[i]
            
            denom = d1 - d2
            if denom == 0:
                denom = 1
                    
            for j in range(row_num):
                data = self.np_data[j, i]
                self.np_data[j, i] = (data - d0) / denom
                
        self.np_data = self.np_data.astype(np.float32)

    def matchOldScaler(self, model_name:str):
        sql = f"select dTime from scaledatasdict where sModelName = '{model_name}' order by dTime desc"
        dt = pd.read_sql(sql, con=self.engine)
        if dt.shape[0] > 0:
            date_ = dt.iloc[0]
            sql2=  f"select * from scaledatasdict where sModelName = '{model_name}' and dTime = '{date_[0]}'"
            al = pd.read_sql(sql2, con=self.engine)
            al.sort_values(by=['nSeq'], inplace=True)
            self.d0 = al['fD0'].to_numpy(dtype=np.float64)
            self.d1 = al['fD1'].to_numpy(dtype=np.float64)
            self.d2 = al['fD2'].to_numpy(dtype=np.float64)
            self.scale_method = al['sScaleMethod'][0]

        else :
            raise Exception()


