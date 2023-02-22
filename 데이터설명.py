import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import graphviz
import os
from sklearn import tree, ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import shap
from sklearn.feature_selection import chi2, SelectKBest, f_classif, mutual_info_classif, f_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/' # windows에서 graphviz를 사용하기 위해 변수경로 설정 

engine = create_engine('mysql://meancl:1234@221.149.119.60:2023/mjtradierdb')
conn = engine.connect()

br_full_data = pd.read_sql_table('buyreports', conn)
# Filtering
get_filter = ( br_full_data['isAllBuyed'] == 1) & ( br_full_data['isAllSelled'] == 1) & (br_full_data['nBuyVolume'] > 0)
br = br_full_data[get_filter]
feature_names_102 =  [   
        'nBuyStrategyIdx',
        'nRqTime' , 
        'fStartGap' ,
        'fPowerWithOutGap' , 
        'fPower' , 
        'fPlusCnt07' , 
        'fMinusCnt07' , 
        'fPlusCnt09' , 
        'fMinusCnt09' ,
        'fPowerJar' , 
        'fOnlyDownPowerJar' , 
        'fOnlyUpPowerJar' , 
        'nTradeCnt' , 
        'nChegyulCnt' , 
        'nHogaCnt' , 
        'nNoMoveCnt' , 
        'nFewSpeedCnt' ,
        'nMissCnt' , 
        'lTotalTradeVolume' , 
        'lTotalBuyVolume' , 
        'lTotalSellVolume' ,
        'nAccumUpDownCount' ,
        'fAccumUpPower' , 
        'fAccumDownPower' ,
        'lTotalTradePrice' , 
        'lTotalBuyPrice' , 
        'lTotalSellPrice' , 
        'lMarketCap' , 
        'nAccumCountRanking' , 
        'nMarketCapRanking' , 
        'nPowerRanking' , 
        'nTotalBuyPriceRanking' , 
        'nTotalBuyVolumeRanking' ,
        'nTotalTradePriceRanking' ,
        'nTotalTradeVolumeRanking' ,
        'nTotalRank' , 
        'nMinuteTotalRank' , 
        'nMinuteTradePriceRanking' ,
        'nMinuteTradeVolumeRanking' , 
        'nMinuteBuyPriceRanking' , 
        'nMinuteBuyVolumeRanking' ,
        'nMinutePowerRanking' , 
        'nMinuteCountRanking' ,
        'nMinuteUpDownRanking' ,
        'nFakeBuyCnt' , 
        'nFakeAssistantCnt' ,
        'nFakeResistCnt' , 
        'nPriceUpCnt' , 
        'nPriceDownCnt' ,
        'nTotalFakeCnt' ,
        'nTotalFakeMinuteCnt' ,
        'nUpCandleCnt' , 
        'nDownCandleCnt' ,
        'nUpTailCnt' , 
        'nDownTailCnt' ,
        'nShootingCnt' ,
        'nCandleTwoOverRealCnt' ,
        'nCandleTwoOverRealNoLeafCnt' , 
        'fSpeedCur' , 
        'fHogaSpeedCur' ,
        'fTradeCur' , 
        'fPureTradeCur' , 
        'fPureBuyCur' , 
        'fHogaRatioCur' ,  
        'fSharePerHoga' , 
        'fSharePerTrade' ,
        'fHogaPerTrade' , 
        'fTradePerPure' , 
        'fMaDownFsVal' , 
        'fMa20mVal' , 
        'fMa1hVal' ,
        'fMa2hVal' ,
        'fMaxMaDownFsVal' ,
        'fMaxMa20mVal' ,
        'fMaxMa1hVal' ,
        'fMaxMa2hVal' ,
        'nMaxMaDownFsTime' ,
        'nMaxMa20mTime' ,
        'nMaxMa1hTime' ,
        'nMaxMa2hTime' ,
        'nDownCntMa20m' ,
        'nDownCntMa1h' ,
        'nDownCntMa2h' ,
        'nUpCntMa20m' ,
        'nUpCntMa1h' ,
        'nUpCntMa2h' ,
        'fMSlope' ,
        'fISlope' ,
        'fTSlope' ,
        'fHSlope' ,
        'fRSlope' ,
        'fDSlope' ,
        'fMAngle' ,
        'fIAngle' ,
        'fTAngle' ,
        'fHAngle' ,
        'fRAngle' ,
        'fDAngle' ,
        'nCrushCnt' ,
        'nCrushUpCnt' ,
        'nCrushDownCnt' ,
        'nCrushSpecialDownCnt' 
]
feature_size = len(feature_names_102)
feat_name = np.array(feature_names_102)
features = feature_names_102
X = br[
   features
]

X = X.to_numpy(dtype=np.float32)
y_condition = (br['fMaxPowerAfterBuyWhile30'] >= 0.05)
br.loc[y_condition , 'target'] = 1
br.loc[~y_condition, 'target'] = 0
y = br['target']

y = y.to_numpy(dtype=np.float32)
min_s = None
max_s = None
mean_s = None
std_s = None
zero_s = None
median_s = None
iqr3_s = None
iqr1_s = None

MINMAX = 'MinMax'
ROBUST = 'Robust'
STANDARD = 'Standard'

def setScaler(p_data):
    np_data = p_data.to_numpy(dtype=np.float32)

    row_num = np_data.shape[0]
    col_num = np_data.shape[1]
    
    # global 사용
    global min_s
    global max_s
    global mean_s
    global std_s
    global zero_s
    global median_s
    global iqr3_s
    global iqr1_s
    
    # MinMaxScaler
    min_s = np_data.min(axis=0)
    max_s = np_data.max(axis=0)
    
    # StandardScaler
    mean_s = np_data.mean(axis=0)
    std_s = np_data.std(axis=0)
    zero_s = np.zeros(col_num, dtype=np.float32)
    
    # RobustScaler
    median_s = np.median(np_data, axis=0)
    iqr3_s = np.quantile(np_data, q=0.75, axis=0)
    iqr1_s = np.quantile(np_data, q=0.25, axis=0)
    
def fitMyScaler(p_data, scale_method='MinMax'):
    np_data = p_data.to_numpy(dtype=np.float32)

    row_num = np_data.shape[0]
    col_num = np_data.shape[1]
    
    d0_s = None
    d1_s = None
    d2_s = None
    
    if scale_method == 'MinMax':
        d0_s = min_s
        d1_s = max_s
        d2_s = min_s
    elif scale_method == 'Standard':
        d0_s = mean_s
        d1_s = std_s
        d2_s = zero_s
    elif scale_method == 'Robust':
        d0_s = median_s
        d1_s = iqr3_s
        d2_s = iqr1_s
    else :
        print('해당하는 스케일함수가 없습니다.')
        return
    
    for i in range(col_num):
        
        d0 = d0_s[i]
        d1 = d1_s[i]
        d2 = d2_s[i]
        
        denom = d1 - d2
        if denom == 0:
            denom = 1
                
        for j in range(row_num):
            np_data[j, i] = (np_data[j, i] - d0) / denom
            
            
    return np_data
    
scale_method = ROBUST

clf = tree.DecisionTreeClassifier(max_depth = 5)
#clf = tree.ExtraTreeClassifier(max_depth= 5)
clf = clf.fit(X, y)

dot_data = tree.export_graphviz(clf,   # 의사결정나무 모형 대입
                               out_file = None,  # file로 변환할 것인가
                               feature_names = features,  # feature 이름
                               # class_names = np.array(['fail', 'suc']),  # target 이름
                               filled = True,           # 그림에 색상을 넣을것인가
                               rounded = True,          # 반올림을 진행할 것인가
                               special_characters = True)   # 특수문자를 사용하나

graph = graphviz.Source(dot_data)
print(graph)

rf = RandomForestClassifier(n_estimators=1)
rf.fit(X, y)

rf.feature_importances_ # 피처들의 중요도

sorted_idx = rf.feature_importances_.argsort()
plt.figure(figsize=(20, 20))
plt.barh(feat_name[sorted_idx], rf.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")

perm_importance = permutation_importance(rf, X, y)

sorted_idx = perm_importance.importances_mean.argsort()
plt.figure(figsize=(20, 20))
plt.barh(feat_name[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")