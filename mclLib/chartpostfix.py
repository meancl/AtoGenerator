chart_absolute_path = r'C:/Users/MJ/source/repos/MJTradier_AI/임시저장소/chart'

just_Every1min = '/Every1min/'
just_AfterBuy10sec = '/AfterBuy10sec/'

Every1min_path = chart_absolute_path + just_Every1min 
AfterBuy10sec_path = chart_absolute_path + just_AfterBuy10sec

txt_ = '.txt'

Every1min_col_names = [ 'idx', 'time', 'open', 'high', 'low', 'close', 'totalVolume', 'tradeRatio', 'count', 'totalPrice', 'buyPrice', 'sellPrice', 'accumUpPower', 'accumDownPower', 'diffRatio', 'initAngle', 'maxAngle', 'medianAngle', 'hourAngle', 'recentAngle', 'dAngle', 'ma0', 'ma1', 'ma2', 'downMa0Count', 'downMa1Count', 'downMa2Count', 'upMa0Count', 'upMa1Count', 'upMa2Count', 'nSummationRanking', 'nSummationMove', 'nMinuteRanking' ]
AfterBuy10sec_col_names = [ 'idx', 'time', 'open', 'high', 'low', 'close', 'totalvolume', 'tradeRatio', 'count', 'totalPrice', 'buyPrice', 'sellPrice', 'accumUpPower', 'accumDownPower', 'diffRatio', 'medianAngle', 'hourAngle', 'recentAngle', 'dAngle', 'entireMedianAngle', 'entireHourAngle', 'entireRecentAngle', 'entireDAngle', 'ma0', 'ma1', 'ma2', 'downMa0Count', 'downMa1Count', 'downMa2Count', 'upMa0Count', 'upMa1Count', 'upMa2Count' ]