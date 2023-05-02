def testClassification(y_test, y_pred_list, crit_val = 0.5, suc_pass_ratio=1.0, fail_pass_ratio=1.0):
    
    print('분류평가를 시작합니다...')

    true_num = 0
    false_num = 0

    true_pred_true = 0
    true_pred_false = 0
    false_pred_true = 0
    false_pred_false = 0
    
    suc_line = max(round(len(y_pred_list) * suc_pass_ratio), 1)
    fail_line = max(round(len(y_pred_list) * fail_pass_ratio), 1)
    
    len_y = y_test.shape[0]

    print('======================================')
    print('len of models : ', len(y_pred_list))
    print(f'suc_line : {suc_line} ({suc_pass_ratio})')
    print(f'fail_line : {fail_line} ({fail_pass_ratio})')
    print('crit value : ', crit_val)
    print('======================================', end='\n\n')


    true_pred_list = []
    false_pred_list = []

    try:
        for i in range(len_y):

            # 데이터의 분포를 체크
            if(y_test[i] > crit_val):
                true_num += 1
            else:
                false_num += 1
            
            # PREDICT 0
            pass_0 = False
            pass_0_check = 0 
            for pred in y_pred_list:
                if pred[i] <= crit_val :
                    pass_0_check += 1
            
            # 0 예상이 fail_pass_ratio 율만큼 측정됐다면
            if pass_0_check >= fail_line:
                pass_0 = True
                false_pred_list.append(0)
            else:
                false_pred_list.append(1)
            
            # 0이라 예측했는데 
            if pass_0: 
                # 실제도 0이라면
                if(y_test[i] == 0.0):
                    false_pred_true += 1
                else: # 예측이 틀렸다면
                    false_pred_false += 1
        
            # PREDICT 1
            pass_1 = False
            pass_1_check = 0 
            for pred in y_pred_list:
                if pred[i] >= crit_val :
                    pass_1_check += 1
            
            # 1 예상이 suc_pass_ratio 율만큼 측정됐다면
            if pass_1_check >= suc_line:
                pass_1 = True
                true_pred_list.append(1)
            else:
                true_pred_list.append(0)
                
            # 1이라 예측했는데 
            if pass_1: 
                # 실제도 1이라면
                if(y_test[i] == 1.0):
                    true_pred_true += 1
                else: # 예측이 틀렸다면
                    true_pred_false += 1

        # 결과 출력
        print('총량 : ', true_num+false_num)
        print('0 : ', false_num, ', 비율 : ', (false_num / (1 if true_num+false_num == 0 else true_num+false_num)) * 100, '(%)')
        print('1 : ', true_num, ', 비율 : ', (true_num / (1 if true_num+false_num == 0 else true_num+false_num)) * 100, '(%)', end='\n\n')

        print('============ predict 0 =============')
        print('총 횟수 : ', false_pred_true+ false_pred_false)
        print('실제 0 : ', false_pred_true)
        print('실제 1 : ', false_pred_false)
        print('정답비율 : ', (false_pred_true / (1 if false_pred_true+false_pred_false == 0 else false_pred_true+false_pred_false)) * 100, '(%)', end='\n\n')
            
        print('============ predict 1 =============')
        print('총 횟수 : ', true_pred_true+ true_pred_false)
        print('실제 1 : ', true_pred_true)
        print('실제 0 : ', true_pred_false)
        print('정답비율 : ', (true_pred_true / (1 if true_pred_true+true_pred_false == 0 else true_pred_true+true_pred_false)) * 100, '(%)', end='\n\n')
    except Exception as ex:
        print('테스트 도중 예외 발생. ', ex)

    return true_pred_list, false_pred_list


# true가 제외할거
# false가 남아있어야 하는거
def testRegression(y_test, y_pred_list, crit_val = 0.015, suc_pass_ratio=1.0, fail_pass_ratio=1.0):
    print('회귀평가를 시작합니다...')
    true_num = 0
    false_num = 0

    true_pred_true = 0
    true_pred_false = 0
    false_pred_true = 0
    false_pred_false = 0
    
    len_y = y_test.shape[0]

    suc_line = max(round(len(y_pred_list) * suc_pass_ratio), 1)
    fail_line = max(round(len(y_pred_list) * fail_pass_ratio), 1)

    print('======================================')
    print('len of models : ', len(y_pred_list))
    print(f'suc_line : {suc_line} ({suc_pass_ratio})')
    print(f'fail_line : {fail_line} ({fail_pass_ratio})')
    print('crit value : ', crit_val)
    print('======================================', end='\n\n')
    
    true_pred_list = []
    false_pred_list = []

    try:
        for i in range(len_y):

            # 데이터의 분포를 체크
            if(y_test[i] < crit_val):
                true_num += 1
            else:
                false_num += 1
           
            pass_0 = False
            pass_0_check = 0 
            for pred in y_pred_list:
                if pred[i] >= crit_val :
                    pass_0_check += 1
            
            # 0 예상이 fail_pass_ratio 율만큼 측정됐다면
            if pass_0_check >= fail_line:
                pass_0 = True
                false_pred_list.append(0)
            else:
                false_pred_list.append(1)
            
            # 0이라 예측했는데 
            if pass_0: 
                # 실제도 0이라면
                if(y_test[i] >= crit_val):
                    false_pred_true += 1
                else: # 예측이 틀렸다면
                    false_pred_false += 1
        
            # PREDICT 1
            pass_1 = False
            pass_1_check = 0 
            for pred in y_pred_list:
                if pred[i] < crit_val :
                    pass_1_check += 1
            
            # 1 예상이 suc_pass_ratio 율만큼 측정됐다면
            if pass_1_check >= suc_line:
                pass_1 = True
                true_pred_list.append(1)
            else:
                true_pred_list.append(0)
                
            # 1이라 예측했는데 
            if pass_1: 
                # 실제도 1이라면
                if(y_test[i] < crit_val):
                    true_pred_true += 1
                else: # 예측이 틀렸다면
                    true_pred_false += 1

        # 결과 출력
        print('총량 : ', true_num+false_num)
        print('0 : ', false_num, ', 비율 : ', (false_num / (1 if true_num+false_num == 0 else true_num+false_num)) * 100, '(%)')
        print('1 : ', true_num, ', 비율 : ', (true_num / (1 if true_num+false_num == 0 else true_num+false_num)) * 100, '(%)', end='\n\n')

        print('============ predict 0 =============')
        print('총 횟수 : ', false_pred_true+ false_pred_false)
        print('실제 0 : ', false_pred_true)
        print('실제 1 : ', false_pred_false)
        print('정답비율 : ', (false_pred_true / (1 if false_pred_true+false_pred_false == 0 else false_pred_true+false_pred_false)) * 100, '(%)', end='\n\n')
            
        print('============ predict 1 =============')
        print('총 횟수 : ', true_pred_true+ true_pred_false)
        print('실제 1 : ', true_pred_true)
        print('실제 0 : ', true_pred_false)
        print('정답비율 : ', (true_pred_true / (1 if true_pred_true+true_pred_false == 0 else true_pred_true+true_pred_false)) * 100, '(%)', end='\n\n')
    except Exception as ex:
        print('테스트 도중 예외 발생. ', ex)

    return true_pred_list, false_pred_list