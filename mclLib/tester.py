def testClassification(y_test, y_pred_list, suc_crit=1.0, fail_crit=0.0, suc_pass_ratio=1.0, fail_pass_ratio=1.0):
    true_num = 0
    false_num = 0

    true_pred_true = 0
    true_pred_false = 0
    false_pred_true = 0
    false_pred_false = 0
    
    suc_line = round(len(y_pred_list) * suc_pass_ratio)
    fail_line = round(len(y_pred_list) * fail_pass_ratio)

    len_y = y_test.shape[0]

    true_pred_list = []
    false_pred_list = []

    try:
        for i in range(len_y):

            # 데이터의 분포를 체크
            if(y_test[i] == 1.0):
                true_num += 1
            elif(y_test[i] == 0.0):
                false_num += 1
            else:
                raise Exception(f"invalie y_data {i}번째 인덱스 : {y_test[i]}")
            # PREDICT 0
            pass_0 = False
            pass_0_check = 0 
            for pred in y_pred_list:
                if pred[i] <= fail_crit :
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
                if pred[i] >= suc_crit :
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
        print('총 횟수 : ', false_pred_true+ false_pred_false, ',  타겟기준 : ', fail_crit)
        print('실제 0 : ', false_pred_true)
        print('실제 1 : ', false_pred_false)
        print('정답비율 : ', (false_pred_true / (1 if false_pred_true+false_pred_false == 0 else false_pred_true+false_pred_false)) * 100, '(%)', end='\n\n')
            
        print('============ predict 1 =============')
        print('총 횟수 : ', true_pred_true+ true_pred_false, ', 타겟기준 : ', suc_crit)
        print('실제 1 : ', true_pred_true)
        print('실제 0 : ', true_pred_false)
        print('정답비율 : ', (true_pred_true / (1 if true_pred_true+true_pred_false == 0 else true_pred_true+true_pred_false)) * 100, '(%)', end='\n\n')
    except Exception as ex:
        print('테스트 도중 예외 발생. ', ex)

    return true_pred_list, false_pred_list