from mylib.onnxtransformer import *
from mylib.cleaner import *


fn_list = ['fMax10_004_v2_cls_Standard_acc_max', 'fMax10_004_v2_cls_Standard_recent', 'fMax10_004_v2_cls_Standard_lss_min']

transformToOnnx(fn_list)
clean()