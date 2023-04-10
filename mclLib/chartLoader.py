from mclLib.ftploader import *
from mclLib.server_info import *
from mclLib.chartpostfix import *

chart_names = []

ftp= FtpLoader(ftp_ip, ftp_port, ftp_id, ftp_pw)
for cn in chart_names :
    ftp.download(cn+txt_, Every1min_path, just_Every1min)