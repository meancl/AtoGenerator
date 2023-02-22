import ftplib

class FtpLoader:
    def __init__(self, ip, port, id, pwd):
        self.ftp=ftplib.FTP()
        self.ftp.connect(ip, port)
        self.ftp.login(id, pwd)

    def __del__(self):
        self.ftp.close()

    def upload(self, filename, in_path, out_path):
        self.ftp.cwd(out_path)
        with open(file=in_path + filename, mode ='rb') as wf:
            self.ftp.storbinary('STOR ' + filename, wf)

    def download(self, filename, out_path, in_path):
        self.ftp.cwd(in_path)  
        with open(file=out_path + filename, mode='wb') as wf:
            self.ftp.retrbinary('RETR ' + filename, wf.write)
    
    

