import socket
import threading


class UDPServer:
    def __init__(self,address,buffer_size = 1024):
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.address = address
        self.buffer_size = buffer_size

    def setDataCallBack(self,callBack):
        self.callback = callBack    

    def start(self):
        self.stop = False
        self.thread = threading.Thread(target=self.__listen__)
        self.thread.start() 

    def stop(self):
        self.sock.close()
               

    def __listen__(self):

        self.sock.bind(self.address)

        while True:
            data, address = self.sock.recvfrom(self.buffer_size)
            self.callback(data)         
        
            
            