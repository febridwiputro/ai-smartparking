import os, sys
import time
import serial
import serial.tools.list_ports as stl

this_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# print("this_path: ", this_path)
sys.path.append(this_path)

from config.logger import logger


class Arduino:
    def __init__(self, baudrate: int, driver: str = None, com: str = None, serial_number: str = None,count = 0, is_send=False):
        self.texts = []
        self.baudrate = baudrate
        self.serial_number = serial_number

        if is_send == False:
        # print("self.get_data_micro(): ", self.get_data_micro())
        
            try:
                if serial_number:
                    print("Serial Number:", serial_number.center(50, "-"))
                    self.com = self.find_serial_port(driver, serial_number)
                elif com is None:
                    self.com = self.find_serial_port(driver)
                else:
                    self.com = com
                
                # if self.com is None:
                #     logger.write("COM port could not be determined.", logger.ERROR)
                #     raise ValueError('COM Port not determined.')

                print("self.com: ", self.com)
                
                self.connection = serial.Serial(
                    baudrate=baudrate,
                    timeout=1,
                    write_timeout=0.1,
                    port= self.com
                )
            
            except serial.SerialException as e:
                logger.write(f"Failed to open COM PORT Arduino: {e}", logger.ERROR)
                raise PermissionError(f"Failed to open COM PORT Arduino: {e}")
    
    def get_data_micro(self):
        ports = stl.comports()
        for port in ports:
            print(port.serial_number)
    
    def connected(self):
        return self.connection.is_open if self.connection else False
    
    def write(self, message):
        self.texts.append(message)
    
    def write_count(self, message):
        self.texts = message

    def sending(self):
        if not self.connected():
            # logger.write("Arduino is not connected.", logger.ERROR)
            return
        
        time.sleep(0.25)
        # text = ''.join(self.texts)
        text = self.texts
        
        if text:
            print(f"sending: {text}")
            try:
                self.connection.write(str(text).encode('utf-8'))
                # self.connection.write(text.encode('utf-8'))
            except serial.serialutil.SerialException as e:
                print(f"An error occurred: {e}")
                self.reconnect()
            
            self.texts = []
            self.connection.flush()
    
    def read(self):
        if not self.connected():
            logger.write("Cannot read, Arduino is not connected.", logger.ERROR)
            return None
        
        try:
            return str(self.connection.read_until().decode()).upper()
        except Exception as e:
            logger.write(f"Read error: {e}", logger.ERROR)
            return None
    
    def write_with_com(self, count, com):
        if not self.connected():
            # logger.write("Arduino is not connected.", logger.ERROR)
            return
        
        try:        
            time.sleep(0.25)
            # text = ''.join(count)
            text = count
            self.connection.port = com
            
            if text:
                print(f"sending: {text}")
                try:
                    self.connection.write(text.encode('utf-8'))
                except serial.serialutil.SerialException as e:
                    print(f"An error occurred: {e}")
                    self.reconnect()
                
                self.texts = []
                self.connection.flush()

        except Exception as e:
            print(f"write_with_com error: {e}")
    
    def reconnect(self):
        try:
            self.connection.close()
            self.connection = serial.Serial(
                baudrate=self.baudrate,
                timeout=1,
                write_timeout=0.1,
                port=self.com
            )
            print("Reconnected to Arduino.")
        except serial.SerialException as e:
            logger.write(f"Failed to reconnect: {e}", logger.ERROR)
    
    @staticmethod
    def find_serial_port(driver: str, serial_number: str = None):
        ports = stl.comports()
        if serial_number is None:
            for port in ports:
                if driver in port.description.upper():
                    print("Found driver port:", port.device)
                    return port.device
        else:
            for port in ports:
                if serial_number == str(port.serial_number):
                    print("Found serial number port:", port.device)
                    return port.device
        
        print("Serial number not found".center(50, "-"))
        return None


if __name__ == "__main__":
    try:
        # ard = Arduino(115200, serial_number="D200RBECA")
        ard = Arduino(115200, com= 'COM6')
        time.sleep(5)

        ard.write(
            # 'M,2,lantai_2;S,1,lantai_2;M,4,lantai_2;M,5,lantai_2;S,3,lantai_2;M,18,lantai_2;S,15,lantai_2;M,11,lantai_2;S,16,lantai_2;M,13,lantai_2;S,17,lantai_2;M,10,lantai_2;S,6,lantai_2;S,12,lantai_2;M,9,lantai_2;S,14,lantai_2;M,8,lantai_2;S,7,lantai_2;')
            '10')
        ard.sending()
        time.sleep(50)
    except Exception as e:
        print(f"An error occurred: {e}")
