import serial
import time

serial_connection = {}

def connect_to_com_port(com_port,baud_rate=115200):
    if com_port in serial_connection:
        print("COM: ", com_port)
        return serial_connection[com_port]

    try:
        close_connection()
        ser = serial.Serial(com_port,baud_rate)
        serial_connection[com_port] = ser
        ser.open()
        print("ser: ", ser)

        return ser
    except serial.SerialException as e:
        print("ERROR COK")
        return None
    
def send_data(com_port,data):
    if com_port in serial_connection:
        ser = serial_connection[com_port]
        if ser.is_open:
            ser.write(data)
    else:
        print("NO CONNECTION")

def close_connection():
    for ser in serial_connection.items():
        if ser.is_open:
            ser.close()