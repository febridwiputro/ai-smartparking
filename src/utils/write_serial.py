import serial
import time

# Konfigurasi port serial
ser = serial.Serial(
    port='COM4',          # Ganti dengan port yang sesuai, misalnya 'COM3' untuk Windows atau '/dev/ttyUSB0' untuk Linux
    baudrate=115200,        # Sesuaikan baud rate dengan perangkat Anda
    timeout=1
)

# Fungsi untuk menulis data ke serial
def write_serial(data):
    if ser.is_open:       # Pastikan port serial terbuka
        # Mengirim data dalam format byte
        ser.write(data.encode())  # .encode() digunakan untuk mengubah string menjadi byte
        print(f"Data '{data}' berhasil dikirim.")
    else:
        print("Port serial tidak terbuka.")


try:
    if not ser.is_open:
        ser.open()
 
    write_serial("Hello, Serial Port!")
    time.sleep(1)  # Tunggu sebentar untuk memastikan data terkirim
except Exception as e:
    print(f"Terjadi kesalahan: {e}")
finally:
    ser.close()  # Pastikan untuk menutup port serial setelah selesai
