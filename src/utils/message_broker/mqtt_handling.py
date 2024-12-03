import paho.mqtt.client as mqtt
import time
import sys

class Mqtt_Handling:
    def __init__(self, broker, port, topic_sub = None, topic_pub = None, msg=None, publish=None, subscribe=None):
        self.client = mqtt.Client()
        self.broker = broker
        self.port = port
        self.msg = msg
        self.publish = publish
        self.subscribe = subscribe
        self.topic_sub = topic_sub
        self.topic_pub = topic_pub
        self.client.on_message = self.on_message
        self.client.on_connect = self.on_connect

        try:
            # Memulai koneksi MQTT dengan broker
            self.client.connect(host=self.broker, port=self.port, keepalive=60)
            self.client.loop_start()
            time.sleep(1)
            
            # Jika publish == True, menjalankan fungsi publish_message
            if self.publish:
                # while True:
                self.publish_message(topic_pub=self.topic_pub, msg=self.msg)
            
            # Jika subscribe == True, menjalankan fungsi subscribe_message
            if self.subscribe:
                self.subscribe_message()
                
        except Exception as e:
            print(f"MQTT Connection Failed: {e}")

    def on_message(self, client, userdata, message):
        print(f"Pesan diterima di topik {message.topic}: {message.payload.decode('utf-8')}")

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"Code RC: {rc}")
            print("Terhubung ke broker")
            # Subscribe ke topik yang ditentukan
            if self.publish:
                self.client.publish(f"{self.topic_pub}") 
                print(f"Publish to topic {self.topic_pub}")
        else:
            print(f"Gagal terhubung, kode: {rc}")
            
    # Fungsi untuk mengirim pesan melalui MQTT    
    def publish_message(self, topic_pub, msg):
        # topic = self.topic_pub
        # pesan = input("Silakan ketikkan pesan Anda: ")
        # result = self.client.publish(topic=topic, payload=pesan, qos=0)
        # if result.rc == 0:
        #     print("Pesan sukses dikirim")
        # else:
        #     print(f"Pesan gagal dikirim. Error code: {result.rc}")
        
        ###############################################################################
        # topic_base = self.topic_pub
        # pesan = input("Silakan ketikkan pesan Anda (pisahkan dengan koma): ")
        
        # Pisahkan pesan berdasarkan tanda koma
        # pesan_list = pesan.split(",")

        topic = f"{topic_pub}"  # Membuat sub-topik
        result = self.client.publish(topic=topic, payload=msg, qos=0)

        # for idx, sub_pesan in enumerate(pesan_list):
        #     topic = f"{topic_base}/{idx+1}"  # Membuat sub-topik
        #     result = self.client.publish(topic=topic, payload=sub_pesan.strip(), qos=0)
            
        if result.rc == 0:
            print(f"Pesan '{msg}' sukses dikirim ke topik '{topic}'")
        else:
            print(f"Pesan '{msg}' gagal dikirim ke topik '{topic}'. Error code: {result.rc}")
    
    # Fungsi untuk menerima pesan masuk melalui MQTT
    def subscribe_message(self):
        print(f"Menunggu pesan masuk di topik {self.topic_sub}... Tekan CTRL+C untuk keluar")
        try:
            while True:
                time.sleep(1)  # Loop untuk menjaga koneksi tetap terbuka
        except KeyboardInterrupt:
            print("Menghentikan subscription dan keluar.")
        finally:
            self.client.loop_stop()
            self.client.disconnect()

# Menjalankan kelas mqtt_handling
if __name__ == "__main__":
    mqtt_hand = Mqtt_Handling("broker.emqx.io", port=1883, publish=True, topic_pub="SMARTPARKINGLT03", msg=15)
