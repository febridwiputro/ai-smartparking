from mqtt_handling import Mqtt_Handling

if __name__ == "__main__":
    
    # mqtt_hand = Mqtt_Handling("broker.emqx.io", port=1883, subscribe= True, topik_subs="SMARTPARKING")
    mqtt_hand = Mqtt_Handling("broker.emqx.io", port=1883, publish= True, topic_pub="SMARTPARKING03")
            




