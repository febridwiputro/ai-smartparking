from src.Integration.api import APIModel
from src.Integration.arduino import Arduino
from src.controller.smart_parking_controller import SmartParking
from src.config.config import *

if __name__ == "__main__":
    video_sources = [VIDEO_SOURCE1, VIDEO_SOURCE2, VIDEO_SOURCE3, VIDEO_SOURCE4]
    # video_sources = [VIDEO_SOURCE1]
    parking_lots_list = [APIModel.bbox(config.LINK[i])[2] for i in range(len(video_sources))]
    slots = [APIModel.bbox(config.LINK[i])[0] for i in range(len(video_sources))]
    arduino_lt_2 = Arduino(baudrate=115200, serial_number=config.SERIAL_NUMBER_LT_2)
    smart_parking = SmartParking(video_sources=video_sources, parking_lots=parking_lots_list, slots_list=slots, links_cam=config.LINK, arduino=arduino_lt_2)
    smart_parking.run()



# from src.controller.smart_parking_controller import SmartParking
# from src.Integration.api import APIModel
# from src.Integration.arduino import Arduino
# import threading
#
# from src.config.config import config, VIDEO_SOURCE1, VIDEO_SOURCE2, VIDEO_SOURCE3, VIDEO_SOURCE4


# def run_smart_parking(app_instance):
#     app_instance.run()


# if __name__ == "__main__":
#     api = APIModel()
#     main_apps = []
#     # try :
#     ard = Arduino(baudrate=115200, area=config.AREA[0], driver="MEGA")
#     # except Exception as e:
#     #     print("Please check arduino cable !!")
#     #     exit()
#     video_sources = [VIDEO_SOURCE1, VIDEO_SOURCE2, VIDEO_SOURCE3, VIDEO_SOURCE4]
#
#     for i in range(len(video_sources)):
#         data = APIModel.bbox(config.LINK[i])
#         print("vid source" , video_sources[i])
#         main_apps.append(SmartParking(
#                 video_source=video_sources[i],
#                 model_path=config.MODEL_PATH,
#                 parking_lots=data[2],
#                 area=config.AREA[0],
#                 slot=data[0],
#                 arduino_object=ard
#         ))
#
#     threads = []
#
#     for app in main_apps:
#         thread = threading.Thread(target=run_smart_parking, args=(app,))
#         threads.append(thread)
#         thread.start()
#
#     threading.active_count()
#
#
#     for thread in threads:
#         thread.join()
#
#     print("Exiting ...")
#     exit()
