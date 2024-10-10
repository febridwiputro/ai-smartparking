import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    # FROM DOTENV
    BASE_URL = os.getenv("API_URL")
    # MANUAL CONFIG
    UPDATE_URL = "/slot/update"
    INSERT_BBOX_URL = "/camera/config"
    GET_BBOX_URL = "/camera/bbox/"
    GET_URL = "/slot/"
    GET_SLOT_URL = "/slot"

    BASE_DIR = r"C:\Users\DOT\Documents\febri"
    # BASE_DIR = r"D:\engine\smart_parking\repository\github"
    # BASE_DIR = "D:/engine/smart_parking/repository/github"
    # BASE_DIR = Path(_file_).parent.parent.resolve()
    
    VEHICLE_DETECTION_MODEL_PATH = os.path.join(BASE_DIR, "weights/multiple-vehicle.pt")
    MODEL_PATH = os.path.join(BASE_DIR, "weights/yolov8n.pt")
    MODEL_PATH_PLAT = os.path.join(BASE_DIR, "weights/license_plate_detector.pt")
    MODEL_PATH_PLAT_v2 = os.path.join(BASE_DIR, "weights/license_plat.pt")
    MODEL_PATH_PLAT_YOLOV8X = os.path.join(BASE_DIR, "weights/yolov8x-supervision-license-plate-recognition.pt")

    # CHARACTER RECOGNITION MODEL
    MODEL_CHAR_RECOGNITION_PATH = os.path.join(BASE_DIR, 'weights/ocr_model/new_model/20240925-11-01-14/character_recognition.json')
    WEIGHT_CHAR_RECOGNITION_PATH = os.path.join(BASE_DIR, 'weights/ocr_model/new_model/20240925-11-01-14/models_cnn.h5')
    LABEL_CHAR_RECOGNITION_PATH = os.path.join(BASE_DIR, 'weights/ocr_model/new_model/20240925-11-01-14/character_classes.npy')
    
    LINK_CAM_PREFIX = os.getenv("CAMERA_RTSP")

    # TODO write constants and configs as UPPER CASE
    #SmartParking-Configuration
    CAM_SOURCE_LT2_IN = fr'{LINK_CAM_PREFIX}192.168.1.10'
    CAM_SOURCE_LT2_OUT = fr'{LINK_CAM_PREFIX}192.168.1.11'
    CAM_SOURCE_LT3_IN = fr'{LINK_CAM_PREFIX}192.168.1.12'
    CAM_SOURCE_LT3_OUT = fr'{LINK_CAM_PREFIX}192.168.1.13'
    CAM_SOURCE_LT4_IN = fr'{LINK_CAM_PREFIX}192.168.1.14'
    CAM_SOURCE_LT4_OUT = fr'{LINK_CAM_PREFIX}192.168.1.15'
    CAM_SOURCE_LT5_IN = fr'{LINK_CAM_PREFIX}192.168.1.16'
    CAM_SOURCE_LT5_OUT = fr'{LINK_CAM_PREFIX}192.168.1.17'

    CAM_SOURCE_LT = [
        'rtsp://admin:Passw0rd@192.168.1.10',
        'rtsp://admin:Passw0rd@192.168.1.11', 
        'rtsp://admin:Passw0rd@192.168.1.12',
        'rtsp://admin:Passw0rd@192.168.1.13',
        'rtsp://admin:Passw0rd@192.168.1.14',
        'rtsp://admin:Passw0rd@192.168.1.15',
        'rtsp://admin:Passw0rd@192.168.1.16',
        'rtsp://admin:Passw0rd@192.168.1.17'
]

    # VIDEO_SOURCE_LT2_IN = fr'C:\Users\DOT\Documents\ai-smartparking\src\Assets\ocr_assets\z.mp4'
    # VIDEO_SOURCE_LT2_OUT = fr'C:\Users\DOT\Documents\ai-smartparking\src\Assets\ocr_assets\keluar_lt_2_out.mp4'
    # VIDEO_SOURCE_LT3_IN = fr'C:\Users\DOT\Documents\ai-smartparking\src\Assets\ocr_assets\z.mp4'
    # VIDEO_SOURCE_LT3_OUT = fr'C:\Users\DOT\Documents\ai-smartparking\src\Assets\ocr_assets\keluar_lt_2_out.mp4'
    # VIDEO_SOURCE_LT4_IN = fr'C:\Users\DOT\Documents\ai-smartparking\src\Assets\ocr_assets\z.mp4'
    # VIDEO_SOURCE_LT4_OUT = fr'C:\Users\DOT\Documents\ai-smartparking\src\Assets\ocr_assets\keluar_lt_2_out.mp4'
    # VIDEO_SOURCE_LT5_IN = fr'C:\Users\DOT\Documents\ai-smartparking\src\Assets\ocr_assets\z.mp4'
    # VIDEO_SOURCE_LT5_OUT = fr'C:\Users\DOT\Documents\ai-smartparking\src\Assets\ocr_assets\keluar_lt_2_out.mp4'

    # PC
    VIDEO_SOURCE_20241004_LT2_IN = fr"C:\Users\DOT\Documents\febri\video\sequence\LT_2_IN.mp4"
    VIDEO_SOURCE_20241004_LT2_OUT = fr"C:\Users\DOT\Documents\febri\video\sequence\LT_2_OUT.mp4"
    VIDEO_SOURCE_20241004_LT3_IN = fr"C:\Users\DOT\Documents\febri\video\sequence\LT_3_IN.mp4"
    VIDEO_SOURCE_20241004_LT3_OUT = fr"C:\Users\DOT\Documents\febri\video\sequence\LT_3_OUT.mp4"
    VIDEO_SOURCE_20241004_LT4_IN = fr"C:\Users\DOT\Documents\febri\video\sequence\LT_4_IN.mp4"
    VIDEO_SOURCE_20241004_LT4_OUT = fr"C:\Users\DOT\Documents\febri\video\sequence\LT_4_OUT.mp4"
    VIDEO_SOURCE_20241004_LT5_IN = fr"C:\Users\DOT\Documents\febri\video\sequence\LT_5_IN.mp4"
    VIDEO_SOURCE_20241004_LT5_OUT = fr"C:\Users\DOT\Documents\febri\video\sequence\LT_5_OUT.mp4"

    # LAPTOP
    # BASE_VIDEO_SOURCE_20241004 = r"D:\engine\smart_parking\dataset\cctv\20241004\split_videos"

    # VIDEO_SOURCE_20241004_LT2_IN = fr"{BASE_VIDEO_SOURCE_20241004}\LT_2_IN_clip.mp4"
    # VIDEO_SOURCE_20241004_LT2_OUT = fr"{BASE_VIDEO_SOURCE_20241004}\LT_2_OUT_clip.mp4"
    # VIDEO_SOURCE_20241004_LT3_IN = fr"{BASE_VIDEO_SOURCE_20241004}\LT_3_IN_clip.mp4"
    # VIDEO_SOURCE_20241004_LT3_OUT = fr"{BASE_VIDEO_SOURCE_20241004}\LT_3_OUT_clip.mp4"
    # VIDEO_SOURCE_20241004_LT4_IN = fr"{BASE_VIDEO_SOURCE_20241004}\LT_4_IN_clip.mp4"
    # VIDEO_SOURCE_20241004_LT4_OUT = fr"{BASE_VIDEO_SOURCE_20241004}\LT_4_OUT_clip.mp4"
    # VIDEO_SOURCE_20241004_LT5_IN = fr"{BASE_VIDEO_SOURCE_20241004}\LT_5_IN_clip.mp4"
    # VIDEO_SOURCE_20241004_LT5_OUT = fr"{BASE_VIDEO_SOURCE_20241004}\LT_5_OUT_clip.mp4"


    # VIDEO_SOURCE_20241004_LT2_IN = fr"C:\Users\DOT\Documents\febri\video\output\car\LT_2_IN.mp4"
    # VIDEO_SOURCE_20241004_LT2_OUT = fr"C:\Users\DOT\Documents\febri\video\output\car\LT_2_OUT.mp4"
    # VIDEO_SOURCE_20241004_LT3_IN = fr"C:\Users\DOT\Documents\febri\video\output\car\LT_3_IN.mp4"
    # VIDEO_SOURCE_20241004_LT3_OUT = fr"C:\Users\DOT\Documents\febri\video\output\car\LT_3_OUT.mp4"
    # VIDEO_SOURCE_20241004_LT4_IN = fr"C:\Users\DOT\Documents\febri\video\output\car\LT_4_IN.mp4"
    # VIDEO_SOURCE_20241004_LT4_OUT = fr"C:\Users\DOT\Documents\febri\video\output\car\LT_4_OUT.mp4"
    # VIDEO_SOURCE_20241004_LT5_IN = fr"C:\Users\DOT\Documents\febri\video\output\car\LT_5_IN.mp4"
    # VIDEO_SOURCE_20241004_LT5_OUT = fr"C:\Users\DOT\Documents\febri\video\output\car\LT_5_OUT.mp4"

    VIDEO_SOURCE_LT2_IN = fr'D:\engine\smart_parking\dataset\cctv\z.mp4'
    # VIDEO_SOURCE_LT2_OUT = fr'D:\engine\smart_parking\dataset\cctv\keluar_lt_2_out.mp4'
    VIDEO_SOURCE_LT2_OUT = fr'D:\engine\cv\dataset_editor\editor\compose_video.mp4'
    VIDEO_SOURCE_LT3_IN = fr'D:\engine\smart_parking\dataset\cctv\z.mp4'
    VIDEO_SOURCE_LT3_OUT = fr'D:\engine\smart_parking\dataset\cctv\keluar_lt_2_out.mp4'
    VIDEO_SOURCE_LT4_IN = fr'D:\engine\smart_parking\dataset\cctv\z.mp4'
    VIDEO_SOURCE_LT4_OUT = fr'D:\engine\smart_parking\dataset\cctv\keluar_lt_2_out.mp4'
    VIDEO_SOURCE_LT5_IN = fr'D:\engine\smart_parking\dataset\cctv\z.mp4'
    VIDEO_SOURCE_LT5_OUT = fr'D:\engine\smart_parking\dataset\cctv\keluar_lt_2_out.mp4'

    VIDEO_SOURCE_20241004 = [VIDEO_SOURCE_20241004_LT2_IN, 
                    VIDEO_SOURCE_20241004_LT2_OUT, 
                    VIDEO_SOURCE_20241004_LT3_IN, 
                    VIDEO_SOURCE_20241004_LT3_OUT, 
                    VIDEO_SOURCE_20241004_LT4_IN, 
                    VIDEO_SOURCE_20241004_LT4_OUT, 
                    VIDEO_SOURCE_20241004_LT5_IN, 
                    VIDEO_SOURCE_20241004_LT5_OUT
                    ]
    
    VIDEO_SOURCE_PC = [
                        fr'C:\Users\DOT\Documents\ai-smartparking\src\Assets\ocr_assets\z.mp4',
                        fr'C:\Users\DOT\Documents\febri\github\combined_video_out.mp4'
        # fr'C:\Users\DOT\Documents\febri\video\LT_5_IN.mp4',
        # fr'C:\Users\DOT\Documents\febri\video\LT_5_OUT.mp4',
    #     fr'C:\Users\DOT\Documents\febri\video\LT_5_IN.mp4',
    #     fr'C:\Users\DOT\Documents\febri\video\LT_5_OUT.mp4',
    #    fr'C:\Users\DOT\Documents\ai-smartparking\src\Assets\ocr_assets\z.mp4',
    #     fr'C:\Users\DOT\Documents\febri\video\LT_5_OUT.mp4',
    #     # fr'C:\Users\DOT\Documents\febri\github\combined_video_out.mp4',
    #      fr'C:\Users\DOT\Documents\febri\video\LT_5_IN.mp4',
    #     fr'C:\Users\DOT\Documents\febri\video\LT_5_OUT.mp4',

    ]

    VIDEO_SOURCE_LAPTOP = [VIDEO_SOURCE_LT2_IN, 
                    VIDEO_SOURCE_LT2_OUT, 
                    # VIDEO_SOURCE_LT3_IN, 
                    # VIDEO_SOURCE_LT3_OUT, 
                    # VIDEO_SOURCE_LT4_IN, 
                    # VIDEO_SOURCE_LT4_OUT, 
                    # VIDEO_SOURCE_LT5_IN, 
                    # VIDEO_SOURCE_LT5_OUT
                    ]

    # video_source1 = [VIDEO_SOURCE_LT2_IN]

    # VIDEO_SOURCE_LT2 = [VIDEO_SOURCE_LT2_IN, 
    #                 VIDEO_SOURCE_LT2_OUT]

    # VIDEO_SOURCE_LT23 = [VIDEO_SOURCE_LT2_IN, 
    #                 VIDEO_SOURCE_LT2_OUT, 
    #                 VIDEO_SOURCE_LT3_IN, 
    #                 VIDEO_SOURCE_LT3_OUT]

    CAM_SOURCE_LT2 = [CAM_SOURCE_LT2_IN, 
                  CAM_SOURCE_LT2_OUT]

    CAM_SOURCE = [CAM_SOURCE_LT2_IN, 
                  CAM_SOURCE_LT2_OUT, 
                  CAM_SOURCE_LT3_IN, 
                  CAM_SOURCE_LT3_OUT, 
                  CAM_SOURCE_LT4_IN, 
                  CAM_SOURCE_LT4_OUT, 
                  CAM_SOURCE_LT5_IN, 
                  CAM_SOURCE_LT5_OUT]
    
    LINK = [f"192.168.1.10{i}" for i in range(1, 5)]
    
    DRIVER_MATRIX = "CP210"
    DRIVER_MATRIX_NUM = "CP210"
    SERIAL_NUMBER_LT_2 = 'D200RBECA'
    LT_2_IN = "rtsp://admin:Passw0rd@192.168.1.10"

    SERIAL_NUMBER_MATRIX_NUM_LT2 = '5626004960'
    SERIAL_NUMBER_MATRIX_TEXT_LT2 = '0001'

    SERIAL_NUMBER_MATRIX_NUM_LT3 = '5626004961'
    SERIAL_NUMBER_MATRIX_TEXT_LT3 = '0001'

    SERIAL_NUMBER_MATRIX_NUM_LT4 = '5626004962'
    SERIAL_NUMBER_MATRIX_TEXT_LT4 = '0002'

    SERIAL_NUMBER_MATRIX_NUM_LT5 = '5626004963'
    SERIAL_NUMBER_MATRIX_TEXT_LT5 = '0003'

    SERIAL_LT2 = [
        SERIAL_NUMBER_MATRIX_TEXT_LT2, SERIAL_NUMBER_MATRIX_NUM_LT2
    ]

    SERIALS = [
        SERIAL_NUMBER_MATRIX_TEXT_LT2, SERIAL_NUMBER_MATRIX_NUM_LT2,
        SERIAL_NUMBER_MATRIX_TEXT_LT3, SERIAL_NUMBER_MATRIX_NUM_LT3,
        SERIAL_NUMBER_MATRIX_TEXT_LT4, SERIAL_NUMBER_MATRIX_NUM_LT4,
        SERIAL_NUMBER_MATRIX_TEXT_LT5, SERIAL_NUMBER_MATRIX_NUM_LT5
    ]

    # CLASS_NAMES = {
    #     # 0: 'person'
    #     2: 'car',
    #     7: 'truck'
    # }
    
    CLASS_NAMES = [2]
    # CLASS_NAMES = [0, 2]
    CLASS_PLAT_NAMES = [8, 11, 12, 13]
    
    # CLASS_PLAT_NAMES = {
    #     8: 'plat',
    #     11: "plat_indo",
    #     12: "plat_mobil",
    #     13: "plat_motor"
    # }
    
    AREA = ["lantai_2"]

    POINTS_BACKGROUND_LT2_IN = [(0.31574074074074077, 0.07222222222222222),
                         (0.012962962962962963, 0.41388888888888886),
                         (0.003703703703703704, 0.9972222222222222),
                         (0.9990740740740741, 0.9958333333333333),
                         (0.9953703703703703, 0.6041666666666666),
                         (0.7861111111111111, 0.29583333333333334),
                         (0.6620370370370371, 0.23472222222222222),
                         (0.6648148148148149, 0.03194444444444444),
                         (0.6648148148148149, 0.002777777777777778),
                         (0.2953703703703704, 0.005555555555555556),
                         (0.31296296296296294, 0.075)
    ]
    
    POINTS_BACKGROUND_LT2_OUT = [
        (0.004629629629629629, 0.9875),
        (0.006481481481481481, 0.5930555555555556),
        (0.28055555555555556, 0.3458333333333333),
        (0.34629629629629627, 0.30277777777777776),
        (0.3768518518518518, 0.14305555555555555),
        (0.5935185185185186, 0.15694444444444444),
        (0.975, 0.3638888888888889),
        (0.9166666666666666, 0.7013888888888888),
        (0.9898148148148148, 0.9861111111111112),
        (0.005555555555555556, 0.9888888888888889),
    
    ]

    POINTS_BACKGROUND_LT3_IN = [(0.31574074074074077, 0.07222222222222222),
                         (0.012962962962962963, 0.41388888888888886),
                         (0.003703703703703704, 0.9972222222222222),
                         (0.9990740740740741, 0.9958333333333333),
                         (0.9953703703703703, 0.6041666666666666),
                         (0.7861111111111111, 0.29583333333333334),
                         (0.6620370370370371, 0.23472222222222222),
                         (0.6648148148148149, 0.03194444444444444),
                         (0.6648148148148149, 0.002777777777777778),
                         (0.2953703703703704, 0.005555555555555556),
                         (0.31296296296296294, 0.075)
    ]
    
    POINTS_BACKGROUND_LT3_OUT = [
        (0.004629629629629629, 0.9875),
        (0.006481481481481481, 0.5930555555555556),
        (0.28055555555555556, 0.3458333333333333),
        (0.34629629629629627, 0.30277777777777776),
        (0.3768518518518518, 0.14305555555555555),
        (0.5935185185185186, 0.15694444444444444),
        (0.975, 0.3638888888888889),
        (0.9166666666666666, 0.7013888888888888),
        (0.9898148148148148, 0.9861111111111112),
        (0.005555555555555556, 0.9888888888888889),
    
    ]

    POINTS_BACKGROUND_LT4_IN = [(0.31574074074074077, 0.07222222222222222),
                         (0.012962962962962963, 0.41388888888888886),
                         (0.003703703703703704, 0.9972222222222222),
                         (0.9990740740740741, 0.9958333333333333),
                         (0.9953703703703703, 0.6041666666666666),
                         (0.7861111111111111, 0.29583333333333334),
                         (0.6620370370370371, 0.23472222222222222),
                         (0.6648148148148149, 0.03194444444444444),
                         (0.6648148148148149, 0.002777777777777778),
                         (0.2953703703703704, 0.005555555555555556),
                         (0.31296296296296294, 0.075)
    ]
    
    POINTS_BACKGROUND_LT4_OUT = [
        (0.004629629629629629, 0.9875),
        (0.006481481481481481, 0.5930555555555556),
        (0.28055555555555556, 0.3458333333333333),
        (0.34629629629629627, 0.30277777777777776),
        (0.3768518518518518, 0.14305555555555555),
        (0.5935185185185186, 0.15694444444444444),
        (0.975, 0.3638888888888889),
        (0.9166666666666666, 0.7013888888888888),
        (0.9898148148148148, 0.9861111111111112),
        (0.005555555555555556, 0.9888888888888889),
    
    ]

    POINTS_BACKGROUND_LT5_IN = [(0.31574074074074077, 0.07222222222222222),
                         (0.012962962962962963, 0.41388888888888886),
                         (0.003703703703703704, 0.9972222222222222),
                         (0.9990740740740741, 0.9958333333333333),
                         (0.9953703703703703, 0.6041666666666666),
                         (0.7861111111111111, 0.29583333333333334),
                         (0.6620370370370371, 0.23472222222222222),
                         (0.6648148148148149, 0.03194444444444444),
                         (0.6648148148148149, 0.002777777777777778),
                         (0.2953703703703704, 0.005555555555555556),
                         (0.31296296296296294, 0.075)
    ]
    
    POINTS_BACKGROUND_LT5_OUT = [
        (0.004629629629629629, 0.9875),
        (0.006481481481481481, 0.5930555555555556),
        (0.28055555555555556, 0.3458333333333333),
        (0.34629629629629627, 0.30277777777777776),
        (0.3768518518518518, 0.14305555555555555),
        (0.5935185185185186, 0.15694444444444444),
        (0.975, 0.3638888888888889),
        (0.9166666666666666, 0.7013888888888888),
        (0.9898148148148148, 0.9861111111111112),
        (0.005555555555555556, 0.9888888888888889),
    
    ]

  
    # LT2
    POINT_LT2_IN_L_START = (0.34629629629629627, 0.3277777777777778)
    POINT_LT2_IN_R_START = (0.5972222222222222, 0.32222222222222224)
    POINT_LT2_IN_L_END = (0.05740740740740741, 0.6611111111111111)
    POINT_LT2_IN_R_END = (0.6907407407407408, 0.6694444444444444)

    POINT_LT2_OUT_L_START = (0.34629629629629627, 0.3277777777777778)
    # POINT_LT2_OUT_L_START = (0.35833333333333334, 0.39166666666666666)
    POINT_LT2_OUT_R_START = (0.5972222222222222, 0.32222222222222224)
    # POINT_LT2_OUT_R_START = (0.5787037037037037, 0.4)
    POINT_LT2_OUT_L_END = (0.05462962962962963, 0.6611111111111111)
    POINT_LT2_OUT_R_END = (0.8111111111111111, 0.6708333333333333)

    # LT3
    POINT_LT3_IN_L_START = (0.34629629629629627, 0.3277777777777778)
    POINT_LT3_IN_R_START = (0.5972222222222222, 0.32222222222222224)
    POINT_LT3_IN_L_END = (0.05740740740740741, 0.6611111111111111)
    POINT_LT3_IN_R_END = (0.6907407407407408, 0.6694444444444444)

    POINT_LT3_OUT_L_START = (0.34629629629629627, 0.3277777777777778)
    # POINT_LT2_OUT_L_START = (0.35833333333333334, 0.39166666666666666)
    POINT_LT3_OUT_R_START = (0.5972222222222222, 0.32222222222222224)
    # POINT_LT2_OUT_R_START = (0.5787037037037037, 0.4)
    POINT_LT3_OUT_L_END = (0.05462962962962963, 0.6611111111111111)
    POINT_LT3_OUT_R_END = (0.8111111111111111, 0.6708333333333333)

    # LT4
    POINT_LT4_IN_L_START = (0.34629629629629627, 0.3277777777777778)
    POINT_LT4_IN_R_START = (0.5972222222222222, 0.32222222222222224)
    POINT_LT4_IN_L_END = (0.05740740740740741, 0.6611111111111111)
    POINT_LT4_IN_R_END = (0.6907407407407408, 0.6694444444444444)

    POINT_LT4_OUT_L_START = (0.34629629629629627, 0.3277777777777778)
    # POINT_LT2_OUT_L_START = (0.35833333333333334, 0.39166666666666666)
    POINT_LT4_OUT_R_START = (0.5972222222222222, 0.32222222222222224)
    # POINT_LT2_OUT_R_START = (0.5787037037037037, 0.4)
    POINT_LT4_OUT_L_END = (0.05462962962962963, 0.6611111111111111)
    POINT_LT4_OUT_R_END = (0.8111111111111111, 0.6708333333333333)

    # LT5
    POINT_LT5_IN_L_START = (0.34629629629629627, 0.3277777777777778)
    POINT_LT5_IN_R_START = (0.5972222222222222, 0.32222222222222224)
    POINT_LT5_IN_L_END = (0.05740740740740741, 0.6611111111111111)
    POINT_LT5_IN_R_END = (0.6907407407407408, 0.6694444444444444)

    POINT_LT5_OUT_L_START = (0.34629629629629627, 0.3277777777777778)
    # POINT_LT2_OUT_L_START = (0.35833333333333334, 0.39166666666666666)
    POINT_LT5_OUT_R_START = (0.5972222222222222, 0.32222222222222224)
    # POINT_LT2_OUT_R_START = (0.5787037037037037, 0.4)
    POINT_LT5_OUT_L_END = (0.05462962962962963, 0.6611111111111111)
    POINT_LT5_OUT_R_END = (0.8111111111111111, 0.6708333333333333)


    POLYGON_POINT_LT2_IN = [
        POINT_LT2_IN_L_START,
        POINT_LT2_IN_R_START,
        POINT_LT2_IN_L_END,
        POINT_LT2_IN_R_END        
    ]

    POLYGON_POINT_LT2_OUT = [
        POINT_LT2_OUT_L_START,
        POINT_LT2_OUT_R_START,
        POINT_LT2_OUT_L_END,
        POINT_LT2_OUT_R_END        
    ]

    POLYGON_POINT_LT3_IN = [
        POINT_LT3_IN_L_START,
        POINT_LT3_IN_R_START,
        POINT_LT3_IN_L_END,
        POINT_LT3_IN_R_END        
    ]    

    POLYGON_POINT_LT3_OUT = [
        POINT_LT3_OUT_L_START,
        POINT_LT3_OUT_R_START,
        POINT_LT3_OUT_L_END,
        POINT_LT3_OUT_R_END
    ]

    POLYGON_POINT_LT4_IN = [
        POINT_LT4_IN_L_START,
        POINT_LT4_IN_R_START,
        POINT_LT4_IN_L_END,
        POINT_LT4_IN_R_END        
    ]    

    POLYGON_POINT_LT4_OUT = [
        POINT_LT4_OUT_L_START,
        POINT_LT4_OUT_R_START,
        POINT_LT4_OUT_L_END,
        POINT_LT4_OUT_R_END
    ]

    POLYGON_POINT_LT5_IN = [
        POINT_LT5_IN_L_START,
        POINT_LT5_IN_R_START,
        POINT_LT5_IN_L_END,
        POINT_LT5_IN_R_END        
    ]    

    POLYGON_POINT_LT5_OUT = [
        POINT_LT5_OUT_L_START,
        POINT_LT5_OUT_R_START,
        POINT_LT5_OUT_L_END,
        POINT_LT5_OUT_R_END
    ]

config = Config()



import random
BASE_DIR = Path(__file__).parent.parent.resolve()


MODEL_PATH = BASE_DIR / "weights/yolov8n.pt"

#MQTT
BROKER = '192.168.88.60'
PORT = 1883
TOPIC = "DOT/UAT/LAMP"
CLIENT_ID = f'publish-{random.randint(0, 1000)}'


# PARKING_LOTS = [
# [(888, 879), (1180, 771), (1398, 917), (1072, 1048), (889, 883)],
# [(450, 994), (887, 878), (1072, 1048), (471, 1048), (450, 993)],
# [(1182, 768), (1384, 697), (1586, 806), (1398, 914), (1182, 768)],
# [(1626, 782), (1911, 899), (1911, 1048), (1508, 1048), (1434, 915), (1627, 784)],
# # [(1590, 805), (1893, 918), (1749, 1048), (1399, 915)],
# [(396, 705), (438, 916), (736, 846), (657, 647), (398, 703)],
# [(656, 647), (737, 847), (964, 781), (880, 598), (659, 647)],
# [(882, 597), (962, 778), (1214, 701), (1098, 548), (882, 594)]
# # [(1479, 521), (1329, 597), (1512, 674), (1631, 592), (1481, 521)],
# # [(1632, 588), (1847, 668), (1704, 755), (1514, 673), (1632, 592)],
#     #data 6
# # [(1063, 1048), (932, 885), (1193, 772), (1340, 900), (1064, 1047)],
# # [(412, 758), (650, 700), (715, 901), (452, 966), (414, 758)],
# # [(650, 701), (867, 651), (942, 813), (718, 905), (653, 700)],
# # [(867, 653), (1078, 599), (1227, 699), (943, 810), (870, 655)],
# # [(1610, 602), (1481, 691), (1694, 795), (1795, 667), (1608, 599)],
# # [(479, 1048), (932, 892), (1062, 1046), (479, 1048)],
# ]
#
# PARKING_LOTS2 = [
# [(313, 749), (791, 798), (830, 604), (499, 578), (313, 746)],
# [(500, 577), (588, 490), (852, 506), (829, 606), (501, 578)],
# [(851, 503), (1094, 514), (1090, 625), (831, 608), (852, 502)],
# [(827, 610), (1089, 628), (1072, 798), (799, 797), (823, 613)],
# [(1617, 745), (1298, 825), (1208, 683), (1450, 632), (1617, 744)],
# [(1326, 542), (1156, 594), (1208, 682), (1451, 631), (1334, 540)],
# [(1270, 493), (1123, 504), (1156, 592), (1334, 540), (1272, 495)],
# [(1378, 568), (1560, 486), (1685, 531), (1457, 634), (1376, 568)],
# [(1684, 530), (1894, 621), (1620, 747), (1456, 632), (1684, 534)],
# ]

CLASS_NAMES = {
    0: 'person',
    # 1: 'bicycle',
    2: 'car',
    # 3: 'motorcycle',
    # 5: 'bus',
    7: 'truck'
    # Add more classes if needed
}

###############################################################

# import os
# from pathlib import Path
# from dotenv import load_dotenv

# load_dotenv()


# class Config:
#     # FROM DOTENV
#     BASE_URL = os.getenv("API_URL")
#     # MANUAL CONFIG
#     UPDATE_URL = "/slot/update"
#     INSERT_BBOX_URL = "/camera/config"
#     GET_BBOX_URL = "/camera/bbox/"
#     GET_URL = "/slot/"
#     GET_SLOT_URL = "/slot"
    
#     BASE_DIR = r"C:\Users\DOT\Documents\febri"
#     # BASE_DIR = "D:/engine/smart_parking/repository/github"
#     # BASE_DIR = Path(_file_).parent.parent.resolve()
    
#     MODEL_PATH = os.path.join(BASE_DIR, "weights/yolov8n.pt")
#     MODEL_PATH_PLAT = os.path.join(BASE_DIR, "weights/license_plate_detector.pt")
#     MODEL_PATH_PLAT_v2 = os.path.join(BASE_DIR, "weights/license_plat.pt")

#     # CHARACTER RECOGNITION MODEL
#     MODEL_CHAR_RECOGNITION_PATH = os.path.join(BASE_DIR, 'weights/ocr_model/new_model/20240925-11-01-14/character_recognition.json')
#     WEIGHT_CHAR_RECOGNITION_PATH = os.path.join(BASE_DIR, 'weights/ocr_model/new_model/20240925-11-01-14/models_cnn.h5')
#     LABEL_CHAR_RECOGNITION_PATH = os.path.join(BASE_DIR, 'weights/ocr_model/new_model/20240925-11-01-14/character_classes.npy')
    
#     LINK_CAM_PREFIX = os.getenv("CAMERA_RTSP")

#     # TODO write constants and configs as UPPER CASE
#     #SmartParking-Configuration
#     CAM_SOURCE_LT2_IN = fr'{LINK_CAM_PREFIX}192.168.1.10'
#     CAM_SOURCE_LT2_OUT = fr'{LINK_CAM_PREFIX}192.168.1.11'
#     CAM_SOURCE_LT3_IN = fr'{LINK_CAM_PREFIX}192.168.1.12'
#     CAM_SOURCE_LT3_OUT = fr'{LINK_CAM_PREFIX}192.168.1.13'
#     CAM_SOURCE_LT4_IN = fr'{LINK_CAM_PREFIX}192.168.1.14'
#     CAM_SOURCE_LT4_OUT = fr'{LINK_CAM_PREFIX}192.168.1.15'
#     CAM_SOURCE_LT5_IN = fr'{LINK_CAM_PREFIX}192.168.1.16'
#     CAM_SOURCE_LT5_OUT = fr'{LINK_CAM_PREFIX}192.168.1.17'

#     VIDEO_SOURCE_LT2_IN = fr'C:\Users\DOT\Documents\ai-smartparking\src\Assets\ocr_assets\z.mp4'
#     VIDEO_SOURCE_LT2_OUT = fr'C:\Users\DOT\Documents\ai-smartparking\src\Assets\ocr_assets\keluar_lt_2_out.mp4'
#     VIDEO_SOURCE_LT3_IN = fr'C:\Users\DOT\Documents\ai-smartparking\src\Assets\ocr_assets\z.mp4'
#     VIDEO_SOURCE_LT3_OUT = fr'C:\Users\DOT\Documents\ai-smartparking\src\Assets\ocr_assets\keluar_lt_2_out.mp4'
#     VIDEO_SOURCE_LT4_IN = fr'C:\Users\DOT\Documents\ai-smartparking\src\Assets\ocr_assets\z.mp4'
#     VIDEO_SOURCE_LT4_OUT = fr'C:\Users\DOT\Documents\ai-smartparking\src\Assets\ocr_assets\keluar_lt_2_out.mp4'
#     VIDEO_SOURCE_LT5_IN = fr'C:\Users\DOT\Documents\ai-smartparking\src\Assets\ocr_assets\z.mp4'
#     VIDEO_SOURCE_LT5_OUT = fr'C:\Users\DOT\Documents\ai-smartparking\src\Assets\ocr_assets\keluar_lt_2_out.mp4'

#     VIDEO_SOURCE_20241004_LT2_IN = fr"C:\Users\DOT\Documents\febri\video\sequence\LT_2_IN.mp4"
#     VIDEO_SOURCE_20241004_LT2_OUT = fr"C:\Users\DOT\Documents\febri\video\sequence\LT_2_OUT.mp4"
#     VIDEO_SOURCE_20241004_LT3_IN = fr"C:\Users\DOT\Documents\febri\video\sequence\LT_3_IN.mp4"
#     VIDEO_SOURCE_20241004_LT3_OUT = fr"C:\Users\DOT\Documents\febri\video\sequence\LT_3_OUT.mp4"
#     VIDEO_SOURCE_20241004_LT4_IN = fr"C:\Users\DOT\Documents\febri\video\sequence\LT_4_IN.mp4"
#     VIDEO_SOURCE_20241004_LT4_OUT = fr"C:\Users\DOT\Documents\febri\video\sequence\LT_4_OUT.mp4"
#     VIDEO_SOURCE_20241004_LT5_IN = fr"C:\Users\DOT\Documents\febri\video\sequence\LT_5_IN.mp4"
#     VIDEO_SOURCE_20241004_LT5_OUT = fr"C:\Users\DOT\Documents\febri\video\sequence\LT_5_OUT.mp4"

#     # VIDEO_SOURCE_20241004_LT2_IN = fr"C:\Users\DOT\Documents\febri\video\output\car\LT_2_IN.mp4"
#     # VIDEO_SOURCE_20241004_LT2_OUT = fr"C:\Users\DOT\Documents\febri\video\output\car\LT_2_OUT.mp4"
#     # VIDEO_SOURCE_20241004_LT3_IN = fr"C:\Users\DOT\Documents\febri\video\output\car\LT_3_IN.mp4"
#     # VIDEO_SOURCE_20241004_LT3_OUT = fr"C:\Users\DOT\Documents\febri\video\output\car\LT_3_OUT.mp4"
#     # VIDEO_SOURCE_20241004_LT4_IN = fr"C:\Users\DOT\Documents\febri\video\output\car\LT_4_IN.mp4"
#     # VIDEO_SOURCE_20241004_LT4_OUT = fr"C:\Users\DOT\Documents\febri\video\output\car\LT_4_OUT.mp4"
#     # VIDEO_SOURCE_20241004_LT5_IN = fr"C:\Users\DOT\Documents\febri\video\output\car\LT_5_IN.mp4"
#     # VIDEO_SOURCE_20241004_LT5_OUT = fr"C:\Users\DOT\Documents\febri\video\output\car\LT_5_OUT.mp4"

#     # VIDEO_SOURCE_LT2_IN = fr'D:\engine\smart_parking\dataset\cctv\z.mp4'
#     # # VIDEO_SOURCE_LT2_OUT = fr'D:\engine\smart_parking\dataset\cctv\keluar_lt_2_out.mp4'
#     # VIDEO_SOURCE_LT2_OUT = fr'D:\engine\cv\dataset_editor\editor\compose_video.mp4'
#     # VIDEO_SOURCE_LT3_IN = fr'D:\engine\smart_parking\dataset\cctv\z.mp4'
#     # VIDEO_SOURCE_LT3_OUT = fr'D:\engine\smart_parking\dataset\cctv\keluar_lt_2_out.mp4'
#     # VIDEO_SOURCE_LT4_IN = fr'D:\engine\smart_parking\dataset\cctv\z.mp4'
#     # VIDEO_SOURCE_LT4_OUT = fr'D:\engine\smart_parking\dataset\cctv\keluar_lt_2_out.mp4'
#     # VIDEO_SOURCE_LT5_IN = fr'D:\engine\smart_parking\dataset\cctv\z.mp4'
#     # VIDEO_SOURCE_LT5_OUT = fr'D:\engine\smart_parking\dataset\cctv\keluar_lt_2_out.mp4'

#     VIDEO_SOURCE_20241004 = [VIDEO_SOURCE_20241004_LT2_IN, 
#                     VIDEO_SOURCE_20241004_LT2_OUT, 
#                     VIDEO_SOURCE_20241004_LT3_IN, 
#                     VIDEO_SOURCE_20241004_LT3_OUT, 
#                     VIDEO_SOURCE_20241004_LT4_IN, 
#                     VIDEO_SOURCE_20241004_LT4_OUT, 
#                     VIDEO_SOURCE_20241004_LT5_IN, 
#                     VIDEO_SOURCE_20241004_LT5_OUT]

#     VIDEO_SOURCE = [VIDEO_SOURCE_LT2_IN, 
#                     VIDEO_SOURCE_LT2_OUT, 
#                     VIDEO_SOURCE_LT3_IN, 
#                     VIDEO_SOURCE_LT3_OUT, 
#                     VIDEO_SOURCE_LT4_IN, 
#                     VIDEO_SOURCE_LT4_OUT, 
#                     VIDEO_SOURCE_LT5_IN, 
#                     VIDEO_SOURCE_LT5_OUT]

#     video_source1 = [VIDEO_SOURCE_LT2_IN]

#     VIDEO_SOURCE_LT2 = [VIDEO_SOURCE_LT2_IN, 
#                     VIDEO_SOURCE_LT2_OUT]

#     VIDEO_SOURCE_LT23 = [VIDEO_SOURCE_LT2_IN, 
#                     VIDEO_SOURCE_LT2_OUT, 
#                     VIDEO_SOURCE_LT3_IN, 
#                     VIDEO_SOURCE_LT3_OUT]

#     CAM_SOURCE_LT2 = [CAM_SOURCE_LT2_IN, 
#                   CAM_SOURCE_LT2_OUT]

#     CAM_SOURCE_LT = [CAM_SOURCE_LT2_IN, 
#                   CAM_SOURCE_LT2_OUT, 
#                   CAM_SOURCE_LT3_IN, 
#                   CAM_SOURCE_LT3_OUT, 
#                   CAM_SOURCE_LT4_IN, 
#                   CAM_SOURCE_LT4_OUT, 
#                   CAM_SOURCE_LT5_IN, 
#                   CAM_SOURCE_LT5_OUT]
    
#     LINK = [f"192.168.1.10{i}" for i in range(1, 5)]
    
#     DRIVER_MATRIX = "CP210"
#     DRIVER_MATRIX_NUM = "CP210"
#     SERIAL_NUMBER_LT_2 = 'D200RBECA'
#     LT_2_IN = "rtsp://admin:Passw0rd@192.168.1.10"

#     SERIAL_NUMBER_MATRIX_NUM_LT2 = '5626004960'
#     SERIAL_NUMBER_MATRIX_TEXT_LT2 = '0001'

#     SERIAL_NUMBER_MATRIX_NUM_LT3 = '5626004961'
#     SERIAL_NUMBER_MATRIX_TEXT_LT3 = '0001'

#     SERIAL_NUMBER_MATRIX_NUM_LT4 = '5626004962'
#     SERIAL_NUMBER_MATRIX_TEXT_LT4 = '0002'

#     SERIAL_NUMBER_MATRIX_NUM_LT5 = '5626004963'
#     SERIAL_NUMBER_MATRIX_TEXT_LT5 = '0003'

#     SERIAL_LT2 = [
#         SERIAL_NUMBER_MATRIX_TEXT_LT2, SERIAL_NUMBER_MATRIX_NUM_LT2
#     ]

#     SERIALS = [
#         SERIAL_NUMBER_MATRIX_TEXT_LT2, SERIAL_NUMBER_MATRIX_NUM_LT2,
#         SERIAL_NUMBER_MATRIX_TEXT_LT3, SERIAL_NUMBER_MATRIX_NUM_LT3,
#         SERIAL_NUMBER_MATRIX_TEXT_LT4, SERIAL_NUMBER_MATRIX_NUM_LT4,
#         SERIAL_NUMBER_MATRIX_TEXT_LT5, SERIAL_NUMBER_MATRIX_NUM_LT5
#     ]

#     # CLASS_NAMES = {
#     #     # 0: 'person'
#     #     2: 'car',
#     #     7: 'truck'
#     # }
    
#     CLASS_NAMES = [2, 7, 5]
#     CLASS_PLAT_NAMES = [8, 11, 12, 13]
    
#     # CLASS_PLAT_NAMES = {
#     #     8: 'plat',
#     #     11: "plat_indo",
#     #     12: "plat_mobil",
#     #     13: "plat_motor"
#     # }
    
#     AREA = ["lantai_2"]

#     POINTS_BACKGROUND_LT2_IN = [(0.31574074074074077, 0.07222222222222222),
#                                 (0.012962962962962963, 0.41388888888888886),
#                                 (0.003703703703703704, 0.9972222222222222),
#                                 (0.9990740740740741, 0.9958333333333333),
#                                 (0.9953703703703703, 0.6041666666666666),
#                                 (0.7861111111111111, 0.29583333333333334),
#                                 (0.6620370370370371, 0.23472222222222222),
#                                 (0.6648148148148149, 0.03194444444444444),
#                                 (0.6648148148148149, 0.002777777777777778),
#                                 (0.2953703703703704, 0.005555555555555556),
#                                 (0.31296296296296294, 0.075)]
    
#     POINTS_BACKGROUND_LT2_OUT = [(0.004629629629629629, 0.9875),
#                                 (0.006481481481481481, 0.5930555555555556),
#                                 (0.28055555555555556, 0.3458333333333333),
#                                 (0.34629629629629627, 0.30277777777777776),
#                                 (0.3768518518518518, 0.14305555555555555),
#                                 (0.5935185185185186, 0.15694444444444444),
#                                 (0.975, 0.3638888888888889),
#                                 (0.9166666666666666, 0.7013888888888888),
#                                 (0.9898148148148148, 0.9861111111111112),
#                                 (0.005555555555555556, 0.9888888888888889)]

#     POINTS_BACKGROUND_LT3_IN = [(0.31574074074074077, 0.07222222222222222),
#                                 (0.012962962962962963, 0.41388888888888886),
#                                 (0.003703703703703704, 0.9972222222222222),
#                                 (0.9990740740740741, 0.9958333333333333),
#                                 (0.9953703703703703, 0.6041666666666666),
#                                 (0.7861111111111111, 0.29583333333333334),
#                                 (0.6620370370370371, 0.23472222222222222),
#                                 (0.6648148148148149, 0.03194444444444444),
#                                 (0.6648148148148149, 0.002777777777777778),
#                                 (0.2953703703703704, 0.005555555555555556),
#                                 (0.31296296296296294, 0.075)]
    
#     POINTS_BACKGROUND_LT3_OUT = [(0.004629629629629629, 0.9875),
#                                 (0.006481481481481481, 0.5930555555555556),
#                                 (0.28055555555555556, 0.3458333333333333),
#                                 (0.34629629629629627, 0.30277777777777776),
#                                 (0.3768518518518518, 0.14305555555555555),
#                                 (0.5935185185185186, 0.15694444444444444),
#                                 (0.975, 0.3638888888888889),
#                                 (0.9166666666666666, 0.7013888888888888),
#                                 (0.9898148148148148, 0.9861111111111112),
#                                 (0.005555555555555556, 0.9888888888888889)]

#     POINTS_BACKGROUND_LT4_IN = [
#                                 (0.12777777777777777, 0.9944444444444445),
#                                 (0.2101851851851852, 0.6555555555555556),
#                                 (0.3907407407407407, 0.23194444444444445),
#                                 (0.3990740740740741, 0.1763888888888889),
#                                 (0.6055555555555555, 0.19722222222222222),
#                                 (0.6240740740740741, 0.38333333333333336),
#                                 (0.6953703703703704, 0.45694444444444443),
#                                 (0.8648148148148148, 0.6430555555555556),
#                                 (0.9972222222222222, 0.9319444444444445),
#                                 (0.9694444444444444, 0.9986111111111111),
#                                 (0.07037037037037037, 0.9902777777777778),
#                                 ]
    
#     POINTS_BACKGROUND_LT4_OUT = [(0.004629629629629629, 0.9875),
#                                 (0.006481481481481481, 0.5930555555555556),
#                                 (0.28055555555555556, 0.3458333333333333),
#                                 (0.34629629629629627, 0.30277777777777776),
#                                 (0.3768518518518518, 0.14305555555555555),
#                                 (0.5935185185185186, 0.15694444444444444),
#                                 (0.975, 0.3638888888888889),
#                                 (0.9166666666666666, 0.7013888888888888),
#                                 (0.9898148148148148, 0.9861111111111112),
#                                 (0.005555555555555556, 0.9888888888888889)]

#     POINTS_BACKGROUND_LT5_IN = [(0.31574074074074077, 0.07222222222222222),
#                                 (0.012962962962962963, 0.41388888888888886),
#                                 (0.003703703703703704, 0.9972222222222222),
#                                 (0.9990740740740741, 0.9958333333333333),
#                                 (0.9953703703703703, 0.6041666666666666),
#                                 (0.7861111111111111, 0.29583333333333334),
#                                 (0.6620370370370371, 0.23472222222222222),
#                                 (0.6648148148148149, 0.03194444444444444),
#                                 (0.6648148148148149, 0.002777777777777778),
#                                 (0.2953703703703704, 0.005555555555555556),
#                                 (0.31296296296296294, 0.075)]

#     POINTS_BACKGROUND_LT5_OUT = [(0.004629629629629629, 0.9875),
#                                 (0.006481481481481481, 0.5930555555555556),
#                                 (0.28055555555555556, 0.3458333333333333),
#                                 (0.34629629629629627, 0.30277777777777776),
#                                 (0.3768518518518518, 0.14305555555555555),
#                                 (0.5935185185185186, 0.15694444444444444),
#                                 (0.975, 0.3638888888888889),
#                                 (0.9166666666666666, 0.7013888888888888),
#                                 (0.9898148148148148, 0.9861111111111112),
#                                 (0.005555555555555556, 0.9888888888888889)
#                                 ]

#     # POINT_LT2_1_START = (0.38055555555555554, 0.40694444444444444)
#     # POINT_LT2_2_START = (0.6370370370370371, 0.40694444444444444)
#     # POINT_LT2_1_MIDDLE = (0.344921875, 0.5381944444444444)
#     # POINT_LT2_2_MIDDLE = (0.722265625, 0.5263888888888889)
#     # POINT_LT2_1_END = (0.05740740740740741, 0.6611111111111111)
#     # POINT_LT2_2_END = (0.8111111111111111, 0.6708333333333333)
#     # POINT_LT2_1_END_2 = (0.11203703703703703, 0.9875)
#     # POINT_LT2_2_END_2 = (0.12222222222222222, 0.006944444444444444)
    
#     # POINT_LT3_1_START = (0.4027777777777778, 0.5208333333333334)
#     # POINT_LT3_2_START = (0.674074074074074, 0.5263888888888889)
#     # POINT_LT3_1_MIDDLE = (0.36574074074074076, 0.7)
#     # POINT_LT3_2_MIDDLE = (0.7601851851851852, 0.6972222222222222)
#     # POINT_LT3_1_END = (0.062037037037037036, 0.8638888888888889)
#     # POINT_LT3_2_END = (0.8388888888888889, 0.8569444444444444)
#     # POINT_LT3_1_END_2 = (0.11203703703703703, 0.9875)
#     # POINT_LT3_2_END_2 = (0.12222222222222222, 0.006944444444444444)

#     # POLYGON_POINT = [
#     #     POINT_LT2_1_START,
#     #     POINT_LT2_2_START,
#     #     POINT_LT2_1_MIDDLE,
#     #     POINT_LT2_2_MIDDLE,
#     #     POINT_LT2_1_END,
#     #     POINT_LT2_2_END,
#     #     POINT_LT2_1_END_2,
#     #     POINT_LT2_2_END_2,
#     # ]

#     # LT2
#     POINT_LT2_IN_L_START = (0.34629629629629627, 0.3277777777777778)
#     POINT_LT2_IN_R_START = (0.5972222222222222, 0.32222222222222224)
#     POINT_LT2_IN_L_END = (0.05740740740740741, 0.6611111111111111)
#     POINT_LT2_IN_R_END = (0.6907407407407408, 0.6694444444444444)
#     # POINT_LT2_OUT_L_START = (0.34629629629629627, 0.3277777777777778)
#     POINT_LT2_OUT_L_START = (0.35833333333333334, 0.39166666666666666)
#     # POINT_LT2_OUT_R_START = (0.5972222222222222, 0.32222222222222224)
#     POINT_LT2_OUT_R_START = (0.5787037037037037, 0.4)
#     POINT_LT2_OUT_L_END = (0.05462962962962963, 0.6611111111111111)
#     POINT_LT2_OUT_R_END = (0.8111111111111111, 0.6708333333333333)

#     # LT3
#     POINT_LT3_IN_L_START = (0.34629629629629627, 0.3277777777777778)
#     POINT_LT3_IN_R_START = (0.5972222222222222, 0.32222222222222224)
#     POINT_LT3_IN_L_END = (0.05462962962962963, 0.6611111111111111)
#     POINT_LT3_IN_R_END = (0.6907407407407408, 0.6694444444444444)
#     POINT_LT3_OUT_L_START = (0.34629629629629627, 0.3277777777777778)
#     POINT_LT3_OUT_R_START = (0.5972222222222222, 0.32222222222222224)
#     POINT_LT3_OUT_L_END = (0.05462962962962963, 0.6611111111111111)
#     POINT_LT3_OUT_R_END = (0.6907407407407408, 0.6694444444444444)

#     # LT4
#     POINT_LT4_IN_L_START = (0.34629629629629627, 0.3277777777777778)
#     POINT_LT4_IN_R_START = (0.5972222222222222, 0.32222222222222224)
#     POINT_LT4_IN_L_END = (0.05462962962962963, 0.6611111111111111)
#     POINT_LT4_IN_R_END = (0.6907407407407408, 0.6694444444444444)
#     POINT_LT4_OUT_L_START = (0.34629629629629627, 0.3277777777777778)
#     POINT_LT4_OUT_R_START = (0.5972222222222222, 0.32222222222222224)
#     POINT_LT4_OUT_L_END = (0.05462962962962963, 0.6611111111111111)
#     POINT_LT4_OUT_R_END = (0.6907407407407408, 0.6694444444444444)

#     # LT5
#     POINT_LT5_IN_L_START = (0.34629629629629627, 0.3277777777777778)
#     POINT_LT5_IN_R_START = (0.5972222222222222, 0.32222222222222224)
#     POINT_LT5_IN_L_END = (0.05462962962962963, 0.6611111111111111)
#     POINT_LT5_IN_R_END = (0.6907407407407408, 0.6694444444444444)
#     POINT_LT5_OUT_L_START = (0.34629629629629627, 0.3277777777777778)
#     POINT_LT5_OUT_R_START = (0.5972222222222222, 0.32222222222222224)
#     POINT_LT5_OUT_L_END = (0.05462962962962963, 0.6611111111111111)
#     POINT_LT5_OUT_R_END = (0.6907407407407408, 0.6694444444444444)


#     POLYGON_POINT_LT2_IN = [
#         POINT_LT2_IN_L_START,
#         POINT_LT2_IN_R_START,
#         POINT_LT2_IN_L_END,
#         POINT_LT2_IN_R_END        
#     ]

#     POLYGON_POINT_LT2_OUT = [
#         POINT_LT2_OUT_L_START,
#         POINT_LT2_OUT_R_START,
#         POINT_LT2_OUT_L_END,
#         POINT_LT2_OUT_R_END        
#     ]

#     POLYGON_POINT_LT3_IN = [
#         POINT_LT3_IN_L_START,
#         POINT_LT3_IN_R_START,
#         POINT_LT3_IN_L_END,
#         POINT_LT3_IN_R_END        
#     ]    

#     POLYGON_POINT_LT3_OUT = [
#         POINT_LT3_OUT_L_START,
#         POINT_LT3_OUT_R_START,
#         POINT_LT3_OUT_L_END,
#         POINT_LT3_OUT_R_END
#     ]

#     POLYGON_POINT_LT4_IN = [
#         POINT_LT4_IN_L_START,
#         POINT_LT4_IN_R_START,
#         POINT_LT4_IN_L_END,
#         POINT_LT4_IN_R_END        
#     ]    

#     POLYGON_POINT_LT4_OUT = [
#         POINT_LT4_OUT_L_START,
#         POINT_LT4_OUT_R_START,
#         POINT_LT4_OUT_L_END,
#         POINT_LT4_OUT_R_END
#     ]

#     POLYGON_POINT_LT5_IN = [
#         POINT_LT5_IN_L_START,
#         POINT_LT5_IN_R_START,
#         POINT_LT5_IN_L_END,
#         POINT_LT5_IN_R_END        
#     ]    

#     POLYGON_POINT_LT5_OUT = [
#         POINT_LT5_OUT_L_START,
#         POINT_LT5_OUT_R_START,
#         POINT_LT5_OUT_L_END,
#         POINT_LT5_OUT_R_END
#     ]

# config = Config()



# import random
# BASE_DIR = Path(__file__).parent.parent.resolve()


# MODEL_PATH = BASE_DIR / "weights/yolov8n.pt"

# #MQTT
# BROKER = '192.168.88.60'
# PORT = 1883
# TOPIC = "DOT/UAT/LAMP"
# CLIENT_ID = f'publish-{random.randint(0, 1000)}'


# # PARKING_LOTS = [
# # [(888, 879), (1180, 771), (1398, 917), (1072, 1048), (889, 883)],
# # [(450, 994), (887, 878), (1072, 1048), (471, 1048), (450, 993)],
# # [(1182, 768), (1384, 697), (1586, 806), (1398, 914), (1182, 768)],
# # [(1626, 782), (1911, 899), (1911, 1048), (1508, 1048), (1434, 915), (1627, 784)],
# # # [(1590, 805), (1893, 918), (1749, 1048), (1399, 915)],
# # [(396, 705), (438, 916), (736, 846), (657, 647), (398, 703)],
# # [(656, 647), (737, 847), (964, 781), (880, 598), (659, 647)],
# # [(882, 597), (962, 778), (1214, 701), (1098, 548), (882, 594)]
# # # [(1479, 521), (1329, 597), (1512, 674), (1631, 592), (1481, 521)],
# # # [(1632, 588), (1847, 668), (1704, 755), (1514, 673), (1632, 592)],
# #     #data 6
# # # [(1063, 1048), (932, 885), (1193, 772), (1340, 900), (1064, 1047)],
# # # [(412, 758), (650, 700), (715, 901), (452, 966), (414, 758)],
# # # [(650, 701), (867, 651), (942, 813), (718, 905), (653, 700)],
# # # [(867, 653), (1078, 599), (1227, 699), (943, 810), (870, 655)],
# # # [(1610, 602), (1481, 691), (1694, 795), (1795, 667), (1608, 599)],
# # # [(479, 1048), (932, 892), (1062, 1046), (479, 1048)],
# # ]
# #
# # PARKING_LOTS2 = [
# # [(313, 749), (791, 798), (830, 604), (499, 578), (313, 746)],
# # [(500, 577), (588, 490), (852, 506), (829, 606), (501, 578)],
# # [(851, 503), (1094, 514), (1090, 625), (831, 608), (852, 502)],
# # [(827, 610), (1089, 628), (1072, 798), (799, 797), (823, 613)],
# # [(1617, 745), (1298, 825), (1208, 683), (1450, 632), (1617, 744)],
# # [(1326, 542), (1156, 594), (1208, 682), (1451, 631), (1334, 540)],
# # [(1270, 493), (1123, 504), (1156, 592), (1334, 540), (1272, 495)],
# # [(1378, 568), (1560, 486), (1685, 531), (1457, 634), (1376, 568)],
# # [(1684, 530), (1894, 621), (1620, 747), (1456, 632), (1684, 534)],
# # ]

# CLASS_NAMES = {
#     0: 'person',
#     # 1: 'bicycle',
#     2: 'car',
#     # 3: 'motorcycle',
#     # 5: 'bus',
#     7: 'truck'
#     # Add more classes if needed
# }