import os
from pathlib import Path
from dotenv import load_dotenv


env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

class Config:
    IS_FOLDER = os.getenv("IS_FOLDER", "False").lower() == "true"
    IS_BLACK_PLATE = os.getenv("IS_BLACK_PLATE", "False").lower() == "true"
    IS_OLD_MODEL = os.getenv("IS_OLD_MODEL", "True").lower() == "true"
    IS_RESTORATION = os.getenv("IS_RESTORATION", "False").lower() == "true"

    OLD_MODEL_PATH = os.getenv("OLD_MODEL_PATH")
    NEW_MODEL_PATH = os.getenv("NEW_MODEL_PATH")

    BLACK_PLATE_PATH = os.getenv("BLACK_PLATE_PATH")
    WHITE_PLATE_PATH = os.getenv("WHITE_PLATE_PATH")

    BLACK_PLATE_RESTORATION_PATH = os.getenv("BLACK_PLATE_RESTORATION_PATH")
    WHITE_PLATE_RESTORATION_PATH = os.getenv("WHITE_PLATE_RESTORATION_PATH")

    IMG_PATH_BLACK = os.getenv("IMG_PATH_BLACK")
    IMG_PATH_WHITE = os.getenv("IMG_PATH_WHITE")

    IMG_PATH_BLACK_RESTORATION = os.getenv("IMG_PATH_BLACK_RESTORATION")
    IMG_PATH_WHITE_RESTORATION = os.getenv("IMG_PATH_WHITE_RESTORATION")

    @staticmethod
    def get_model_paths():
        # if Config.IS_OLD_MODEL:
        #     main_model_path = Config.OLD_MODEL_PATH
        #     # print("OLD MODEL")
        # else:
        #     main_model_path = Config.NEW_MODEL_PATH
        #     # print("NEW MODEL")

        main_model_path = r"D:\engine\smart_parking\repository\feature-OCR-Dev\ai-smartparking\src\weights\ocr_model\new_model\20240925-11-01-14"

        model_path = os.path.join(main_model_path, "character_recognition.json")
        weight_path = os.path.join(main_model_path, "models_cnn.h5")
        labels_path = os.path.join(main_model_path, "character_classes.npy")

        return model_path, weight_path, labels_path


config = Config()