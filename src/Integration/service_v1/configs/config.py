from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    DATABASE_URL: str = 'postgresql://postgres:postgres@localhost:5432/ocr_db'    
    # DATABASE_URL: str = os.getenv("DATABASE_URL") if os.getenv("STATUS") != "DEVELOPMENT" else os.getenv("DATABASE_URL_DEV")
    SECRET_KEY: str = os.getenv("SECRET_KEY")
    ALGORITHM: str = os.getenv("ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")
    VERSION: str = os.getenv("VERSION")
    PREFIX: str = os.getenv("PREFIX")
    PORT: int = os.getenv("PORT")


setting = Settings()
