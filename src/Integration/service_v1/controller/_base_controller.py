from src.Integration.service_v1.configs.config import setting
from src.Integration.service_v1.db.database import Base, engine, SessionLocal, scop_ses


class SingletonMeta(type):
    """
    A Singleton metaclass that creates a single instance for the class.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Singleton(metaclass=SingletonMeta):
    """
    A base class to be inherited by classes that should be singletons.
    """
    _initialized = False
    
    
# class BaseController(Singleton):
#
#     _session = None
#
#     def __init__(self):
#         Base.metadata.create_all(bind=engine)
#         if BaseController._session is None:
#             BaseController._session = SessionLocal()
#         self._base_url = setting.DATABASE_URL
#
#     @property
#     def session(self):
#         return self._session

class BaseController(Singleton):

    def __init__(self):
        Base.metadata.create_all(bind=engine)
        self._base_url = setting.DATABASE_URL

    @property
    def session(self):
        return scop_ses()

    def close_session(self):
        """Close the session. Useful if you're managing the session manually."""
        scop_ses.remove()


