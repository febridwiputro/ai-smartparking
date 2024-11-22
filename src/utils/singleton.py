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
