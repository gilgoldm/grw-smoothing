import configparser
from typing import Optional


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class GrwSmoothingModelsConfig(metaclass=Singleton):
    def __init__(self, config_path: Optional[str] = None):
        self.config_path: str = config_path
        config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        config.read(config_path)
        self.config: configparser.ConfigParser = config
