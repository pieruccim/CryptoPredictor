import os

class Utils:
    env_path = os.path("../.env")

    def __init__(self):
        pass

    def load_config(self, config_key : str) -> str :
        return os.getenv(config_key)