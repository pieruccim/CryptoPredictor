from dotenv import load_dotenv
import os

class Utils:

    def __init__(self):
        pass

    def load_config(config_key: str) -> str:
        load_dotenv()
        return os.getenv(config_key)
