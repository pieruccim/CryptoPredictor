from dotenv import load_dotenv
import os


class Utils:
    env_path = "../.env"

    def __init__(self):
        print(self.load_config("PROVA"))

    def load_config(self, config_key: str) -> str:
        load_dotenv(self.env_path)
        return os.getenv(config_key)
