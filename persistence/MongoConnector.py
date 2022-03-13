import pymongo as pymongo
from utilities.utils import Utils


class MongoConnector:
    CONNECTION_STRING = Utils.load_config("CONNECTION_STRING")
    DATABASE_NAME = Utils.load_config("DATABASE_NAME")

    __mongo_client = pymongo.MongoClient(CONNECTION_STRING)
    __mongo_database = __mongo_client[DATABASE_NAME]

    @staticmethod
    def get_collection(collection_name: str):
        return MongoConnector.__mongo_database.get_collection(collection_name)

    @staticmethod
    def get_database():
        return MongoConnector.__mongo_database

    def get_client(self):
        return MongoConnector.__mongo_client

    def connection_test(self):
        try:
            client = pymongo.MongoClient(self.CONNECTION_STRING)
            client.server_info()
        except pymongo.errors.ServerSelectionTimeoutError as err:
            print(err)
