import pymongo as pymongo
from utilities.utils import Utils


class MongoConnector:
    CONNECTION_STRING = Utils.load_config("CONNECTION_STRING")
    DATABASE_NAME = "cryptopredictor"

    COLLECTION_NAME = "bitcoin"

    __mongo_client = pymongo.MongoClient(CONNECTION_STRING)
    __mongo_database = __mongo_client[DATABASE_NAME]

    def get_collection(collection_name: str):
        return MongoConnector.__mongo_database.get_collection(collection_name)

    @staticmethod
    def get_database():
        return MongoConnector.__mongo_database

    def get_client(self):
        return MongoConnector.__mongo_client

    def connection_test(self):
        print(self.CONNECTION_STRING)
        try:
            client = pymongo.MongoClient(self.CONNECTION_STRING)
            client.server_info()  # force connection on a request as the
            # connect=True parameter of MongoClient seems
            # to be useless here
        except pymongo.errors.ServerSelectionTimeoutError as err:
            # do whatever you need
            print(err)
