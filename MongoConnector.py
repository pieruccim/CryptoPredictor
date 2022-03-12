import pymongo as pymongo
from utilities.utils import Utils


class MongoConnector:
    #CONNECTION_STRING = "mongodb+srv://cryptopredictor:cryptopredictor@cluster0.lwnpx.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"
    CONNECTION_STRING = Utils.load_config("CONNECTION_STRING")
    DATABASE_NAME = "cryptopredictor"

    COLLECTION_NAME = "bitcoin"

    __mongo_client = pymongo.MongoClient(CONNECTION_STRING)
    __mongo_database = __mongo_client[DATABASE_NAME]

    def __init__(self):
        self.connection_test()

    def get_collection(collection_name: str):
        return MongoConnector.__mongo_database.get_collection(collection_name)

    def get_database(self):
        return MongoConnector.__mongo_database

    def connection_test(self):
        try:
            client = pymongo.MongoClient(self.CONNECTION_STRING)
            client.server_info()  # force connection on a request as the
            # connect=True parameter of MongoClient seems
            # to be useless here
        except pymongo.errors.ServerSelectionTimeoutError as err:
            # do whatever you need
            print(err)
