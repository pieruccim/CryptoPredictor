import pymongo as pymongo

class MongoConnectionManager:

    #username = cryptopredictor
    #password = cryptopredictor
    # db name = cryptopredictor
    CONNECTION_STRING = "mongodb+srv://cryptopredictor:<cryptopredictor>@cluster0.lwnpx.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"
    DATABASE_NAME = "cryptopredictor"

    COLLECTION_NAME = "bitcoin"

    __mongo_client = pymongo.MongoClient(CONNECTION_STRING)
    __mongo_database = __mongo_client[DATABASE_NAME]

    def get_collection(collection_name: str):
        return MongoConnectionManager.__mongo_database.get_collection(collection_name)

    def get_database(self):
        return MongoConnectionManager.__mongo_database