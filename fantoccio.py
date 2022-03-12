from persistence.MongoConnector import MongoConnector


result = MongoConnector.get_collection('bitcoin').find()
print(result.next())


