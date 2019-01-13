from pymongo import MongoClient
import configurations
import constants
import json

class MongoDBProvider:
    def __init__(self):
        self._mongo_client = MongoClient(configurations.MONGO_DB_HOST, int(configurations.MONGO_DB_PORT))
        print("Successfully connected to Mongo DB host: {0} and port: {1}".format(configurations.MONGO_DB_HOST,
                                                                                  str(configurations.MONGO_DB_PORT)))
        self._database = self._mongo_client[constants.database_name]

    def insert_record(self, record, collection_name):
        self._database[collection_name].insert_one(record)

    def retrieve_record(self,collection_name, query_obj):
        if query_obj:
            return self._database[collection_name].find_one(query_obj)

if __name__ == '__main__':
    mongodbprovider=MongoDBProvider()
    doc={"piece": "wk", "position": "h5"}

    # mongodbprovider.insert_record(doc, "mongo")
    # mongodbprovider.insert_record(doc, "mongo")
    print(mongodbprovider.retrieve_record("mongo", {"piece": "wf"}))