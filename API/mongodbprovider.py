from pymongo import MongoClient
import configurations
import constants
import json

class MongoDBProvider:
    def __init__(self):
        self._use_mongo_db = configurations.APP_FLAGS["USE_MONGO"]
        if self._use_mongo_db:
            self._mongo_client = MongoClient(configurations.MONGO_DB_HOST, int(configurations.MONGO_DB_PORT))
            print("Successfully connected to Mongo DB host: {0} and port: {1}".format(configurations.MONGO_DB_HOST,
                                                                                    str(configurations.MONGO_DB_PORT)))
            self._database = self._mongo_client[constants.database_name]
        else:
            print("Not using MongoDB...")
    
    def insert_record_with_properties(self, record, additional_properties, collection_name):
        if self._use_mongo_db and record and additional_properties:
            record_to_insert = record
            for k in additional_properties:
                if k not in record_to_insert:
                    record_to_insert[k] = additional_properties[k]

            self.insert_record(record_to_insert, collection_name)
        else:
            print("{0} with properties {1}".format(record, additional_properties))

    def insert_record(self, record, collection_name):
        if self._use_mongo_db:
            self._database[collection_name].insert_one(record)
        else:
            print(record)

    def retrieve_record(self,collection_name, query_obj):
        if query_obj and self._use_mongo_db:
            return self._database[collection_name].find_one(query_obj)
        else:
            print("Not connected to MongoDB, so cannot retrieve values. Check APP_FLAGS in configuration.")

if __name__ == '__main__':
    mongodbprovider=MongoDBProvider()
    doc={"piece": "wk", "position": "h5"}

    # mongodbprovider.insert_record(doc, "mongo")
    # mongodbprovider.insert_record(doc, "mongo")
    print(mongodbprovider.retrieve_record("mongo", {"piece": "wf"}))