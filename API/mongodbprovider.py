from pymongo import MongoClient
from mongo_db_operations import Configurations

class MongoDBOperations:
    def __init__(self):
        self._mongo_client = MongoClient(Configurations.MONGO_DB_HOST, int(Configurations.MONGO_DB_PORT))
        print("Successfully connected to Mongo DB host: {0} and port: {1}".format(Configurations.MONGO_DB_HOST,
                                                                                  str(Configurations.MONGO_DB_PORT)))
        self._set_up_db_if_not_exists()

    def _set_up_db_if_not_exists(self):

        database_name = Configurations.DATABASE_NAME
        uninserted_collection_name = Configurations.UNINSERTED_COLLECTION_NAME

        self._database = self._mongo_client[database_name]
        self._uninserted_collection_name = uninserted_collection_name
        self._team_names = Configurations.TEAM_NAMES

    def insert_details(self, team_name, details):
        """Inserts multiple details in the collection for the team_name

        Args:
            team_name: The name of the team from ["kolkata", "bengaluru", "chennai", "delhi", "goa", "jamshedpur", "kerala", "mumbai", "northeast", "pune"]
            details (list): The details captured for the team as a **list** 

        Note:
            Stores all exceptions to the UNINSERTED collection

        """
        if details and len(details) > 0:

            if team_name not in self._team_names:
                raise ValueError("The provided team name {0} is not configured for storing".format(team_name))
            
            insert_list = []
            # print(details)
            for detail in details:
                are_all_mandatory_fields_present = True
                for mandatory_field in Configurations.MANDATORY_FIELDS:
                    if mandatory_field not in detail:
                        print("Mandatory field {0} is not present for this record {1}.".format(mandatory_field, detail))
                        are_all_mandatory_fields_present = False
                        break
                
                if are_all_mandatory_fields_present:
                    insert_list.append(detail)

            try:
                # print(insert_list)
                self._database[team_name].insert_many(insert_list)
                print("Inserted {0} documents in the collection".format(str(len(insert_list))))
            except Exception as e:
                print("An error occurred during the insertion. Check Mongo DB for details")
                error_details = {}
                error_details["error_text"] = "Runtime error: " + str(e)
                error_details["details"] = details
                self._database[self._uninserted_collection_name].insert_one(error_details)

            return len(insert_list)
        else:
            print("No details to insert...")
            return False
    
    def insert_crawling_error(self, scraped_detail, error_text):
        if scraped_detail:
            scraped_detail["error_text"] = "Crawling error: " + error_text
            self._database[self._uninserted_collection_name].insert_one(scraped_detail)