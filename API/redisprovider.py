import redis
import configurations
import json

class RedisProvider:
    def __init__(self):
        self._use_redis_db = configurations.APP_FLAGS["USE_REDIS"]
        if self._use_redis_db:
            try:
                self.db = redis.Redis(host=configurations.redis_host, port=configurations.redis_port, decode_responses=True)
                print("Successfully connected to Redis database")
            except Exception as e:
                print("Failed to connect to Redis when using Redis...")
                print(e)
        else:
            print("Not using Redis DB...")
            self.db = {}
        

    def set_value_in_redis(self,key,value):
        if key and value:
            value_to_store = json.dumps(value)
            if self._use_redis_db:
                self.db.set(key,value_to_store)
            else:
                self.db[key] = value_to_store

    def get_value_in_redis(self, key):
        if key:
            if self._use_redis_db:
                retrieved_value = self.db.get(key)
                if retrieved_value:
                    return json.loads(retrieved_value)
                return None
            else:
                return json.loads(self.db.get(key))
        print("No key specified...")
        return None

        
if __name__ == '__main__':
    redisprovider=RedisProvider()
    redisprovider.set_value_in_redis("test","chess")
    x= redisprovider.get_value_in_redis("test")
    print(x)