import redis
import configurations
import json

class RedisProvider:
    def __init__(self):
        try:
            self.db = redis.Redis(host=configurations.redis_host, port=configurations.redis_port, decode_responses=True)
        except Exception as e:
            print(e)

    def set_value_in_redis(self,key,value):
        if key and value:
            value_to_store = json.dumps(value)
            self.db.set(key,value_to_store)

    def get_value_in_redis(self, key):
        if key:
            retrieved_value = self.db.get(key)
            if retrieved_value:
                return json.loads(retrieved_value)
            return None
        return None

        
if __name__ == '__main__':
    redisprovider=RedisProvider()
    redisprovider.set_value_in_redis("test","chess")
    x= redisprovider.get_value_in_redis("test")
    print(x)