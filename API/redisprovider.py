import redis
import configurations

class RedisProvider:
    def __init__(self):
        try:
            self.db = redis.Redis(host=configurations.redis_host, port=configurations.redis_port, decode_responses=True)
        except Exception as e:
            print(e)

    def set_value_in_redis(self,key,value):
        self.db.set(key,value)

    def get_value_in_redis(self, key):
        return self.db.get(key)

        
if __name__ == '__main__':
    redisprovider=RedisProvider()
    redisprovider.set_value_in_redis("test","chess")
    x= redisprovider.get_value_in_redis("test")
    print(x)