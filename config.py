import redis

REDIS_ENGINE = redis.StrictRedis.from_url("redis://:password@localhost:6380/1")
