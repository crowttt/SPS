import redis

REDIS_ENGINE = redis.StrictRedis.from_url("redis://:password@localhost:6380/1")

high_risk_class = [
    'pole vault',
    'passing American football (in game)',
    'capoeira',
    'dunking basketball',
    'wrestling',
    'dodgeball',
    'gymnastics tumbling',
    'hammer throw',
    'shot put',
    # 'high jump'
    # 'dunking basketball'
    'javelin throw'
]

low_risk_class = [
    'playing violin',
    'playing chess',
    'folding clothes',
    'playing ukulele',
    'tai chi',
    'blowing out candles',
    'dying hair',
    'playing clarinet',
    'tango dancing',
    'eating spaghetti',
]
