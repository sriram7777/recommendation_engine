import redis
import os
import json
import logging
from datetime import datetime


class RedisClient:
    __client = None

    @staticmethod
    def get_client():
        if RedisClient.__client == None:
            RedisClient()
        return RedisClient.__client

    def __init__(self, host, port):
        if RedisClient.__client != None:
            raise Exception("This class is a singleton!")
        else:
            RedisClient.__client = redis.Redis(host=host, port=port, db=0)
