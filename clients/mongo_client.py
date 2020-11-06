from mongoengine import *
import functools
from datetime import datetime

from utils import constants


def mongo_decorator(func):
    """
    A decorator that wraps the passed in function, sandwiching it between a
    db connection
    """
    @functools.wraps(func)
    def wrap_function(*args, **kwargs):
        db_client = MongoClient.connect_db(user=constants.MONGO_USER,
                                           password=constants.MONGO_PASSWORD,
                                           host=constants.MONGO_HOST,
                                           port=constants.MONGO_PORT,
                                           db_name=constants.MONGO_DATABASE)

        func_response = func(*args, **kwargs)
        db_client.close()
        db_client = None
        try:
            MongoClient.disconnect_db()
        except Exception as e:
            print(e)
            pass
        return func_response
    return wrap_function


class MongoClient:

    @staticmethod
    def connect_db(host, port, db_name, user=None, password=None):
        # print("call to initialize db connection")
        return connect(db_name, host=host, port=port, username=user, password=password)

    @staticmethod
    def disconnect_db():
        disconnect()
