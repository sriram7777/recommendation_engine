import json
import os
import numpy as np
import requests


script_name = os.path.basename(__file__)
script_dir = os.path.dirname(os.path.abspath(__file__))

from mapper import Mapper
from utils import constants
from utils.function_factory import timing, divide_chunks
from clients.logger_client import LoggerClient
from clients.redis_client import RedisClient
# from utils.statsd_events import *
from mongo_factory.user_data_dao import UserData
from clients.mongo_client import MongoClient
from mongo_factory.user_data_dao_impl import UserDataDaoImpl

logger = LoggerClient.set_logger("logs/background_sync.log")
redis_client = RedisClient(host=constants.REDIS_HOST, port=constants.REDIS_PORT).get_client()

MongoClient.connect_db(host=constants.MONGO_HOST, port=constants.MONGO_PORT, db_name=constants.MONGO_DATABASE,
                       user=constants.MONGO_USER, password=constants.MONGO_PASSWORD)


def update_mongo(doc):
    UserDataDaoImpl.insert_doc(doc)


@timing(logger=logger)
def reload_faiss_index():
    url = "http://" + constants.FAISS_APP_HOST + ":" + constants.FAISS_APP_PORT + "/faiss_app/v1/reload_vectors"
    result = requests.get(url)


@timing(logger=logger)
def update_faiss_index(items, item_ids, vectors):
    if type(vectors) != list:
        vectors = vectors.tolist()
    if type(items) != list:
        items = items.tolist()
    if type(item_ids) != list:
        item_ids = item_ids.tolist()

    payload = json.dumps({'ids': item_ids,
            'item_ids': items,
            'vectors': vectors})

    url = "http://" + constants.FAISS_APP_HOST + ":" + constants.FAISS_APP_PORT + "/faiss_app/v1/update_vectors"
    result = requests.post(url, payload)


@timing(logger=logger)
def update_redis_cache(mapper_dict):
    for key, value in mapper_dict.items():
        redis_client.hmset(key, value)

# @statsd_client.timer("faiss_update_time")
def update_item_vectors():
    # update faiss
    logger.info("Number of items to update: {}".format(len(updated_items)))
    # trigger reload of the index when there are too many updates
    if len(updated_items) > 0.10 * len(item_mapper):
        logger.info("Number of items to update is more than 10% of total items. Triggering reloading of index")
        reload_faiss_index()
    else:
        item_embeddings = np.load(os.path.join(script_dir, 'models', 'item_embeddings.npy'))
        updated_item_ids = np.array([item_mapper.retrieve_index(item)[0] for item in updated_items])
        updated_item_vectors = np.take(item_embeddings, updated_item_ids, axis=0)
        del item_embeddings
        logger.info("Number of faiss index update requests: {}".format(len(updated_items)//10000+1))
        for a, b, c in zip(divide_chunks(updated_items, 100), divide_chunks(updated_item_ids, 100),
                           divide_chunks(updated_item_vectors, 100)):
            update_faiss_index(a, b, c)
        del updated_item_vectors


# @statsd_client.timer("mongo_update_time")
@timing(logger=logger)
def update_user_info():
    # update mongodb with user info
    logger.info("Starting user info update in mongo")
    user_embeddings = np.load(os.path.join(script_dir, 'models', 'user_embeddings.npy'))

    updated_user_ids = np.array([user_mapper.retrieve_index(user) for user in updated_users])
    updated_user_vectors = np.take(user_embeddings, updated_user_ids, axis=0)
    del user_embeddings

    # todo: can do update many by using native mongo queries, instead of doing it sequentially
    for updated_user, updated_user_id, updated_user_vector in zip(updated_users, updated_user_ids, updated_user_vectors):
        update_mongo(UserData(user_id=updated_user, user_embedding=updated_user_vector,
                              user_index=updated_user_id))

    del updated_user_vectors

# @statsd_client.timer("redis_update_time")
def update_item_info():
    # update redis cache with item info
    logger.info("Starting item info update in redis")
    item_mapper_updated = {k: item_mapper.retrieve_index(k) for k in updated_items}
    item_mapper_updated = {k: {x: y for x, y in zip(['index', 'ontology', 'price'], v)}
                               for k, v in item_mapper_updated.items()}

    update_redis_cache(item_mapper_updated)


if __name__ == '__main__':
    # try:
    user_mapper = Mapper()
    user_mapper.load(os.path.join(script_dir, 'mappers', 'user'))
    item_mapper = Mapper()
    item_mapper.load(os.path.join(script_dir, 'mappers', 'item'))
    metadata = json.load(open(os.path.join(script_dir, 'mappers', 'metadata.json'), 'r'))
    logger.info("User and Item mappers loaded successfully")

    if 'updated_items' in metadata:
        updated_items = metadata['updated_items']
    else:
        updated_items = list(item_mapper.mapper_dict.keys())

    if 'updated_users' in metadata:
        updated_users = metadata['updated_users']
    else:
        updated_users = list(user_mapper.mapper_dict.keys())

    # update_item_vectors()
    update_user_info()
    update_item_info()

