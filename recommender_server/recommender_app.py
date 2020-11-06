import os
import falcon
import numpy as np
import sys
import json
import requests

script_name = os.path.basename(__file__)
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.join(script_dir, "..") not in sys.path:
    sys.path.insert(0, os.path.join(script_dir, ".."))

# from general_utils.statsd_events import *
from utils import constants
from utils.function_factory import timing
from clients.logger_client import LoggerClient
from clients.redis_client import RedisClient
from clients.mongo_client import MongoClient
# from utils.statsd_events import *
from mongo_factory.user_data_dao_impl import UserDataDaoImpl

logger = LoggerClient.set_logger(os.path.join(script_dir, "../logs/recommender_app.log"))

redis_client = RedisClient(host=constants.REDIS_HOST, port=constants.REDIS_PORT).get_client()

MongoClient.connect_db(host=constants.MONGO_HOST, port=constants.MONGO_PORT, db_name=constants.MONGO_DATABASE,
                       user=constants.MONGO_USER, password=constants.MONGO_PASSWORD)


def recommend_items_by_vector(embedding, k=10):
    url = "http://" + constants.FAISS_APP_HOST + ":" + constants.FAISS_APP_PORT + "/faiss_app/v1/search_by_vectors"
    payload = json.dumps({"vectors": [embedding], "k": k})

    res = requests.post(url, payload)
    if res.status_code == 200:
        response_dict = res.json()
        if response_dict['es'] == 0:
            return response_dict["result"]["0"]
        else:
            logger.error("error state true in response")
            return []

    else:
        logger.error("Request to faiss app failed")
        return []


def recommend_items_by_user_id(user_id, k):
    user_doc = UserDataDaoImpl.find_user(user_id)
    if user_doc is not None:
        user_embedding = user_doc.user_embedding
        recommended_items = recommend_items_by_vector(user_embedding, k)
        recommended_items_dict = {}
        for item in recommended_items:
            if redis_client.exists(item):
                recommended_items_dict[item] = float(redis_client.hmget(item, 'price')[0])
        if len(recommended_items_dict) > 0:
            recommended_items_dict = {k: v for k, v in sorted(recommended_items_dict.items(), key=lambda item: item[1])}
            return list(recommended_items_dict.keys())
        else:
            return []
    else:
        return []


def recommend_items_by_item_id(item_id, k):
    if redis_client.exists(item_id):
        item_index = int(redis_client.hmget(item_id, "index")[0])
        item_ontology = redis_client.hmget(item_id, "ontology")[0].decode("utf-8")
        item_price = float(redis_client.hmget(item_id, "price")[0])

        url = "http://" + constants.FAISS_APP_HOST + ":" + constants.FAISS_APP_PORT + "/faiss_app/v1/search_by_ids"
        payload = json.dumps({"ids": [item_index], "k": 2*k})

        res = requests.post(url, payload)

        if res.status_code == 200:
            response_dict = res.json()
            if response_dict['es'] == 0:
                recommended_items = response_dict["result"][str(item_index)]
                if item_id in recommended_items:
                    recommended_items.remove(item_id)
                recommended_items_same_ontology_dict = {}
                recommended_items_diff_ontology_dict = {}
                # giving preference to same ontology and ordering by price closeness to current item
                for item in recommended_items:
                    if redis_client.exists(item):
                        if redis_client.hmget(item, "ontology")[0].decode('utf-8') == item_ontology:
                            recommended_items_same_ontology_dict[item] = abs(float(redis_client.hmget(item, 'price')[0]) - item_price)
                        else:
                            recommended_items_diff_ontology_dict[item] = abs(float(redis_client.hmget(item, 'price')[0]) - item_price)
                final_recommendations = []
                if len(recommended_items_same_ontology_dict) > 0:
                    same_ontology_items = list({k: v for k, v in
                                              sorted(recommended_items_same_ontology_dict.items(), key=lambda item: item[1])}.keys())
                    final_recommendations.append(same_ontology_items)
                if len(recommended_items_diff_ontology_dict) > 0:
                    diff_ontology_items = list({k: v for k, v in
                                              sorted(recommended_items_diff_ontology_dict.items(), key=lambda item: item[1])}.keys())
                    final_recommendations.append(diff_ontology_items)
                return final_recommendations

            else:
                logger.error("error state true in response")
                return []

        else:
            logger.error("Request to faiss app failed")
            return []


class RecommendItems(object):

    """

    """

    def __init__(self):
        self.k = 5

    # @statsd_client.timer("recommender_latency")
    @timing(logger=logger)
    def on_get(self, req, resp):
        """Handles GET requests"""
        user_id = req.get_param('user_id', default=None)
        if type(user_id) != str and user_id is not None:
            raise falcon.HTTPError(falcon.HTTP_400, "user_id parameter should be a string")

        item_id = req.get_param('item_id', default=None)
        if type(item_id) != str and item_id is not None:
            raise falcon.HTTPError(falcon.HTTP_400, "item_id parameter should be a string")

        if user_id is None and item_id is None:
            raise falcon.HTTPError(falcon.HTTP_400, "either user_id or item_id has to be provided")

        k = int(req.get_param('k', default=self.k))
        logger.debug("Got the recommendation request for user id and item id. {}".format(user_id, item_id))
        recommend_response = {}
        recommended_items = []
        if user_id is not None:
            recommended_items = recommend_items_by_user_id(user_id, k)
            based_on = "user_id"
            if len(recommended_items) == 0:
                logger.warning("user_id not found in index")

        if len(recommended_items) == 0 and item_id is not None:
            recommended_items = recommend_items_by_item_id(item_id, k)
            based_on = 'item_id'
            if len(recommended_items) == 0:
                logger.warning("item_id not found in index")

        if len(recommended_items) == 0:
            based_on = "popular_items"
            recommended_items = redis_client.get('popular_items')["items"][:k]

        recommend_response['es'] = 0
        recommend_response['recommendations'] = recommended_items
        recommend_response['based_on'] = based_on

        resp.status = falcon.HTTP_200
        resp.body = json.dumps(recommend_response)


recommendItems = RecommendItems()

# API initialization
api = application = falcon.API()
api.add_route('/faiss_app/v1/recommend_items', recommendItems)

if __name__ == '__main__':
    # print(recommend_items_by_user_id("62b2ea64f905f87426778f9bd6e98293", k=10))
    print(recommend_items_by_item_id("000598800348bfdf28bb4c8062cc35d1", k=10))