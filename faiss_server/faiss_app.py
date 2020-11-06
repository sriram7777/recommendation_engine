import os
import falcon
import numpy as np
import sys
import json
import time

script_name = os.path.basename(__file__)
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.join(script_dir, "..") not in sys.path:
    sys.path.insert(0, os.path.join(script_dir, ".."))

# from general_utils.statsd_events import *
from clients.logger_client import LoggerClient
from utils.function_factory import timing
from faiss_server.faiss_index import FaissIndex

logger = LoggerClient.set_logger(os.path.join(script_dir, "../logs/faiss_app.log"))


@timing(logger)
def load_mapper():
    items = list(json.load(open(os.path.join(script_dir, '..', 'mappers', 'item', 'mapper_dict.json'), 'r')).keys())
    item_mapper = {v: k for k, v in zip(items, range(len(items)))}
    del items

    return item_mapper


@timing(logger)
def load_index():
    item_embeddings = np.load(os.path.join(script_dir, '..', 'models', 'item_embeddings.npy'))
    item_index = FaissIndex()
    logger.info("Starting training of faiss index")
    item_index.train(item_embeddings)
    logger.info("index trained. {}".format(item_index.is_trained()))
    if item_index.is_trained():
        logger.info("Index successfully trained")
        item_index.add(item_embeddings, np.arange(0, len(item_embeddings), 1))
        logger.info("embeddings added to index. Total number of encodings: {}".format(item_index.ntotal()))

        del item_embeddings

        return item_index
    else:
        raise RuntimeError("Error while training index")


item_mapper = load_mapper()
item_index = load_index()
map_item_id = np.vectorize(lambda x: item_mapper[x])


class SearchByVectors(object):

    """

    """

    def __init__(self):
        self.k = 5
        self.dimension = 40
        self.return_item_id = True

    # @statsd_client.timer("faiss_vector_search_latency")
    @timing(logger=logger)
    def on_post(self, req, resp):
        """Handles POST requests"""
        body = req.stream.read()
        request = json.loads(body.decode("utf-8"))
        logger.debug("Got the following request. {}".format(request))

        vectors = request.get('vectors')
        if not type(vectors) == list:
            raise falcon.HTTPError(falcon.HTTP_400, "vectors parameter should be a list of floats")
        vectors = np.array(vectors).astype('float32')
        if not vectors.shape[1] == self.dimension:
            raise falcon.HTTPError(falcon.HTTP_400, "vectors are to be of length {}".format(self.dimension))

        k = request.get('k', self.k)
        if not k < 50:
            raise falcon.HTTPError(falcon.HTTP_400, "k parameter (no. of neighbors) cannot be greater than 50")

        return_item_id = request.get('return_item_id', self.return_item_id)
        if not type(return_item_id) == bool:
            raise falcon.HTTPError(falcon.HTTP_400, "return_item_id has to be bool")

        _, neighbors = item_index.search(vectors, k)
        if len(neighbors) > 0 and return_item_id:
            neighbors = map_item_id(neighbors)
        neighbors_dict = {i: neighbor_array.tolist() for i, neighbor_array in enumerate(neighbors)
                          if len(neighbor_array) > 0}

        search_response = {}
        if len(neighbors_dict) > 0:
            search_response['es'] = 0
            search_response['result'] = neighbors_dict
        else:
            search_response['es'] = 1
            search_response['result'] = None

        resp.status = falcon.HTTP_200
        resp.body = json.dumps(search_response)


class SearchByIds(object):

    """

    """

    def __init__(self):
        self.k = 5
        self.return_item_id = True

    # @statsd_client.timer("faiss_id_search_latency")
    @timing(logger=logger)
    def on_post(self, req, resp):
        """Handles POST requests"""
        body = req.stream.read()
        request = json.loads(body.decode("utf-8"))
        logger.debug("Got the following request. {}".format(request))

        ids = request.get('ids')

        if not type(ids) == list:
            raise falcon.HTTPError(falcon.HTTP_400, "ids parameter should be a list of integers")

        if not all([type(id) == int for id in ids]):
            raise falcon.HTTPError(falcon.HTTP_400, "ids parameter should be a list of integers")
        ids = np.array(ids)

        k = request.get('k', self.k)
        if not k < 50:
            raise falcon.HTTPError(falcon.HTTP_400, "k parameter (no. of neighbors) cannot be greater than 50")

        return_item_id = request.get('return_item_id', self.return_item_id)
        if not type(return_item_id) == bool:
            raise falcon.HTTPError(falcon.HTTP_400, "return_item_id has to be bool")

        ids, (_, neighbors) = item_index.search_by_ids(ids, k, logger)

        if len(neighbors) > 0 and return_item_id:
            neighbors = map_item_id(neighbors)

        neighbors_dict = {id: neighbor_array.tolist() for id, neighbor_array in zip(ids, neighbors)
                          if len(neighbor_array) > 0}

        search_response = {}
        if len(neighbors_dict) > 0:
            search_response['es'] = 0
            search_response['result'] = neighbors_dict
        else:
            search_response['es'] = 1
            search_response['result'] = None

        resp.status = falcon.HTTP_200
        resp.body = json.dumps(search_response)


class UpdateVectors(object):
    """

    """

    def __init__(self):
        self.dimension = 40

    # @statsd_client.timer("faiss_update_latency")
    @timing(logger=logger)
    def on_post(self, req, resp):
        """Handles POST requests"""
        body = req.stream.read()
        request = json.loads(body.decode("utf-8"))
        logger.debug("Got update vectors request")

        ids = request.get('ids')
        if not type(ids) == list:
            raise falcon.HTTPError(falcon.HTTP_400, "ids parameter should be a list of integers")

        if not all([type(id) == int for id in ids]):
            raise falcon.HTTPError(falcon.HTTP_400, "ids parameter should be a list of integers")
        ids = np.array(ids).astype('int64')

        item_ids = request.get('item_ids')
        if not type(item_ids) == list:
            raise falcon.HTTPError(falcon.HTTP_400, "item_ids parameter should be a list of strings")

        if not all([type(id) == str for id in item_ids]):
            raise falcon.HTTPError(falcon.HTTP_400, "item_ids parameter should be a list of strings")

        vectors = request.get('vectors')
        if not type(vectors) == list:
            raise falcon.HTTPError(falcon.HTTP_400, "vectors parameter should be a list of floats")
        vectors = np.array(vectors).astype('float32')

        if not vectors.shape[1] == self.dimension:
            raise falcon.HTTPError(falcon.HTTP_400, "vectors are to be of length {}".format(self.dimension))

        if not len(ids) == len(vectors) == len(item_ids):
            raise falcon.HTTPError(falcon.HTTP_400, "ids, vectors, item_ids are to be of same length")

        try:
            item_index.replace(vectors, ids)
            new_mapper = {k: v for k, v in zip(ids, item_ids) if k not in item_mapper}
            item_mapper.update(new_mapper)
            insert_response = {}
            insert_response['es'] = 0
            insert_response['num_vectors_added'] = len(vectors)

        except Exception as e:
            logger.error("Unknown exception while adding vectors {}".format(e))
            insert_response = {}
            insert_response['es'] = 1
            insert_response['num_vectors_added'] = 0

        logger.info(insert_response)
        resp.status = falcon.HTTP_200
        resp.body = json.dumps(insert_response)


class ReloadVectors(object):
    """

    """

    def __init__(self):
        pass

    # @statsd_client.timer("faiss_reload_latency")
    @timing(logger=logger)
    def on_get(self, req, resp):
        global item_index, item_mapper
        """Handles GET requests"""
        logger.debug("Got reload vectors request")
        try:
            item_mapper = load_mapper()
            item_index = load_index()
            insert_response = {}
            insert_response['es'] = 0
            insert_response['total_vectors'] = item_index.ntotal()

        except Exception as e:
            logger.error("Unknown exception while reloading vectors {}".format(e))
            insert_response = {}
            insert_response['es'] = 1
            insert_response['total_vectors'] = 0

        logger.info(insert_response)
        resp.status = falcon.HTTP_200
        resp.body = json.dumps(insert_response)


searchByVectors = SearchByVectors()
searchByIds = SearchByIds()
updateVectors = UpdateVectors()
reloadVectors = ReloadVectors()

# API initialization
api = application = falcon.API()
api.add_route('/faiss_app/v1/search_by_vectors', searchByVectors)
api.add_route('/faiss_app/v1/search_by_ids', searchByIds)
api.add_route('/faiss_app/v1/update_vectors', updateVectors)
api.add_route('/faiss_app/v1/reload_vectors', reloadVectors)

if __name__ == '__main__':
    # print(recommend_items_by_user_id("62b2ea64f905f87426778f9bd6e98293", k=10))
    ids = np.array([17])
    print(item_index.search_by_ids(ids, 10, logger))
