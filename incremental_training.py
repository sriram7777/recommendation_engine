# hide
import os
import json
import subprocess
import numpy as np
import pandas as pd
import requests
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

script_name = os.path.basename(__file__)
script_dir = os.path.dirname(os.path.abspath(__file__))

# relative imports
from model_step import Step
from cf_model import SimpleCF
from utils import constants
# from utils.statsd_events import *
from clients.logger_client import LoggerClient
from mapper import Mapper
from utils.function_factory import *


logger = LoggerClient.set_logger("logs/incremental_training.log")

event_id_mapper = {'pageView': 1, 'addToCart': 2, 'buy': 3}


@timing(logger)
def dump_model():
    # dump the mappers
    user_mapper.save(os.path.join(script_dir, 'mappers', 'user'))
    item_mapper.save(os.path.join(script_dir, 'mappers', 'item'))

    # dump the embeddings separately
    for embedding, num_embeddings in [('user', len(user_mapper)), ('item', len(item_mapper))]:
        embeddings = np.array(model.model.get_embeddings(embedding)).astype('float32')[:num_embeddings]
        np.save(os.path.join(script_dir, 'models', embedding + '_embeddings.npy'), embeddings)

    # dump the model weights for reference
    model.save(os.path.join(script_dir, 'models', 'model_weights.pth'))

    # dump the model metadata
    with open(os.path.join(script_dir, 'mappers', 'metadata.json'), 'w') as f:
        f.write(json.dumps(metadata))

    logger.info("Spanning sync vectors script..")
    subprocess.call("conda activate msd_recommendation_engine; python sync_vectors.py")


def inference(features, k=10):
    features = np.array(features)
    user_indices = features[:, 0]
    item_indices = features[:, 1]
    user_embeddings = np.take(model.model.get_embeddings("user"), user_indices, axis=0).tolist()

    payload = json.dumps({'k': k,
                          'return_item_id': False,
                          'vectors': user_embeddings})

    url = "http://" + constants.FAISS_APP_HOST + ":" + constants.FAISS_APP_PORT + "/faiss_app/v1/search_by_vectors"
    res = requests.post(url, payload)
    if res.status_code == 200:
        response_dict = res.json()
        if response_dict['es'] == 0:
            recommended_items_dict = response_dict["result"]
            recommended_items_dict = {user_indices[int(k)]: v for k, v in recommended_items_dict.items()}

            recall_list = []
            precision_list = []
            for k, v in recommended_items_dict.items():
                recall_list.append(recall_at_k(predictions=v, targets=item_indices[k], k=k))
                precision_list.append(precision_at_k(predictions=v, targets=item_indices[k], k=k))

            logger.info("precision@10: {}".format(np.mean(precision_list)))
            logger.info("recall@10: {}".format(np.mean(recall_list)))

            #send_value("recall@10", np.mean(recall_list))
            #send_value("precision@10", np.mean(precision_list))


# @statsd_client.timer("training_time")
@timing(logger)
def train_samples(features):
    data_set = TensorDataset(torch.tensor(features),
                             torch.tensor(np.ones(len(features))))
    data_loader = DataLoader(data_set, batch_size=constants.INCREMENTAL_BATCH_SIZE, shuffle=False)
    # training
    for epoch in range(constants.INCREMENTAL_TRAINING_EPOCHS):
        logger.info("Epoch: {}".format(epoch+1))
        for _, (features, target) in enumerate(data_loader):
            model.batch_fit(features, target)


# simulate the Stream
def run_incremental_training(data_path):
    batch_data = []
    inference_data = []
    run = False
    seen_user_set = set()
    seen_item_set = set()
    data_files = os.listdir(data_path)
    for file in data_files[:1]:
        stream_df = pd.read_csv(os.path.join(data_path, file), sep='|')
        stream_df = stream_df.sort_values('clicked_epoch').reset_index(drop=True)
        stream_df.loc[:, 'clicked_time'] = pd.to_datetime(stream_df.clicked_epoch*10**9)
        current_time = stream_df.clicked_time.iloc[0]
        logger.debug("current time: {}".format(current_time))
        for i, row in stream_df.iterrows():
            if run is True and (row.clicked_time - current_time).total_seconds() > \
                    constants.MODEL_DUMP_FREQUENCY_IN_SECONDS:
                logger.info("Periodic dumping of model ")
                metadata['updated_users'] = list(seen_user_set)
                metadata['updated_items'] = list(seen_item_set)
                dump_model()
                run = False
                seen_item_set = set()
                seen_user_set = set()

            user, item, event = row.uuid, row.sourceprodid, row.userevent
            known_item, known_user = True, True
            if not user_mapper.check_key(user):
                user_mapper.add(user)
                if len(user_mapper) > metadata['num_users']:
                    logger.info("User embeddings overflow. Adding new embeddings")
                    metadata['num_users'] = model.model.add_embeddings('user')
                    model.optimizer = model.initialize_optimizer()
                known_user = False
            if not item_mapper.check_key(item):
                attributes = [row[attribute] for attribute in constants.ITEM_ATTRIBUTES]
                user_mapper.add(item, attributes)
                if len(item_mapper) > metadata['num_items']:
                    logger.info("Item embeddings overflow. Adding new embeddings")
                    metadata['num_items'] = model.model.add_embeddings('item')
                    model.optimizer = model.initialize_optimizer()
                known_item = False

            user_code = user_mapper.retrieve_index(user)
            item_code = item_mapper.retrieve_index(item)[0]
            event_code = event_id_mapper[event]

            seen_user_set.add(user)
            seen_item_set.add(item)

            # append to inference_data
            if known_user and known_item:
                inference_data.append([user_code, item_code])

            batch_data.append([user_code, item_code, event_code])
            if len(batch_data) == constants.INCREMENTAL_SAMPLE_SIZE:
                logger.debug("last record time: {}".format(row.clicked_time))
                logger.info("batch inferencing starts")
                # todo: batch inferencing using faiss_server search and local if not available in faiss_server

                inference_data = []
                logger.info("batch training starts")
                train_samples(batch_data)
                run = True
                batch_data = []


if __name__ == '__main__':
    try:
        user_mapper = Mapper()
        user_mapper.load(os.path.join(script_dir, 'mappers', 'user'))
        item_mapper = Mapper()
        item_mapper.load(os.path.join(script_dir, 'mappers', 'item'))
        metadata = json.load(open(os.path.join(script_dir, 'mappers', 'metadata.json'), 'r'))
        logger.info("User and Item mappers loaded successfully")

        # loading model
        cf_model = SimpleCF(metadata['num_users'], metadata['num_items'], embedding_length=constants.EMBEDDING_LENGTH,
                            init=torch.nn.init.normal_, mean=0., std=.1)
        objective = weighted_mse_loss
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = Step(cf_model, objective, device=device, mode='incremental')
        model.load(os.path.join(script_dir, 'models', 'model_weights.pth'))
        logger.info("CF Model loaded successfully")

        # incremental training
        run_incremental_training(os.path.join(script_dir, 'data'))

    except Exception as e:
        logger.error("Exception occurred. {}".format(e))
