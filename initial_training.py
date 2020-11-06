# hide
import os
import json
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

script_name = os.path.basename(__file__)
script_dir = os.path.dirname(os.path.abspath(__file__))

# relative imports
from model_step import Step
from cf_model import SimpleCF
from mapper import Mapper
from utils import constants
from utils.function_factory import *
from clients.redis_client import RedisClient

redis_client = RedisClient(host=constants.REDIS_HOST, port=constants.REDIS_PORT).get_client()

def main():
    # load the data
    user_item_df = pd.read_csv('data/0000_part_00', sep='|')
    user_item_df = user_item_df.sort_values('clicked_epoch', ascending=True).reset_index(drop=True)

    # getting popular items and setting redis key
    popular_items = user_item_df.groupby('sourceprodid', as_index=False).count().sort_values("uuid", ascending=False).sourceprodid.iloc[:10]
    redis_client.set("popular_items", {"items": list(popular_items)})

    # constructing user and item mappers
    user_mapper = Mapper(keys=list(user_item_df.uuid.unique()))
    item_groupby = user_item_df.groupby('sourceprodid', as_index=False).first()
    item_mapper = Mapper(keys=list(item_groupby['sourceprodid'].values),
                         attributes=item_groupby[constants.ITEM_ATTRIBUTES].values.tolist())

    user_item_df['user_code'] = user_item_df['uuid'].map(user_mapper.mapper_dict)
    user_item_df['item_code'] = user_item_df['sourceprodid'].map(item_mapper.mapper_dict)
    user_item_df['item_code'] = user_item_df.item_code.apply(lambda x: x[0])

    n_users = len(user_mapper)
    n_items = len(item_mapper)

    user_item_df['target'] = 1.0
    event_id_mapper = {'pageView': 1, 'addToCart': 2, 'buy': 3}
    user_item_df['event_code'] = user_item_df.userevent.map(event_id_mapper)

    cf_model = SimpleCF(n_users, n_items, embedding_length=40, init=torch.nn.init.normal_, mean=0., std=.1)
    objective = weighted_mse_loss
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Step(cf_model, objective, device=device, mode='bulk')

    features = ['user_code', 'item_code', 'event_code']
    target = ['target']
    # user_item_df = user_item_df
    data_set = TensorDataset(torch.tensor(user_item_df[features].values),
                             torch.tensor(user_item_df[target].values))
    data_loader = DataLoader(data_set, batch_size=constants.INITIAL_BATCH_SIZE, shuffle=False)

    # training
    for epoch in range(constants.INITIAL_TRAINING_EPOCHS):
        print("Epoch: {}".format(epoch+1))
        with tqdm(total=len(data_loader)) as pbar:
            for _, (features, target) in enumerate(data_loader):
                model.batch_fit(features, target)

                pbar.update(1)
    print("Done with training")

    # dump the mappers
    user_mapper.save(os.path.join(script_dir, 'mappers', 'user'))
    item_mapper.save(os.path.join(script_dir, 'mappers', 'item'))

    # dump the model weights
    model.save(os.path.join(script_dir, 'models', 'model_weights.pth'))

    for embedding, num_embeddings in [('user', len(user_mapper)), ('item', len(item_mapper))]:
        embeddings = np.array(model.model.get_embeddings(embedding)).astype('float32')[:num_embeddings]
        np.save(os.path.join(script_dir, 'models', embedding + '_embeddings.npy'), embeddings)

    # dump the model metadata
    model_metadata = {}
    model_metadata['num_users'] = model.model.user_embeddings.weight.data.shape[0]
    model_metadata['num_items'] = model.model.item_embeddings.weight.data.shape[0]
    with open(os.path.join(script_dir, 'mappers', 'metadata.json'), 'w') as f:
        f.write(json.dumps(model_metadata))

    print("Done with dumping of model and mappers.")


if __name__ == '__main__':
    print("Initial training of cf model to avoid cold starting problem")
    main()
    print("Spanning sync vectors script")
    subprocess.call("conda activate msd_recommendation_engine; python {}/sync_vectors.py".format(script_dir))
