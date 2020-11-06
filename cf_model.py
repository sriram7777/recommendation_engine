import torch
import torch.nn as nn


# Cell
class SimpleCF(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_length: int = 16,
                 user_embeddings: torch.tensor = None, item_embeddings: torch.tensor = None,
                 init: torch.nn.init = torch.nn.init.normal_, binary: bool = False, **kwargs):
        super().__init__()
        self.binary = binary
        self.init = init

        self.kwargs = kwargs
        self.user_embeddings = self._create_embedding(n_users, embedding_length,
                                                      user_embeddings, init, **kwargs)
        self.item_embeddings = self._create_embedding(n_items, embedding_length,
                                                      item_embeddings, init, **kwargs)
        self.linear = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, u: torch.tensor, i: torch.tensor) -> torch.tensor:
        user_embedding = self.user_embeddings(u)
        user_embedding = user_embedding[:, None, :]
        item_embedding = self.item_embeddings(i)
        item_embedding = item_embedding[:, None, :]
        rating = torch.matmul(user_embedding, item_embedding.transpose(1, 2))
        rating = self.linear(rating)  # adding a simple neuron for including bias
        rating = self.sigmoid(rating)

        return rating

    def _create_embedding(self, n_items, embedding_length, weights, init, **kwargs):
        embedding = nn.Embedding(n_items, embedding_length)
        init(embedding.weight.data, **kwargs)

        if weights is not None:
            embedding.load_state_dict({'weight': weights})

        return embedding

    def add_embeddings(self, embedding, ratio=0.1):
        assert embedding == 'user' or embedding == 'item', "embedding has to be user or item"
        embedding_layer = self.user_embeddings if embedding == 'user' else self.item_embeddings
        new_rows = int(ratio * embedding_layer.weight.shape[0])
        new_embedding = torch.FloatTensor(new_rows, embedding_layer.weight.shape[1])
        new_embedding.requires_grad = True
        self.init(new_embedding, **self.kwargs)
        e = torch.cat([embedding_layer.weight.data, new_embedding])
        if embedding == 'user':
            self.user_embeddings = nn.Embedding(e.shape[0], e.shape[1])
            self.user_embeddings.weight.data = e

        else:
            self.item_embeddings = nn.Embedding(e.shape[0], e.shape[1])
            self.item_embeddings.weight.data = e

        return e.shape[0]

    def get_embeddings(self, embedding):
        assert embedding == 'user' or embedding == 'item', "embedding has to be user or item"
        if embedding == 'user':
            return self.user_embeddings.weight.data
        else:
            return self.item_embeddings.weight.data
