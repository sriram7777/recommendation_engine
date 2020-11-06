import torch
from torch.optim import SGD
from typing import Callable


# Cell
class Step:
    """Incremental and batch training of recommender_server systems."""

    def __init__(self, model: torch.nn.Module, objective: Callable,
                 weighing_func: Callable = lambda x: torch.tensor(1),
                 mode: str = 'bulk', device: str = 'cpu'):
        assert mode in ['bulk', 'incremental'], "mode has to be bulk or incremental"
        self.model = model.to(device)
        self.objective = objective
        self.weighing_func = weighing_func
        self.mode = mode
        self.optimizer = self.initialize_optimizer()
        self.device = device
        self.model.train()

        # check if the user has provided user and item embeddings
        assert self.model.user_embeddings, 'User embedding matrix could not be found.'
        assert self.model.item_embeddings, 'Item embedding matrix could not be found.'

    @property
    def user_embeddings(self):
        return self.model.user_embeddings

    @property
    def item_embeddings(self):
        return self.model.item_embeddings

    def initialize_optimizer(self):
        if self.mode == 'bulk':
            return SGD(self.model.parameters(), lr=0.001, weight_decay=0.1)  # l2 regulariser included as weight decay
        else:
            # not including the linear layer while training incrementally
            return SGD([param for name, param in self.model.state_dict().items()
                             if 'embeddings' in name], lr=0.001, weight_decay=0.1)

    def batch_fit(self, features: torch.Tensor, targets: torch.Tensor):
        """Trains the model on a batch of user-item interactions."""
        self.model.train()
        users = features[:, 0].to(self.device)
        items = features[:, 1].to(self.device)
        events = features[:, 2].to(self.device)

        predictions = self.model(users, items)
        weights = self.weighing_func(events)

        loss = self.objective(predictions, targets, weights)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

    def step(self, user: torch.tensor, item: torch.tensor,
             rating: torch.tensor = None, preference: torch.tensor = None):
        """Trains the model incrementally."""
        user = user.to(self.device)
        item = item.to(self.device)
        events = rating.to(self.device)
        pref = preference.to(self.device)

        pred = self.model(user, item)
        weights = self.weighing_func(events)

        loss = self.objective(pred, pref, weights)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

    def predict(self, user: torch.tensor, k: int = 10) -> torch.tensor:
        """Recommends the top-k items to a specific user."""
        self.model.eval()
        user = user.to(self.device)
        user_embedding = self.user_embeddings(user)
        item_embeddings = self.item_embeddings.weight
        score = item_embeddings @ user_embedding.transpose(0, 1)
        predictions = score.squeeze().argsort()[-k:]
        return predictions.cpu()

    def save(self, path: str):
        """Saves the model parameters to the given path."""
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        """Loads the model parameters from a given path."""
        self.model.load_state_dict(torch.load(path))
