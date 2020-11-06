import torch
from typing import List
from functools import wraps
from time import time


def weighted_mse_loss(input_: torch.tensor, target: torch.tensor, weight: torch.tensor):
    return torch.mean(weight * (input_ - target) ** 2)


def event_weighing_function(x: torch.tensor) -> torch.tensor:
    x[x == 1.] = 0.5  # page view
    x[x == 2.] = 0.7  # add to cart
    x[x == 3.] = 1.0  # buy

    return x.float()


def recall_at_k(predictions: List[int], targets: List[int], k: int = 10) -> float:
    """Computes `Recall@k` from the given predictions and targets sets."""
    predictions_set = set(predictions[:k])
    targets_set = set(targets)
    result = len(targets_set & predictions_set) / float(len(targets_set))
    return result


def precision_at_k(predictions: List[int], targets: List[int], k: int = 10) -> float:
    """Computes `Precision@k` from the given predictions and targets sets."""
    predictions_set = set(predictions[:k])
    targets_set = set(targets)
    result = len(targets_set & predictions_set) / float(len(predictions_set))
    return result


def timing(logger):
    def wrap(f):
        def wrapper(*args, **kwargs):
            start = time()
            result = f(*args, **kwargs)
            logger.info('Function {0} - Elapsed time: {1}'.format(f.__name__, time()-start))
            return result
        return wrapper
    return wrap


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]
