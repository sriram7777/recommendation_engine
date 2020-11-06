import os
from typing import List
import json


class Mapper:

    def __init__(self, keys: List = None, attributes: List = None):
        if keys is None and attributes is None:
            self._mapper_dict = {}
        elif attributes is None:
            self._mapper_dict = {k: v for k, v in zip(keys, range(len(keys)))}
        else:
            assert len(keys) == len(attributes), "keys and attributes are to be of same length"
            self._mapper_dict = {k: [v1] + v2 for k, v1, v2 in zip(keys, range(len(keys)), attributes)}

    def __len__(self):
        return len(self._mapper_dict)

    @property
    def mapper_dict(self):
        return self._mapper_dict

    def add(self, key: str, attributes: List = None):
        if attributes is None:
            self._mapper_dict[key] = len(self._mapper_dict)
        else:
            self._mapper_dict[key] = [len(self._mapper_dict)] + attributes

    def check_key(self, key: str):

        return key in self._mapper_dict

    def retrieve_index(self, key: str):

        return self._mapper_dict[key]

    def save(self, path):
        with open(os.path.join(path, 'mapper_dict.json'), 'w') as f:
            f.write(json.dumps(self._mapper_dict))

    def load(self, path):
        assert os.path.exists(os.path.join(path, 'mapper_dict.json')), "index_mapper.json file not found"
        self._mapper_dict = json.load(open(os.path.join(path, 'mapper_dict.json'), 'r'))


