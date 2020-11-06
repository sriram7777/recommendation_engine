import os
import faiss
import numpy as np
from faiss import downcast_index

N_LIST = 100
N_PROBE = 10
VECTOR_DIMENSION = 40

script_name = os.path.basename(__file__)
script_dir = os.path.dirname(os.path.abspath(__file__))


class FaissIndex:
    def __init__(self, dim=VECTOR_DIMENSION):
        # if isfile(save_path):
        #     self._index = faiss.read_index(save_path)
        # else:
        quantizer = faiss.IndexFlatL2(dim)
        self._index = faiss.IndexIVFFlat(quantizer, dim, N_LIST, faiss.METRIC_L2)
        # self._index = faiss.IndexIDMap(quantizer)
        # self.set_nprobe(N_PROBE)

    def train(self, xb):
        self._index.train(xb)

    def is_trained(self):

        return self._index.is_trained

    def replace(self, xb, ids):
        self.remove(ids)
        return self._index.add_with_ids(xb, ids)

    def add(self, xb, ids):
        self._index.add_with_ids(xb, ids)
        downcast_index(self._index).make_direct_map()

    def search(self, xq, k=10):
        return self._index.search(xq, k)

    def search_by_id(self, id, k=10, logger=None):
        try:
            x = self._index.reconstruct(id)
            xq = np.expand_dims(x, axis=0)
        except Exception as e:
            logger.warning("id {} not found in the index. {}".format(id, e))
            return [], []

        return self.search(xq, k)

    def search_by_ids(self, ids, k=10, logger=None):
        xq = []
        qualified_ids = []
        for id in ids:
            try:
                x = self._index.reconstruct(int(id))
                xq.append(x)
                qualified_ids.append(int(id))
            except RuntimeError as e:
                logger.warning("id {} not found in the index. {}".format(id, e))
        xq = np.array(xq)
        if qualified_ids:
            return qualified_ids, self.search(xq, k)
        else:
            return [], [], []

    def ntotal(self):
        return self._index.ntotal

    def remove(self, ids):
        return self._index.remove_ids(ids)

    def restore(self, filepath):
        pre_index = self._index
        self._index = faiss.read_index(filepath)
        if pre_index:
            pre_index.reset()

    def save(self, filepath):
        if self.ntotal() > 0:
            faiss.write_index(self._index, filepath)

    def set_nprobe(self, nprobe):
        faiss.ParameterSpace().set_index_parameter(self._index, "nprobe", nprobe)


if __name__ == '__main__':
    item_embeddings = np.load(os.path.join(script_dir, '..', 'models', 'item_embeddings.npy'))

    item_index = FaissIndex()
    item_index.train(item_embeddings)
    if item_index.is_trained():
        item_index.add(item_embeddings, np.arange(0, len(item_embeddings), 1))

    del item_embeddings

    xq = np.array([[-0.007227, -0.008026, -0.002986, -0.000763, -0.002926, -0.003281, -0.002636, -0.000781, 0.003644, -0.000507,
                    -0.002085,  0.001349,  0.006111, -0.004744, 0.00172, 0.006821, -0.002909, -0.001771, 0.002742, 0.000582,
                    -0.001083, -0.005921, -0.011519, -0.012683, -0.003193, -0.000233, 0.003307, -0.006075, 0.003365, 0.0018,
                    0.000095, -0.005648,  0.002228, -0.001124, -0.000964,
                    -0.006769, -0.000565, -0.000927, 0.001521, 0.007124]]).astype('float32')
    print(item_index.search(xq))
