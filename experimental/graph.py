import numpy as np
from typing import List
# from device import Device

class Graph:
    def __init__(self, adj, xadj, eweights, vweights, options=None):
        self.adj, self.xadj, self.eweights, self.vweights, self.options = adj, xadj, eweights, vweights, options

        self._check()
    
    def __len__(self):
        return len(self.xadj) - 1

    def _check(self):
        assert len(self.adj) == len(self.eweights)
        assert len(self.vweights) == len(self.xadj) - 1

        self.adj = np.array(self.adj)
        self.xadj = np.array(self.xadj)
        self.eweights = np.array(self.eweights)
        self.vweights = np.array(self.vweights)


def construct_graph(tensor_cores, comm_bws):
    """
        Assume the graph is connected. I.e. every value in communication bandwidth matrix is nonzero.
    """

    tensor_cores = np.around(tensor_cores, decimals=0).astype('int')
    comm_bws = np.around(comm_bws, decimals=0).astype('int')

    adj = np.array([[i for i in range(len(tensor_cores)) if i != j] for j in range(len(tensor_cores))]).flatten()
    xadj = [i for i in range(0, (comm_bws.shape[0] + 1) * comm_bws.shape[1], comm_bws.shape[1])]
    eweights = comm_bws.flatten()


    G = Graph(adj=adj, xadj=xadj, eweights=eweights, vweights=tensor_cores, options=None)

    return G


