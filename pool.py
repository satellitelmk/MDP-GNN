from os import device_encoding
from models.models import MLP,glorot
import torch
import numpy as np
import random
from torch.nn import Parameter
from data.load_data import Graph
from collections import defaultdict

class graph_pool():
    def __init__(self,args,dataloader=None,device=None,parameters = True,addr = None):
        super(graph_pool, self).__init__()
        self.syn_graphs = []

        if addr != None:
            graphs  = np.load(addr,allow_pickle=True)['arr_0']
            for graph in graphs:
                node = torch.FloatTensor(graph[0]).to(device)
                edge = torch.from_numpy(graph[1].astype(np.long)).to(device)
                self.syn_graphs.append(Graph(node,edge,True))
        else:
            for _ in range(args.syn_graph_num):
                


                ################# degree base
                
                dataset = args.train_dataset[_ % len(args.train_dataset)]
                subgraph,matrix = dataloader.sampling(dataset,args.subgraph_scale)



                edge_index = subgraph.edge_index
                edge = torch.from_numpy(edge_index).to(device)

                if parameters:
                    node = Parameter(torch.FloatTensor(args.subgraph_scale, args.share_dims).to(device))
                    glorot(node)
                else: node = torch.FloatTensor(subgraph.x).to(device)


                degrees = defaultdict(lambda: set())
                for i in range(edge_index.shape[1]):
                    x,y = edge_index[0,i],edge_index[1,i]
                    degrees[x].add(y)
                    degrees[y].add(x)


                labels = degrees
                
                g = Graph(node,edge,True)
                g.labels = labels
                self.syn_graphs.append(g)
                









    def detach(self):
        results = []
        for graph in self.syn_graphs:
            x = graph.x.detach()
            edge_index = graph.edge_index
            results.append(Graph(x,edge_index,True))
        return results


    def transformation(self,mlps):
        results = []
        for i in range(len(mlps)):
            graph = self.syn_graphs[i]
            mlp = mlps[i]
            x = mlp(graph.x).detach()
            edge_index = graph.edge_index
            results.append(Graph(x,edge_index,True))
        return results












