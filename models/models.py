
import torch
import math
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, GCNConv, GAE, VGAE,SAGEConv,TransformerConv
from torch.nn import Parameter,BatchNorm1d,Dropout
from torch_scatter import scatter_add,scatter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops,add_self_loops
from torch.distributions import Normal
from torch import dropout, nn
from .layers import *
import torch.nn.functional as F
from utils import uniform
from collections import OrderedDict
from torch_geometric.nn import global_mean_pool
from models.autoencoder import negative_sampling
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from torch_geometric.data import Batch, Data
from typing import Union, Tuple, Optional
from torch_geometric.typing import PairTensor, Adj, OptTensor
import numpy as np
import torch_geometric



def softplus(x):
  return torch.nn.functional.softplus(x, beta=100)

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

class Encoder(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.args = args
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=False)
        if args.model in ['GAE']:
            self.conv2 = GCNConv(2 * out_channels, out_channels, cached=False)
        elif args.model in ['VGAE']:
            self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=False)
            self.conv_logvar = GCNConv(
                2 * out_channels, out_channels, cached=False)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        if self.args.model in ['GAE']:
            return self.conv2(x, edge_index)
        elif self.args.model in ['VGAE']:
            return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)



class MetaMLPEncoder(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(MetaMLPEncoder, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(in_channels, 2 * out_channels, bias=True)
        if args.model in ['GAE']:
            self.fc_mu = nn.Linear(2 * out_channels, out_channels, bias=True)
        elif args.model in ['VGAE']:
            self.fc_mu = nn.Linear(2 * out_channels, out_channels, bias=True)
            self.fc_logvar = nn.Linear(2 * out_channels, out_channels, bias=True)

    def forward(self, x, edge_index, weights, inner_loop=True):
        x = F.relu(F.linear(x, weights['encoder.fc1.weight'],weights['encoder.fc1.bias']))
        if self.args.model in ['GAE']:
            return F.relu(F.linear(x, weights['encoder.fc_mu.weight'],weights['encoder.fc_mu.bias']))
        elif self.args.model in ['VGAE']:
            return F.relu(F.linear(x,weights['encoder.fc_mu.weight'],\
                    weights['encoder.fc_mu.bias'])),F.relu(F.linear(x,\
                    weights['encoder.fc_logvar.weight'],weights['encoder.fc_logvar.bias']))

class MLPEncoder(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(MLPEncoder, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(in_channels, 2 * out_channels, bias=True)
        self.fc2 = nn.Linear(2 * out_channels, out_channels, bias=True)

    def forward(self, x, edge_index):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x





class TransEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransEncoder, self).__init__()
        self.conv1 = TransformerConv(in_channels, out_channels)
        self.conv2 = TransformerConv(out_channels, out_channels)


    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


class MetaEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MetaEncoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, out_channels)
        self.conv2 = SAGEConv(out_channels, out_channels)


    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)



class TransformerEncoder(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(TransformerEncoder, self).__init__()
        self.args = args
        self.conv1 = TransformerConv(in_channels, 2*out_channels)
        self.conv2 = TransformerConv(2*out_channels, out_channels)
        self.reset_parameters()


    def forward(self, x, edge_index, weights, inner_loop=True,edge_weight = None):

        if edge_weight is None:
            x = F.relu(self.conv1(x, edge_index))
            return self.conv2(x, edge_index)
        else:
            x = F.relu(self.conv1(x,edge_index = None,edge_weight= edge_weight))
            return self.conv2(x, edge_index = None,edge_weight=edge_weight)


    def reset_parameters(self):
        for tensor in self.parameters():
            if len(tensor.size())==1:
                bound = 1.0 / math.sqrt(tensor.size()[0])
                tensor.data.uniform_(-bound, bound)
            else:
                stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
                tensor.data.uniform_(-stdv, stdv)




class MetaEncoder2(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(MetaEncoder, self).__init__()
        self.args = args
        self.conv1 = MetaGCNConv(in_channels, 2 * out_channels, cached=False)
        if args.model in ['GAE']:
            self.conv2 = MetaGCNConv(2 * out_channels, out_channels, cached=False)
        elif args.model in ['VGAE']:
            self.conv_mu = MetaGCNConv(2 * out_channels, out_channels, cached=False)
            self.conv_logvar = MetaGCNConv(
                2 * out_channels, out_channels, cached=False)

    def forward(self, x, edge_index, weights, inner_loop=True):
        x = F.relu(self.conv1(x, edge_index, \
                weights['encoder.conv1.weight'],weights['encoder.conv1.bias']))
        if self.args.model in ['GAE']:
            return self.conv2(x, edge_index,\
                    weights['encoder.conv2.weight'],weights['encoder.conv2.bias'])
        elif self.args.model in ['VGAE']:
            return self.conv_mu(x,edge_index,weights['encoder.conv_mu.weight'],\
                    weights['encoder.conv_mu.bias']),\
                self.conv_logvar(x,edge_index,weights['encoder.conv_logvar.weight'],\
                weights['encoder.conv_logvar.bias'])






class Net(torch.nn.Module):
    def __init__(self,train_dataset):
        super(Net, self).__init__()
        self.conv1 = GATConv(train_dataset.num_features, 256, heads=4)
        self.lin1 = torch.nn.Linear(train_dataset.num_features, 4 * 256)
        self.conv2 = GATConv(4 * 256, 256, heads=4)
        self.lin2 = torch.nn.Linear(4 * 256, 4 * 256)
        self.conv3 = GATConv(
            4 * 256, train_dataset.num_classes, heads=6, concat=False)
        self.lin3 = torch.nn.Linear(4 * 256, train_dataset.num_classes)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        x = self.conv3(x, edge_index) + self.lin3(x)
        return x




class WDiscriminator(torch.nn.Module):
    def __init__(self, hidden_size, hidden_size2=512):
        super(WDiscriminator, self).__init__()
        self.hidden = MyGCNConv(hidden_size, hidden_size2, cached=False)
        self.hidden2 = torch.nn.Linear(hidden_size2, hidden_size2)
        self.output = torch.nn.Linear(hidden_size2, 1)

    def forward(self, input_embd,edge_index):
        return self.output(F.leaky_relu(self.hidden2(F.leaky_relu(self.hidden(input_embd,edge_index), 0.2, inplace=True)), 0.2, inplace=True))

    def reset_parameters(self):
        #print('reset')
        glorot(self.hidden.weight)
        glorot(self.hidden2.weight)
        glorot(self.output.weight)


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_out)
        self.Dropout = nn.Dropout(p=0.0)
        glorot(self.fc1.weight)
        glorot(self.fc2.weight)


    def forward(self, x):

        return (self.fc2(F.relu(self.Dropout(self.fc1(x)))))

    def reset_parameters(self):
        print('reset')
        glorot(self.fc1.weight)
        glorot(self.fc2.weight)













class matchGAT3(torch.nn.Module):

    def __init__(self,in_channels,out_channels):
        super(matchGAT3, self).__init__()
        self.in_channels = in_channels
        self.out_channels= out_channels
        self.conv = TransEncoder(in_channels,out_channels)

        self.weight_l = nn.Sequential(nn.Linear(out_channels, out_channels, bias=True),nn.Linear(out_channels, 1, bias=True) )
        self.weight_r = nn.Sequential(nn.Linear(out_channels, out_channels, bias=True), nn.Linear(out_channels, 1, bias=True))

        self.reset_parameters()

    def reset_parameters(self):
        for tensor in self.parameters():
            if len(tensor.size())==1:
                bound = 1.0 / math.sqrt(tensor.size()[0])
                tensor.data.uniform_(-bound, bound)
            else:
                stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
                tensor.data.uniform_(-stdv, stdv)


    def forward(self,graphs):

        Xs = []
        for key in graphs.keys():
            graph = graphs[key]
            Xs.append(self.conv(graph.x.detach(), graph.edge_index))

        features = torch.cat(Xs,dim = 0)
        alpha_l = self.weight_l(features)
        alpha_r = self.weight_r(features)


        alpha = alpha_l+alpha_r.t()
        alpha = (alpha+alpha.t())/2
        adj = F.sigmoid(alpha)

        return adj
    
    def get_final_adj(self,adj,graphs,threshold):

        cum = 0
        for key in graphs.keys():
            graph = graphs[key]

            adj[cum:cum+graph.x.shape[0],cum:cum+graph.x.shape[0]] = 0.0
            adj[graph.edge_index[0]+cum,graph.edge_index[1]+cum] = 1.0
            cum+=graph.x.shape[0]

        adj = torch.where(adj>threshold,adj,torch.tensor(0, dtype=torch.float).to(adj))
        print('adj.sum()',adj.sum())
        return adj.nonzero().t().detach()

        

        



class matchGAT2(torch.nn.Module):

    def __init__(self,in_channels,out_channels):
        super(matchGAT2, self).__init__()
        self.in_channels = in_channels
        self.out_channels= out_channels
        self.conv = MetaEncoder(in_channels,out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        for tensor in self.parameters():
            if len(tensor.size())==1:
                bound = 1.0 / math.sqrt(tensor.size()[0])
                tensor.data.uniform_(-bound, bound)
            else:
                stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
                tensor.data.uniform_(-stdv, stdv)


    def forward(self,graphs):

        Xs = []
        for key in graphs.keys():
            graph = graphs[key]
            Xs.append(self.conv(graph.x.detach(), graph.edge_index))

        features = torch.cat(Xs,dim = 0)
        alpha =  features@features.t()  
        adj = F.sigmoid(alpha)
        print(adj)

        return adj
    
    def get_final_adj(self,adj,graphs,threshold):

        cum = 0
        for key in graphs.keys():
            graph = graphs[key]

            adj[cum:cum+graph.x.shape[0],cum:cum+graph.x.shape[0]] = 0.0
            adj[graph.edge_index[0]+cum,graph.edge_index[1]+cum] = 1.0
            cum+=graph.x.shape[0]

        adj = torch.where(adj>threshold,adj,torch.tensor(0, dtype=torch.float).to(adj))
        print('adj.sum()',adj.sum())
        return adj.nonzero().t().detach()

        










class ConnectMatch_mlp(torch.nn.Module):
    def __init__(self,args,node_dim,device,proto_num=256):
        super(ConnectMatch_mlp, self).__init__()
        self.node_dim =node_dim
        self.proto_num = proto_num
        self.device = device
        self.MLPs =torch.nn.ModuleDict()
        for key in args.data_dims.keys():
            self.MLPs[key] = MLP( args.data_dims[key], args.hidden_dims, self.node_dim).to(device)
        self.super_nodes = None
        self.reset_parameters()

    def reset_parameters(self):
        for tensor in self.parameters():
            if len(tensor.size())==1:
                bound = 1.0 / math.sqrt(tensor.size()[0])
                tensor.data.uniform_(-bound, bound)
            else:
                stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
                tensor.data.uniform_(-stdv, stdv)

    def set_super_nodes(self,graphs):
        num = self.proto_num//4
        self.super_nodes_dict = {}
        for key in self.MLPs.keys():
            x = graphs[key].x[np.random.choice(graphs[key].x.shape[0],num)]
            self.super_nodes_dict[key] = torch.from_numpy(x).to(torch.float).to(self.device)



    def forward(self,graphs):

        super_nodes = []
        for key in self.super_nodes_dict.keys():
            super_nodes.append(self.MLPs[key](self.super_nodes_dict[key]))
        super_nodes= torch.cat(super_nodes,0)

                


        cum = 0
        tmp = np.sum([graphs[key].x.shape[0] for key in graphs.keys()])
        adj = torch.tensor(np.zeros((tmp,tmp)))
        for key in graphs.keys():
            graph = graphs[key]
            adj[graph.edge_index[0]+cum,graph.edge_index[1]+cum] = 1.0
            cum+=graph.x.shape[0]

        features = torch.cat([graphs[name].x for name in graphs.keys()],dim = 0)
        adj = adj.to(features)
        down = F.sigmoid(super_nodes@features.t())
        features = torch.cat([features,super_nodes],dim=0)
        right = F.sigmoid(features@super_nodes.t())

        adj = torch.cat([adj,down],dim=0)
        adj = torch.cat([adj,right],dim=1)
        print(adj)

        self.super_nodes = super_nodes

        return adj
    

    def get_final_adj(self,adj,threshold):

        
        return torch.where(adj>threshold,1,0).nonzero().t().detach()

















class ConnectMatch(torch.nn.Module):
    def __init__(self,node_dim,proto_num=256):
        super(ConnectMatch, self).__init__()
        self.node_dim =node_dim
        self.proto_num = proto_num


        self.super_nodes = torch.nn.Parameter(torch.randn(self.proto_num,self.node_dim,dtype=torch.float))
        self.reset_parameters()

    def reset_parameters(self):
        for tensor in self.parameters():
            if len(tensor.size())==1:
                bound = 1.0 / math.sqrt(tensor.size()[0])
                tensor.data.uniform_(-bound, bound)
            else:
                stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
                tensor.data.uniform_(-stdv, stdv)


    def forward(self,graphs):


        cum = 0
        tmp = np.sum([graphs[key].x.shape[0] for key in graphs.keys()])
        adj = torch.tensor(np.zeros((tmp,tmp)))
        for key in graphs.keys():
            graph = graphs[key]
            adj[graph.edge_index[0]+cum,graph.edge_index[1]+cum] = 1.0
            cum+=graph.x.shape[0]

        features = torch.cat([graphs[name].x for name in graphs.keys()],dim = 0)
        adj = adj.to(features)
        print(self.super_nodes.dtype,features.dtype)
        down = F.sigmoid(self.super_nodes@features.t())
        features = torch.cat([features,self.super_nodes],dim=0)
        right = F.sigmoid(features@self.super_nodes.t())

        adj = torch.cat([adj,down],dim=0)
        adj = torch.cat([adj,right],dim=1)
        print(adj)

        return adj
    

    def get_final_adj(self,adj,threshold):



        adj = torch.where(adj>threshold,1,0)


        print('rrrrrr',adj[:4096,:4096].sum())
        print('rrrrrr',adj[-self.proto_num:,:4096].sum())
        print('rrrrrr',adj[:4096,-self.proto_num:].sum())
        print('rrrrrr',adj[-self.proto_num:,-self.proto_num:].sum())
        return adj.nonzero().t().detach()


    






class AttMatch(torch.nn.Module):
    def __init__(self,in_channels,out_channels,heads = 1):
        super(AttMatch, self).__init__()
        self.in_channels = in_channels
        self.out_channels= out_channels
        self.heads = heads

        self.conv1 = SAGEConv(in_channels*2,out_channels)
        self.conv2 = SAGEConv(out_channels*2,out_channels)


        self.lin_key1 = torch.nn.Linear(in_channels, heads * out_channels)
        self.lin_query1 = torch.nn.Linear(in_channels, heads * out_channels)
        self.lin_value1 = torch.nn.Linear(in_channels, heads * in_channels)

        self.lin_key2 = torch.nn.Linear(out_channels, heads * out_channels)
        self.lin_query2 = torch.nn.Linear(out_channels, heads * out_channels)
        self.lin_value2 = torch.nn.Linear(out_channels, heads * out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        for tensor in self.parameters():
            if len(tensor.size())==1:
                bound = 1.0 / math.sqrt(tensor.size()[0])
                tensor.data.uniform_(-bound, bound)
            else:
                stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
                tensor.data.uniform_(-stdv, stdv)


    def get_attention(self,Xs,index,layer):
        source = Xs[index]
        target = torch.cat(Xs,dim = 0)

        if layer == 1:
            query = self.lin_query1(source).view(-1, self.heads * self.out_channels)
            key = self.lin_key1(target).view(-1, self.heads * self.out_channels)
            alpha = ( key@query.t() ) / math.sqrt(self.out_channels)
            alpha = F.softmax(alpha,dim=0)
            alpha = alpha/(alpha.sum(0)+1e-16)


        if layer == 2:
            query = self.lin_query2(source).view(-1, self.heads * self.out_channels)
            key = self.lin_key2(target).view(-1, self.heads * self.out_channels)
            alpha = ( key@query.t() ) / math.sqrt(self.out_channels)
            alpha = F.softmax(alpha,dim=0)
            alpha = alpha/(alpha.sum(0)+1e-16)
        

        return alpha



    def aggr_and_update(self,graphs,Xs,layer):
        Xs_new = []
        target = torch.cat(Xs,dim = 0)
        for i,key in enumerate(graphs.keys()):
            attention = self.get_attention(Xs,i,layer)

            if layer == 1:
            
                out = self.lin_value1(target).view(-1, self.heads * self.in_channels)
                out = attention.t() @ out

            if layer == 2:
            
                out = self.lin_value2(target).view(-1, self.heads * self.out_channels)
                out = attention.t() @ out







            ot_feature = Xs[i] - out
            features = torch.cat((Xs[i],ot_feature),dim = 1)
            if layer == 1:Xs_new.append(F.relu(self.conv1(x = features, edge_index = graphs[key].edge_index)))
            if layer == 2:Xs_new.append(self.conv2(x = features, edge_index = graphs[key].edge_index))

        return Xs_new
    


    def get_adj(self,Xs):

        features = torch.cat(Xs,dim = 0)
        
        alpha =  features@features.t()    #(self.lin(features).view(-1, self.head, self.out_channels)*self.att).sum(dim=-1).mean(1,keepdim = True)
        
        adj = F.sigmoid(alpha)
        print(adj)

        return adj


    def get_final_adj(self,adj,graphs,threshold):

        cum = 0
        for key in graphs.keys():
            graph = graphs[key]

            adj[cum:cum+graph.x.shape[0],cum:cum+graph.x.shape[0]] = 0.0
            adj[graph.edge_index[0]+cum,graph.edge_index[1]+cum] = 1.0
            cum+=graph.x.shape[0]

        adj = torch.where(adj>threshold,adj,torch.tensor(0, dtype=torch.float).to(adj))
        print('adj.sum()',adj.sum())
        return adj.nonzero().t().detach()


    def forward(self,graphs):
        Xs = [graphs[key].x for key in graphs.keys()]
        Xs1 = self.aggr_and_update( graphs,Xs,1)
        Xs2 = self.aggr_and_update(graphs,Xs1, 2)
        adj = self.get_adj(Xs2)

        
        return adj






class matchGAT(torch.nn.Module):

    def __init__(self,in_channels,out_channels,head = 1):
        super(matchGAT, self).__init__()
        self.in_channels = in_channels
        self.out_channels= out_channels
        self.head = head
        self.wight1 = nn.Linear(in_channels*2, out_channels, bias=True)
        self.wight2 = nn.Linear(out_channels*2, out_channels, bias=True)

        self.lin_l1 = nn.Linear(in_channels, head * out_channels, bias=True)
        self.lin_r1 = nn.Linear(in_channels, head * out_channels, bias=True)

        self.lin_l2 = nn.Linear(out_channels, head * out_channels, bias=True)
        self.lin_r2 = nn.Linear(out_channels, head * out_channels, bias=True)

        #self.lin_l = nn.Linear(out_channels, head * out_channels, bias=True)
        #self.lin_r = nn.Linear(out_channels, head * out_channels, bias=True)
        
        
        self.att_l1 = Parameter(torch.Tensor(1,head, out_channels))
        self.att_r1 = Parameter(torch.Tensor(1,head, out_channels))

        self.att_l2 = Parameter(torch.Tensor(1,head, out_channels))
        self.att_r2 = Parameter(torch.Tensor(1,head, out_channels))

        #self.att_l = Parameter(torch.Tensor(1,head, out_channels))
        #self.att_r = Parameter(torch.Tensor(1,head, out_channels))

        self.lin = nn.Linear(out_channels, head * out_channels, bias=True)
        self.att = Parameter(torch.Tensor(1,head, out_channels))

        self.reset_parameters()



    def reset_parameters(self):
        for tensor in self.parameters():
            if len(tensor.size())==1:
                bound = 1.0 / math.sqrt(tensor.size()[0])
                tensor.data.uniform_(-bound, bound)
            else:
                stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
                tensor.data.uniform_(-stdv, stdv)



    def get_attention(self,Xs,index,layer):
        source = Xs[index]
        target = torch.cat(Xs,dim = 0)


        if layer == 1:
            alpha_l = (self.lin_l1(source).view(-1, self.head, self.out_channels)*self.att_l1).sum(dim=-1).mean(1,keepdim = True)
            alpha_r = (self.lin_r1(target).view(-1, self.head, self.out_channels)*self.att_r1).sum(dim=-1).mean(1,keepdim = True)

        if layer == 2:
            alpha_l = (self.lin_l2(source).view(-1, self.head, self.out_channels)*self.att_l2).sum(dim=-1).mean(1,keepdim = True)
            alpha_r = (self.lin_r2(target).view(-1, self.head, self.out_channels)*self.att_r2).sum(dim=-1).mean(1,keepdim = True)

        
        
        alpha = alpha_l+alpha_r.t()

        

        alpha = F.leaky_relu(alpha,0.2)
        alpha = F.softmax(alpha,dim = 1)

        return alpha
    

    def aggr_and_update(self,graphs,Xs,layer):
        Xs_new = []
        target = torch.cat(Xs,dim = 0)
        for i,key in enumerate(graphs.keys()):
            attention = self.get_attention(Xs,i,layer)
            ot_feature = Xs[i] - torch.mm(attention, target)
            features = torch.cat((Xs[i],ot_feature),dim = 1)


            source_idx,target_idx = add_self_loops(graphs[key].edge_index)[0]
            messages = features.index_select(0, target_idx)
            aggregation = scatter(messages, source_idx, dim=0, dim_size=features.shape[0], reduce='mean')

            if layer == 1:aggregation = self.wight1(F.relu(aggregation))
            if layer == 2:aggregation = self.wight2(F.relu(aggregation))

            Xs_new.append(aggregation)

        return Xs_new
    

    def get_adj2(self,Xs):

        source = torch.cat(Xs,dim = 0)
        target = torch.cat(Xs,dim = 0)


        




        alpha_l = (self.lin_l(source).view(-1, self.head, self.out_channels)*self.att_l).sum(dim=-1).mean(1,keepdim = True)
        alpha_r = (self.lin_r(target).view(-1, self.head, self.out_channels)*self.att_r).sum(dim=-1).mean(1,keepdim = True)

        #alpha = (alpha_l+alpha_r)/2.0
        #alpha = alpha+alpha.t()

        alpha = alpha_l+alpha_r.t()
        alpha = F.sigmoid(alpha)

        adj = torch.where(alpha>0.6,alpha,torch.tensor(0, dtype=torch.float).to(alpha))
        return adj
    

    def get_adj(self,Xs):

        features = torch.cat(Xs,dim = 0)

        features = self.lin(features)
        
        alpha =  features@features.t()    #(self.lin(features).view(-1, self.head, self.out_channels)*self.att).sum(dim=-1).mean(1,keepdim = True)
        
        adj = F.sigmoid(alpha)
        print(adj)

        return adj
    
    def get_final_adj(self,adj,graphs,threshold):

        cum = 0
        for key in graphs.keys():
            graph = graphs[key]

            adj[cum:cum+graph.x.shape[0],cum:cum+graph.x.shape[0]] = 0.0
            adj[graph.edge_index[0]+cum,graph.edge_index[1]+cum] = 1.0
            cum+=graph.x.shape[0]

        adj = torch.where(adj>threshold,adj,torch.tensor(0, dtype=torch.float).to(adj))
        print('adj.sum()',adj.sum())
        return adj.nonzero().t().detach()


    def forward(self,graphs):
        Xs = [graphs[key].x for key in graphs.keys()]
        Xs1 = self.aggr_and_update( graphs,Xs,1)
        Xs2 = self.aggr_and_update(graphs,Xs1, 2)
        adj = self.get_adj(Xs2)

        
        return adj











        
        



        




