
from copy import deepcopy
from sqlite3 import paramstyle
import ssl
import numpy as np
from sklearn.utils import shuffle
import torch
import itertools
seed = 0
torch.manual_seed(seed)
import torch.nn.functional as F
import torch_geometric
import time
import argparse
from tqdm import tqdm
from pool import *
from models.models import *
from data.load_data import *
from models.autoencoder import MyGAE,MyVGAE,MyTask,MLPDecoder
from train import *
from utils import get_positional_embedding, seed_everything,compute_acc_unsupervised
import scipy.sparse as sp

parser = argparse.ArgumentParser()


parser.add_argument('--train_dataset', type=list, default=['academic','product','yelp','reddit'])
parser.add_argument('--test_dataset', type=list, default=['reddit'])
parser.add_argument('--share_dims', type=int, default=128)
parser.add_argument('--data_dims', type=dict, default={'academic':128,'product':100,'yelp':300,'reddit':602})
parser.add_argument('--hidden_dims', type=int, default=128)
parser.add_argument('--subgraph_scale', type=int, default=4096)
parser.add_argument('--fuse_scale', type=int, default=1024)
parser.add_argument('--model_lr', type=float, default=0.005)
parser.add_argument('--feature_lr', type=float, default=0.0001)
parser.add_argument('--wd_lr', type=float, default=0.01)
parser.add_argument('--inner_lr', type=float, default=0.01)
parser.add_argument('--output_dims', default=128, type=int)
parser.add_argument('--inner_train_steps', default=10, type=int)
parser.add_argument('--layer_norm', default=False, action='store_true',help='use layer norm')
parser.add_argument('--cuda', type = int,default=3)









def MDP_train(args, device, task):


    file = open('./result/pretrain_process_for_{}_{}_{}.csv'.format('-'.join(args.train_dataset),task), 'w')
    dataloader = dataGraph(args)

    MLPs = {}
    args.dims = [dataloader.original_datas[dataset].x.shape[1] for dataset in args.train_dataset ]
    dim_sum=int(np.sum(args.dims))

    for dataset in args.train_dataset:
        dim=dataloader.original_datas[dataset].x.shape[1]
        MLPs[dataset] = MLP(dim, args.hidden_dims, dim_sum-dim).to(device)

    wdiscriminator = WDiscriminator(dim_sum)
    optimizer_wd = torch.optim.Adam(wdiscriminator.parameters(), lr=args.wd_lr, weight_decay=5e-4)
    wdiscriminator.to(device)
    params = []
    for value in MLPs.values():
        params.extend(value.parameters())
    pretrain_model = MyGAE(TransformerEncoder(args, int(dim_sum), args.output_dims))
    pretrain_model.to(device)

    model_params = []
    model_params.extend(pretrain_model.parameters())

    if task == 'nmk':
        auxiliary = nn.Linear(args.output_dims, dim_sum, bias=True).to(device)
        model_params.extend(auxiliary.parameters())
    elif task == 'dgi':
        auxiliary = nn.Linear(args.output_dims, args.output_dims, bias=True).to(device)
        model_params.extend(auxiliary.parameters())
    elif task == 'sim':
        auxiliary = nn.Linear(args.output_dims, args.output_dims, bias=False).to(device)
        model_params.extend(auxiliary.parameters())
    else:
        auxiliary = None

    optimizer_all = torch.optim.Adam([{'params': params, 'lr': args.feature_lr},{'params': model_params}], lr=args.model_lr,
                                     weight_decay=5e-4)  # 图数据还是应该单独更新

    epoch = 0
    loss_value = 0
    cnt = 0
    optimal_model = pretrain_model
    last_model = pretrain_model


    #match_graph = AttMatch(dim_sum,args.share_dims).to(device)
    match_graph = ConnectMatch(dim_sum).to(device)
    # match_graph = ConnectMatch_mlp(args,dim_sum,device,proto_num=256).to(device)
    # match_graph.set_super_nodes(dataloader.fetch_subgraph())

    while True:
        subgraphs = dataloader.fetch_subgraph()
        fugraphs = dataloader.fetch_for_fused()

        dataloader.subgraph_to_tensor(subgraphs, device)
        dataloader.subgraph_to_tensor(fugraphs, device)


        for idx,name in enumerate(args.train_dataset):
            fugraphs[name] = fugraphs[name].transformation(MLPs[name],int(np.sum(args.dims[:idx])))
            fugraphs[name].x = fugraphs[name].x.detach()

        pretrain_model, loss = MDP_gradient_fuse_new(args, task,pretrain_model,auxiliary,subgraphs,fugraphs,optimizer_all, MLPs,wdiscriminator,optimizer_wd,match_graph, epoch, device, file)

        if epoch%3==0:
            accs = []
            datas = args.train_dataset
            for valid in datas:
                valid_graph = load_test_graph(valid, 5)
                valid_graph.to_tensor(device)
                valid_graph.identify('node', 0.05)
                x = MLPs[valid](valid_graph.x)
                dim = int(np.sum(args.dims[:args.train_dataset.index(valid)]))
                x = torch.cat([x[:,:dim],valid_graph.x,x[:,dim:]],dim=1)
                weights = OrderedDict(pretrain_model.named_parameters())
                emb = pretrain_model.encode(x, valid_graph.edge_index, weights)
                accs.append( compute_acc_unsupervised(emb.detach(), valid_graph.labels, valid_graph.train_labels, valid_graph.test_labels))
            print(datas,accs)
            loss = np.mean(accs)
            file.write(','.join([str(acc) for acc in accs] )+',--,'+str(loss)+'\n')
            file.flush()


            print('loss', loss, cnt, epoch)
            if loss > loss_value  and epoch > 20:
                optimal_model = last_model
                loss_value = loss
                cnt = 0
                torch.save(optimal_model.encoder.state_dict(),
                        './result/pretrain_encoder_for_{}_{}.pth'.format('-'.join(args.train_dataset),task))

                mlps_state = {}
                for data in args.train_dataset:mlps_state[data] = MLPs[data].state_dict()
                torch.save(mlps_state,'./result/pretrain_mlp_for_{}_{}.pth'.format('-'.join(args.train_dataset),task))

            else:
                cnt += 1

            if cnt == 400:  # 500
                torch.save(optimal_model.encoder.state_dict(),
                        './result/pretrain_encoder_for_{}_{}.pth'.format('-'.join(args.train_dataset),task))

                mlps_state = {}
                for data in args.train_dataset:mlps_state[data] = MLPs[data].state_dict()
                torch.save(mlps_state,'./result/pretrain_mlp_for_{}_{}.pth'.format('-'.join(args.train_dataset),task))

                file.close()
                break

            last_model = deepcopy(pretrain_model)

        epoch += 1




def node_tuning(args, device, index, pretrain_task, ratio, finetuning=True,mlp_flag=True):
    dataset = args.test_dataset[0]

    graph = load_test_graph(dataset, index)
    graph.to_tensor(device)
    graph.identify('node', ratio)

    dims_sum = np.sum([args.data_dims[data] for data in args.train_dataset])
    mlp = MLP(graph.x.shape[1], args.hidden_dims, dims_sum - graph.x.shape[1]).to(device)
    if mlp_flag: mlp.load_state_dict(
        torch.load('./result/pretrain_mlp_for_{}_{}.pth'.format('-'.join(args.train_dataset), pretrain_task))[
            args.test_dataset[0]], strict=True)

    encoder = TransformerEncoder(args, int(dims_sum), args.output_dims)
    mlp.Dropout.p = 0.8
    if len(graph.labels.shape) == 2:
        num_classes = int(graph.labels.shape[1])
        model = MyTask('node_m', encoder, args.output_dims, num_classes)
    else:
        num_classes = int(graph.labels.max().item() + 1)
        model = MyTask('node', encoder, args.output_dims, num_classes)

    model.encoder.load_state_dict(
        torch.load('./result/pretrain_encoder_for_{}_{}.pth'.format('-'.join(args.train_dataset), pretrain_task)),strict=True)

    if finetuning:
        optimizer = torch.optim.Adam(
            [{'params': mlp.parameters(), 'lr': 0.001}, {'params': model.parameters(), 'lr': 0.001}],
            weight_decay=5e-4)
    else:
        for p in model.encoder.parameters(): p.requires_grad = False
        optimizer = torch.optim.Adam(
            [{'params': mlp.parameters(), 'lr': 0.001}, {'params': model.decoder.parameters(), 'lr': 0.001}],
            weight_decay=5e-4)

    args.dims = [args.data_dims[dataset] for dataset in args.train_dataset]
    return train_graph_node(mlp=mlp, model=model, graph=graph, optimizer=optimizer,
                            dim =  int(np.sum(args.dims[:args.train_dataset.index(args.test_dataset[0])])), device=device)




def link_tuning(args, device, index, pretrain_task,ratio, finetuning = True,mlp_flag=True):
    dataset = args.test_dataset[0]

    graph = load_test_graph(dataset, index)
    graph.to_tensor(device)
    graph.identify('link', ratio)

    dims_sum = np.sum([args.data_dims[data] for data in args.train_dataset])
    mlp = MLP(graph.x.shape[1], args.hidden_dims, dims_sum - graph.x.shape[1]).to(device)
    if mlp_flag: mlp.load_state_dict(
        torch.load('./result/pretrain_mlp_for_{}_{}.pth'.format('-'.join(args.train_dataset), pretrain_task))[
            args.test_dataset[0]], strict=True)

    encoder = TransformerEncoder(args, int(dims_sum), args.output_dims)
    mlp.Dropout.p = 0.6
    model = MyTask('link', encoder, args.share_dims)

    model.encoder.load_state_dict(
        torch.load('./result/pretrain_encoder_for_{}_{}.pth'.format('-'.join(args.train_dataset), pretrain_task)),
        strict=True)

    if finetuning:
        optimizer = torch.optim.Adam(
                [{'params': mlp.parameters(), 'lr': 0.001}, {'params': model.parameters(), 'lr': 0.001}],
                weight_decay=5e-4)
    else:
        for p in model.encoder.parameters(): p.requires_grad = False
        optimizer = torch.optim.Adam(
                [{'params': mlp.parameters(), 'lr': 0.001}, {'params': model.decoder.parameters(), 'lr': 0.001}],
                weight_decay=5e-4)

    args.dims = [args.data_dims[data] for data in args.train_dataset ]
    return train_graph_link(mlp=mlp, model=model, graph=graph, optimizer=optimizer,device=device)






def graph_tuning(args, device, pretrain_task, num, finetuning = True):
    dataset = args.test_dataset
    graphs = TUDataset(root='./MDP-data/graph/' + dataset, name=dataset, use_node_attr=True)

    all_loader = DataLoader(graphs, batch_size=len(graphs), shuffle=False)
    graph_ = next(iter(all_loader))
    labels = graph_.y.detach().numpy()

    true_index = np.where(labels == 1)[0]
    false_index = np.where(labels == 0)[0]

    train_index = true_index[:num].tolist() + false_index[:num].tolist()

    # valid_index = true_index[num:num+int(len(labels)*0.1/2)]+false_index[num:num+int(len(labels)*0.1/2)]
    test_index = true_index[num + int(len(labels) * 0.1 / 2):].tolist() + false_index[
                                                                          num + int(len(labels) * 0.1 / 2):].tolist()

    train_loader = DataLoader(graphs[train_index], batch_size=len(train_index), shuffle=False)
    test_loader = DataLoader(graphs[test_index], batch_size=len(test_index), shuffle=False)

    train_graph_ = next(iter(train_loader))
    train_graph = Graph(train_graph_.x, train_graph_.edge_index)
    train_graph.to(device)
    train_graph.batch = train_graph_.batch.to(device)
    train_graph.y = train_graph_.y.to(device).to(torch.float)

    test_graph_ = next(iter(test_loader))
    test_graph = Graph(test_graph_.x, test_graph_.edge_index)
    test_graph.to(device)
    test_graph.batch = test_graph_.batch.to(device)
    test_graph.y = test_graph_.y.to(device).to(torch.float)

    dims_sum = np.sum([args.data_dims[data] for data in args.train_dataset])
    mlp = MLP(test_graph.x.shape[1], args.hidden_dims, dims_sum).to(device)
    mlp.Dropout.p = 0.5

    encoder = TransformerEncoder(args, int(dims_sum), args.output_dims)
    model = MyTask('node_m', encoder, args.output_dims, 1).to(device)
    model.encoder.load_state_dict(
        torch.load('./result/pretrain_encoder_for_{}_{}.pth'.format('-'.join(args.train_dataset), pretrain_task)),
        strict=True)

    if finetuning:
        optimizer = torch.optim.Adam(
            [{'params': mlp.parameters(), 'lr': 0.0001}, {'params': model.parameters(), 'lr': 0.0001}],
            weight_decay=5e-4)
    else:
        for p in model.encoder.parameters(): p.requires_grad = False
        optimizer = torch.optim.Adam(
            [{'params': mlp.parameters(), 'lr': 0.0001}, {'params': model.decoder.parameters(), 'lr': 0.0001}],
            weight_decay=5e-4)

    return train_graph_graph(mlp, model, train_graph, test_graph, optimizer, device)





args = parser.parse_args()
if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")








MDP_train(args, device, 'edge')
MDP_train(args, device, 'nmk')
MDP_train(args, device, 'dgi')
MDP_train(args, device, 'sim')

node_tuning(args, device,0,'edge',0.05, 5,True,True)
node_tuning(args, device,0,'nmk',0.05, 5,True,True)
node_tuning(args, device,0,'dgi',0.05, 5,True,True)
node_tuning(args, device,0,'sim',0.05, 5,True,True)






