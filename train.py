from turtle import pos
from copy import deepcopy
from unittest import result
import torch
import torch.nn.functional as F
import sklearn.neighbors
import copy
from collections import OrderedDict
import numpy as np
from utils import test,test2,seed_everything
from models.models import MLP,WDiscriminator,WDiscriminator_old,matchGAT
from models.autoencoder import negative_sampling
import time
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import roc_auc_score, f1_score
from scipy.sparse import coo_array
from data.load_data import *
from utils import *



EPS = 1e-15











def train_wdiscriminator(graph_s, graph_t, wdiscriminator, optimizer_d, batch_d_per_iter=20):


    wdiscriminator.reset_parameters()

    for p in wdiscriminator.parameters(): p.requires_grad = True


    if not isinstance(graph_t,list):graph_t = [graph_t]


    for j in range(batch_d_per_iter):
        wdiscriminator.train()

        w1s = []
        for graph in graph_t:
            w1s.append(wdiscriminator(graph.x.detach(),graph.edge_index))


        w0 = wdiscriminator(graph_s.x.detach(),graph_s.edge_index)
        w1 = torch.vstack(w1s)

        loss = -torch.mean(w1) + torch.mean(w0)
        #if j% 40 ==0:print(loss.item())

        optimizer_d.zero_grad()
        loss.backward()
        optimizer_d.step()

        for p in wdiscriminator.parameters(): p.data.clamp_(-0.01, 0.01)

    return wdiscriminator



def train_wdiscriminator_align(graph_s, graph_t, wdiscriminator, optimizer_d, batch_d_per_iter=20):


    wdiscriminator.reset_parameters()

    for p in wdiscriminator.parameters(): p.requires_grad = True


    for j in range(batch_d_per_iter):
        wdiscriminator.train()


        w0 = wdiscriminator(graph_s.x.detach(),graph_s.edge_index)
        w1 = wdiscriminator(graph_t.x.detach(),graph_t.edge_index)

        loss = -torch.mean(w1) + torch.mean(w0)
        #if j% 40 ==0:print(loss.item())

        optimizer_d.zero_grad()
        loss.backward()
        optimizer_d.step()

        for p in wdiscriminator.parameters(): p.data.clamp_(-0.01, 0.01)



    w0 = wdiscriminator(graph_s.x.detach(),graph_s.edge_index).cpu().detach().numpy()[:,0]
    w1 = wdiscriminator(graph_t.x.detach(),graph_t.edge_index).cpu().detach().numpy()[:,0]

    index0 = np.argsort(w0)
    index1 = np.argsort(-w1)


    

    return index0[np.argsort(index1)]

    
    

def domain_tansformation_old(graph_s,graph_syn, args, device):

    model = MLP(graph_s.x.shape[1], args.hidden_dims, args.share_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    wdiscriminator = WDiscriminator_old(args.share_dims*2).to(device)
    optimizer_wd = torch.optim.Adam(wdiscriminator.parameters(), lr=args.wd_lr, weight_decay=5e-4)

    best_value = 1e8
    cnt = 0
    best_model = copy.deepcopy(model).to(device)
    while True:

        model.train()

        loss_all = torch.tensor(0.0).to(device)
        for graph_t in graph_syn.syn_graphs:

            space_s = construct_space(model,graph_s.x.size(0)*2,graph_s.edge_index,graph_s.x)
            space_t = construct_space(None,graph_t.x.size(0)*2,graph_t.edge_index,graph_t.x)
            wdiscriminator_copy = copy.deepcopy(
                train_wdiscriminator_old(space_s.detach(), space_t.detach(),graph_s.x.size(0), wdiscriminator,
                                     optimizer_wd, batch_d_per_iter=80)) 

            for p in wdiscriminator_copy.parameters(): p.requires_grad = False
            wdiscriminator_copy.to(device)


            w0 = wdiscriminator_copy(space_s[:graph_s.x.size(0)])
            w1 = wdiscriminator_copy(space_t[:graph_t.x.size(0)])

            w2 = wdiscriminator_copy(space_s[graph_s.x.size(0):])
            w3 = wdiscriminator_copy(space_t[graph_t.x.size(0):])

            loss = (torch.mean(w1) - torch.mean(w0) + torch.mean(w3) - torch.mean(w2))

            loss_all+=loss

        


        value = loss_all.item()
        #print('XXXXXXXXXXXXXXXXXXXXX',value)
        if cnt >0:return best_model
        if value < 0:
            cnt+=1
        elif value>=0 and value < best_value:
            best_value = value
            cnt = 0
            best_model = copy.deepcopy(model)
        else: cnt+=1       
        print(cnt)
        if cnt >0:return best_model

        
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()









def domain_tansformation(graph_s,graph_syn, args, device):

    model = MLP(graph_s.x.shape[1], args.hidden_dims, args.share_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    wdiscriminator = WDiscriminator(args.share_dims).to(device)
    optimizer_wd = torch.optim.Adam(wdiscriminator.parameters(), lr=args.wd_lr, weight_decay=5e-4)

    best_value = 1e8
    cnt = 0
    best_model = copy.deepcopy(model).to(device)
    while True:

        model.train()

        loss_all = torch.tensor(0.0).to(device)
        for graph_t in graph_syn.syn_graphs:

            wdiscriminator_copy = copy.deepcopy(train_wdiscriminator(graph_s.transformation(model) , graph_t,\
                 wdiscriminator, optimizer_wd, batch_d_per_iter=80))

            for p in wdiscriminator_copy.parameters(): p.requires_grad = False

            wdiscriminator_copy.to(device)
            

            
            w0 = wdiscriminator_copy(model(graph_s.x), graph_s.edge_index)
            w1 = wdiscriminator_copy(graph_t.x, graph_t.edge_index)

            loss = torch.mean(w1) - torch.mean(w0)
            #print((torch.mean(w1) - torch.mean(w0)).item())

            loss_all+=loss

        


        value = loss_all.item()
        print('XXXXXXXXXXXXXXXXXXXXX',value)
        if cnt >0:return best_model
        if value < 0:
            cnt+=1
        elif value>=0 and value < best_value:
            best_value = value
            cnt = 0
            best_model = copy.deepcopy(model)
        else: cnt+=1       
        #print(cnt)


        
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()









def domain_tansformation2(graph_s,graph_syn, args, device):

    model = MLP(graph_s.x.shape[1], args.hidden_dims, args.share_dims).to(device)

    


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    wdiscriminator = WDiscriminator(args.share_dims).to(device)
    optimizer_wd = torch.optim.Adam(wdiscriminator.parameters(), lr=args.wd_lr, weight_decay=5e-4)

    best_value = 1e8
    cnt = 0
    best_model = copy.deepcopy(model).to(device)
    while True:

        model.train()

        loss_all = torch.tensor(0.0).to(device)

        wdiscriminator_copy = copy.deepcopy(train_wdiscriminator(graph_s.transformation(model) , graph_syn.syn_graphs,\
                 wdiscriminator, optimizer_wd, batch_d_per_iter=40))

        for p in wdiscriminator_copy.parameters(): p.requires_grad = False

        wdiscriminator_copy.to(device)
            

        w1s = []
        w0 = wdiscriminator_copy(model(graph_s.x), graph_s.edge_index)

        for graph in graph_syn.syn_graphs:
            w1s.append(wdiscriminator_copy(graph.x, graph.edge_index))
        w1 = torch.vstack(w1s)
        loss_all = torch.mean(w1) - torch.mean(w0)



        value = loss_all.item()
        print('XXXXXXXXXXXXXXXXXXXXX',value)
        if cnt >0:return best_model
        if value < 0:
            cnt+=1
        elif value>=0 and value < best_value:
            best_value = value
            cnt = 0
            best_model = copy.deepcopy(model)
        else: cnt+=1       
        #print(cnt)


        
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()




def domain_tansformation3(graph_s,graph_syn, args, device):

    model = MLP(graph_s.x.shape[1], args.hidden_dims, args.share_dims).to(device)
    model_out = MLP(args.share_dims, args.hidden_dims, graph_s.x.shape[1]).to(device)
    


    optimizer = torch.optim.Adam([{'params':model.parameters(),'lr':0.001},{'params':model_out.parameters(),'lr':0.0005}])

    wdiscriminator = WDiscriminator(args.share_dims).to(device)
    optimizer_wd = torch.optim.Adam(wdiscriminator.parameters(), lr=args.wd_lr, weight_decay=5e-4)

    best_value = 1e8
    cnt = 0
    best_model = copy.deepcopy(model).to(device)
    while True:

        model.train()

        loss_all = torch.tensor(0.0).to(device)

        wdiscriminator_copy = copy.deepcopy(train_wdiscriminator(graph_s.transformation(model) , graph_syn.syn_graphs,\
                 wdiscriminator, optimizer_wd, batch_d_per_iter=80))

        for p in wdiscriminator_copy.parameters(): p.requires_grad = False

        wdiscriminator_copy.to(device)
            

        w1s = []
        w0 = wdiscriminator_copy(model(graph_s.x), graph_s.edge_index)

        for graph in graph_syn.syn_graphs:
            w1s.append(wdiscriminator_copy(graph.x, graph.edge_index))
        w1 = torch.vstack(w1s)
        loss = torch.mean(w1) - torch.mean(w0)

        loss_AE =torch.nn.MSELoss(size_average=True)(graph_s.x,model_out(model(graph_s.x)))

        loss_all = loss_AE #+loss



        value = loss_all.item()
        print('XXXXXXXXXXXXXXXXXXXXX',value,loss.item(),loss_AE.item())
        if cnt >2:return best_model
        if value < 0:
            cnt+=1
        elif value>=0 and value < best_value:
            best_value = value
            cnt = 0
            best_model = copy.deepcopy(model)
        else: cnt+=1       
        print(cnt)


        
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()





def train_graph(mlp,model, graph,optimizer,dim,device,file=None):

    if mlp!= None:mlp.to(device)
    model.to(device)

    value = 0
    count = 0
    result = []
    for epoch in range(1,1000):

        model.train()
        if mlp!= None:mlp.train()
        
        weights = OrderedDict(model.named_parameters())
        if mlp!= None:
            x = mlp(graph.x)
            x = torch.cat([x[:,:dim],graph.x,x[:,dim:]],dim=1)
            if model.task == 'link':z = model.encode(x, graph.train_edge_index, weights, inner_loop=True)
            else:z = model.encode(x, graph.edge_index, weights, inner_loop=True)
        else:
            if model.task == 'link':z = model.encode(graph.x, graph.train_edge_index, weights, inner_loop=True)
            else:z = model.encode(graph.x, graph.edge_index, weights, inner_loop=True)


        
        if model.task == 'link':loss = model.recon_loss(z,graph.train_edge_index)
        else:loss = model.class_loss(z[graph.train_labels], graph.labels[graph.train_labels])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for p in model.parameters():
            p.data.clamp_(-0.1, 0.1)

        model.eval()
        if mlp!= None:mlp.eval()




        with torch.no_grad():
            weights = OrderedDict(model.named_parameters())
            
            if mlp!= None:
                x = mlp(graph.x)
                x = torch.cat([x[:,:dim],graph.x,x[:,dim:]],dim=1)
                if model.task == 'link':z = model.encode(x, graph.train_edge_index, weights, inner_loop=True)
                else:z = model.encode(x, graph.edge_index, weights, inner_loop=True)
            else:
                if model.task == 'link':z = model.encode(graph.x, graph.train_edge_index, weights, inner_loop=True)
                else:z = model.encode(graph.x, graph.edge_index, weights, inner_loop=True)

            if model.task == 'link':score = model.test(z,graph.test_edge_index,graph.test_edge_index_negative )[0]
            else: 
                score = model.class_test(z[graph.test_labels], graph.labels[graph.test_labels])



        if epoch in [0,2,4,6,8,10,20,50,100,150,200,300,400,500,600,700,800,900,1000]:print(epoch,value,loss.item())
        
        #if epoch<=500:print(epoch,score,loss.item())
        if file!=None:file.write('{},{},{},{}\n'.format(epoch,count,score))
        if value<score:
            value = score
            count = 0
        else:count+=1
        result.append(value)
    if file!=None:file.write('the best value:,{}\n'.format(value))
    print('the best value: ', value)
    return result






def train_graph2(mlp,model, graph,optimizer,device,file=None):

    if mlp!= None:mlp.to(device)
    model.to(device)

    value = 0
    count = 0
    result = []
    for epoch in range(1,2000):

        model.train()

        
        weights = OrderedDict(model.named_parameters())
        if mlp!= None:

            x = mlp.add(graph.x)
            if model.task == 'link':z = model.encode(x, graph.train_edge_index, weights, inner_loop=True)
            else:z = model.encode(x, graph.edge_index, weights, inner_loop=True)
        else:
            if model.task == 'link':z = model.encode(graph.x, graph.train_edge_index, weights, inner_loop=True)
            else:z = model.encode(graph.x, graph.edge_index, weights, inner_loop=True)


        
        if model.task == 'link':loss = model.recon_loss(z,graph.train_edge_index)
        else:loss = model.class_loss(z[graph.train_labels], graph.labels[graph.train_labels])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for p in model.parameters():
            p.data.clamp_(-0.1, 0.1)

        model.eval()





        with torch.no_grad():
            weights = OrderedDict(model.named_parameters())
            
            if mlp!= None:
                x = mlp.add(graph.x)
                if model.task == 'link':z = model.encode(x, graph.train_edge_index, weights, inner_loop=True)
                else:z = model.encode(x, graph.edge_index, weights, inner_loop=True)
            else:
                if model.task == 'link':z = model.encode(graph.x, graph.train_edge_index, weights, inner_loop=True)
                else:z = model.encode(graph.x, graph.edge_index, weights, inner_loop=True)

            if model.task == 'link':score = model.test(z,graph.test_edge_index,graph.test_edge_index_negative )[0]
            else: 
                score = model.class_test(z[graph.test_labels], graph.labels[graph.test_labels])



        if epoch in [0,2,4,6,8,10,20,50,100,150,200,300,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000]:print(epoch,value,loss.item())
        
        #if epoch<=500:print(epoch,score,loss.item())
        if file!=None:file.write('{},{},{},{}\n'.format(epoch,count,score))
        if value<score:
            value = score
            count = 0
        else:count+=1
        result.append(value)
    if file!=None:file.write('the best value:,{}\n'.format(value))
    print('the best value: ', value)
    return result






def train_graph_link(mlp, model, graph, optimizer, device):
    mlp.to(device)
    model.to(device)
    auc, acc = 0, 0

    result = []
    for epoch in range(1, 1000):

        model.train()
        mlp.train()

        weights = OrderedDict(model.named_parameters())
        x = mlp(graph.x)
        z = model.encode(x, graph.train_edge_index, weights, inner_loop=True)
        loss = model.recon_loss(z, graph.train_edge_index)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for p in model.parameters():
            p.data.clamp_(-0.1, 0.1)

        model.eval()
        mlp.eval()

        with torch.no_grad():
            weights = OrderedDict(model.named_parameters())
            x = mlp(graph.x)
            z = model.encode(x, graph.train_edge_index, weights, inner_loop=True)
            auc_, acc_ = model.test(z, graph.test_edge_index, graph.test_edge_index_negative)
            result.append((auc, acc))

        if epoch in [0, 2, 4, 6, 8, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]: print(epoch, (
        auc, acc), loss.item())
        if acc < acc_:
            acc = acc_
        if auc < auc_:
            auc = auc_

    print('the best value: ', (auc, acc))
    return result



def train_graph_graph(mlp,model, train_graph, test_graph,optimizer,device):

    model.to(device)
    mlp.to(device)
    f1,auc = 0,0
    result = []
    for epoch in range(1,1000):

        model.train()
        mlp.train()
        weights = OrderedDict(model.named_parameters())

        z = model.encode(mlp(train_graph.x), train_graph.edge_index, weights, inner_loop=True)
        z = global_mean_pool(z,train_graph.batch)

        loss = model.class_loss(z, train_graph.y.reshape(-1,1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for p in model.parameters():
            p.data.clamp_(-0.1, 0.1)
        model.eval()
        mlp.eval()

        with torch.no_grad():
            weights = OrderedDict(model.named_parameters())
            z = model.encode(mlp(test_graph.x), test_graph.edge_index, weights, inner_loop=True)
            z = global_mean_pool(z,test_graph.batch)
            f1_,auc_ = model.class_test_all(z, test_graph.y.reshape(-1,1))

        if epoch in [0,2,4,6,8,10,20,50,100,150,200,300,400,500,600,700,800,900,1000]:print(epoch,(f1,auc),loss.item())

        if f1<f1_:
            f1 = f1_
        if auc<auc_:
            auc = auc_
        result.append((f1,auc))
    print('the best value: ', (f1,auc))
    return result




def train_graph_node(mlp, model, graph, optimizer, dim, device):
    mlp.to(device)
    model.to(device)

    f1, accuracy, auc = 0, 0, 0
    count = 0
    result = []
    for epoch in range(1, 1000):

        model.train()
        mlp.train()
        weights = OrderedDict(model.named_parameters())

        x = mlp(graph.x)
        x = torch.cat([x[:, :dim], graph.x, x[:, dim:]], dim=1)
        z = model.encode(x, graph.edge_index, weights, inner_loop=True)
        loss = model.class_loss(z[graph.train_labels], graph.labels[graph.train_labels])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for p in model.parameters():
            p.data.clamp_(-0.1, 0.1)

        model.eval()
        mlp.eval()

        with torch.no_grad():
            weights = OrderedDict(model.named_parameters())
            x = mlp(graph.x)
            x = torch.cat([x[:, :dim], graph.x, x[:, dim:]], dim=1)
            z = model.encode(x, graph.edge_index, weights, inner_loop=True)

            f1_, auc_ = model.class_test_all(z[graph.test_labels], graph.labels[graph.test_labels])

        if epoch in [0, 2, 4, 6, 8, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200,
                     1300, 1400, 1500]: print(epoch, (f1, auc), loss.item())

        if f1 < f1_:
            f1 = f1_
        if auc < auc_:
            auc = auc_

    print('the best value: ', (f1, auc))
    return result

















def fused_split(fused_graph,graphs,task):

    fused_graph.x = torch.cat([graphs[name].x for name in graphs.keys()],dim =0).detach()

    if task == 'edge':
        cum = 0
        test_index = []

        mask = torch.tensor(np.ones((fused_graph.adj.shape[0],fused_graph.adj.shape[1]))*2).to(fused_graph.adj)


        for key in graphs.keys():  #### adj的排序是.keys()
            graph = graphs[key]
            
            graph.link_split(0.2, 0.2)
            test_index.append(cum+graph.test_edge_index)

            mask[cum:cum+graph.x.shape[0],cum:cum+graph.x.shape[0]] = 0.0
            mask[graph.train_edge_index[0]+cum,graph.train_edge_index[1]+cum] = 1.0

            cum+=graph.x.shape[0]

     
        adj_tensor = torch.where(mask==2,fused_graph.adj,mask)
                                   
        fused_graph.adj_tensor = adj_tensor

        fused_graph.test_index = test_index

        

        
    
    elif task == 'nmk':
        
        cum = 0
        test_index = []
        mask = torch.tensor(np.ones((fused_graph.adj.shape[0],fused_graph.adj.shape[1]))*2).to(fused_graph.adj)
        for key in graphs.keys():
            graph = graphs[key]
            graph.attr_split2(0.2)
            test_index.append(graph.mask+cum)
            mask[cum:cum+graph.x.shape[0],cum:cum+graph.x.shape[0]] = 0.0
            mask[graph.edge_index[0]+cum,graph.edge_index[1]+cum] = 1.0
            cum+=graph.x.shape[0]


        adj_tensor = torch.where(mask==2,fused_graph.adj,mask)


        
        fused_graph.adj_tensor = adj_tensor
        fused_graph.test_index = test_index

        
        fused_graph.xx = torch.cat([graphs[name].xx for name in graphs.keys()],dim =0)
        
    
    elif task == 'sim':


        cum = 0

        mask = torch.tensor(np.ones((fused_graph.adj.shape[0],fused_graph.adj.shape[1]))*2).to(fused_graph.adj)
        for key in graphs.keys():
            graph = graphs[key]
            mask[cum:cum+graph.x.shape[0],cum:cum+graph.x.shape[0]] = 0.0
            mask[graph.edge_index[0]+cum,graph.edge_index[1]+cum] = 1.0
            cum+=graph.x.shape[0]

        adj_tensor = torch.where(mask==2,fused_graph.adj,mask)

        fused_graph.adj_tensor = adj_tensor



        




    else:

        all_edge_index = []

        cum = 0

        mask = torch.tensor(np.ones((fused_graph.adj.shape[0],fused_graph.adj.shape[1]))*2).to(fused_graph.adj)
        for key in graphs.keys():
            graph = graphs[key]

            all_edge_index.append(graph.edge_index+cum)

            mask[cum:cum+graph.x.shape[0],cum:cum+graph.x.shape[0]] = 0.0
            mask[graph.edge_index[0]+cum,graph.edge_index[1]+cum] = 1.0
            cum+=graph.x.shape[0]

        adj_tensor = torch.where(mask==2,fused_graph.adj,mask)

        fused_graph.adj_tensor = adj_tensor
        fused_graph.original_adj = torch.cat(all_edge_index,dim=1)

        tmp = []
        node_nums = np.cumsum([0]+[graphs[name].x.shape[0] for name in graphs.keys()])
        for i in range(len(node_nums)-1):
            tmp.append(np.arange(node_nums[i],node_nums[i+1])+1)
            tmp[-1][-1] = node_nums[i]
        arr = np.concatenate(tmp)
        
        fused_graph.arr = arr
        
        
        






    return fused_graph,graphs


        

            




def lambda_training(args, task, base_model, base_auxiliary, fused_graph,graphs,device):
    
    models = []
    #fused_graph,graphs = fused_split(fused_graph,graphs,task)
    cum = 0
    for index,key in enumerate(graphs.keys()):
        graph = graphs[key]
        model = deepcopy(base_model),deepcopy(base_auxiliary)
        model_params = []
        model_params.extend(model[0].parameters())
        if model[1] is not None: model_params.extend(model[1].parameters())




        optimizer= torch.optim.Adam([{'params': model_params}], lr=0.0005 ,weight_decay=5e-4)   ########args.model_lr  dgi:0.001
        
        for epoch in range(10): #20
            model[0].train()
            if model[1] is not None:model[1].train()

            weights = OrderedDict(model[0].named_parameters())
            if task == 'edge':
                z_val1 = model[0].encode(x = fused_graph.x, edge_index = None,
                                           weights = weights, edge_weight = fused_graph.adj_tensor.detach())

                loss1 = model[0].recon_loss(z_val1, fused_graph.test_index[index])

                z_val2 = model[0].encode(graph.x.detach(), graph.train_edge_index, weights)
                loss2 = model[0].recon_loss(z_val2, graph.test_edge_index)

            elif task == 'sim':


                
                vice_model = gen_ran_output(model[0])

                embedding = fused_graph.x
                output_positive = model[0].encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor.detach())
                #output_positive = output_positive[cum:cum+graph.x.shape[0],:]

                output_negative = vice_model.encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor.detach())
                #output_negative = output_negative[cum:cum+graph.x.shape[0],:]

                loss1 = loss_cal2(output_positive,output_negative,model[1])


                output_positive = model[0].encode(graph.x, graph.edge_index, weights)
                output_negative = vice_model.encode(graph.x, graph.edge_index, weights)

                loss2 = loss_cal2(output_positive,output_negative,model[1])
                

            elif task == 'nmk':

                z_val = model[0].encode(x = fused_graph.xx, edge_index = None,
                                           weights = weights, edge_weight = fused_graph.adj_tensor.detach())
                z_val1 = model[1](z_val)
                loss1 = torch.nn.MSELoss()(z_val1[fused_graph.test_index[index],int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]], 
                                          graph.x[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]])
                
                z_val = model[0].encode(graph.xx, graph.edge_index, weights)
                z_val2 = model[1](z_val)
                loss2 = torch.nn.MSELoss()(z_val2[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]], 
                                          graph.x[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]])

            else:
                embedding = fused_graph.x
                output_positive = model[0].encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor.detach())
                output_positive = output_positive[cum:cum+graph.x.shape[0],:]

                arr = np.arange(embedding.shape[0])
                arr[cum+graph.x.shape[0]-1]= arr[cum]-1
                arr[cum:cum+graph.x.shape[0]] = arr[cum:cum+graph.x.shape[0]] +1
                

                #arr = fused_graph.arr

                embedding = embedding[arr]

                output_negative = model[0].encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor.detach())
                output_negative = output_negative[cum:cum+graph.x.shape[0],:]
                summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
                discriminator_summary = model[1](summary_emb).T
                positive_score = output_positive @ discriminator_summary
                negative_score = output_negative @ discriminator_summary
                loss1 = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
                negative_score, torch.zeros_like(negative_score))


                output_positive = model[0].encode(graph.x, graph.edge_index, weights)
                arr = torch.arange(graph.x.shape[0]) + 1
                arr[-1:] = torch.arange(1)
                output_negative = model[0].encode(graph.x[arr], graph.edge_index, weights)
                summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
                discriminator_summary = model[1](summary_emb).T
                positive_score = output_positive @ discriminator_summary
                negative_score = output_negative @ discriminator_summary
                loss2 = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
                negative_score, torch.zeros_like(negative_score))

            
            loss = loss1+loss2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        print(loss1,loss2)
        cum+=graph.x.shape[0]


        for p in model[0].parameters(): p.requires_grad = False
        if model[1] is not None: 
            for p in model[1].parameters(): p.requires_grad = False
        models.append(model)


        

    return models

    


def lambda_for_adj(args, task, base_model, base_auxiliary,graphs,match_graph,device):

   
    optimizer= torch.optim.Adam(match_graph.parameters(), lr=0.0001,weight_decay=5e-4) ############0.005  dgi:0.0005


    

    for loop in range(5): #20
        

        adj = match_graph(graphs)

        fused_graph = Graph(None,None)
        fused_graph.adj = adj




        fused_graph,graphs = fused_split(fused_graph,graphs,task)

        print(fused_graph.adj_tensor.sum(),fused_graph.adj_tensor)

        models = lambda_training(args, task, base_model, base_auxiliary, fused_graph,graphs,device)

  
        cum = 0
        loss = 0
        for index,key in enumerate(graphs.keys()):
            model=models[index]
            graph = graphs[key]
            weights = OrderedDict(model[0].named_parameters())
            if task == 'edge':
                z_val1 = model[0].encode(x = fused_graph.x, edge_index = None,
                                           weights = weights, edge_weight = fused_graph.adj_tensor)

                loss1 = model[0].recon_loss(z_val1, fused_graph.test_index[index])

                z_val2 = model[0].encode(graph.x.detach(), graph.train_edge_index, weights)
                loss2 = model[0].recon_loss(z_val2, graph.test_edge_index)
                #original_test = torch.cat(fused_graph.test_index,dim=1)

            elif task == 'sim':
                vice_model = gen_ran_output(model[0])

                embedding = fused_graph.x
                output_positive = model[0].encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor)
                #output_positive = output_positive[cum:cum+graph.x.shape[0],:]

                output_negative = vice_model.encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor)
                #output_negative = output_negative[cum:cum+graph.x.shape[0],:]

                loss1 = loss_cal2(output_positive,output_negative,model[1])


                output_positive = model[0].encode(graph.x, graph.edge_index, weights)
                output_negative = vice_model.encode(graph.x, graph.edge_index, weights)

                loss2 = loss_cal2(output_positive,output_negative,model[1])




            elif task == 'nmk':
                z_val = model[0].encode(x = fused_graph.xx, edge_index = None,
                                           weights = weights, edge_weight = fused_graph.adj_tensor)
                z_val1 = model[1](z_val)
                loss1 = torch.nn.MSELoss()(z_val1[fused_graph.test_index[index],int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]], 
                                          graph.x[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]])
                
                z_val = model[0].encode(graph.xx, graph.edge_index, weights)
                z_val2 = model[1](z_val)
                loss2 = torch.nn.MSELoss()(z_val2[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]], 
                                          graph.x[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]])

            else:
                embedding = fused_graph.x
                output_positive = model[0].encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor)
                output_positive = output_positive[cum:cum+graph.x.shape[0],:]
                
                arr = np.arange(embedding.shape[0])
                arr[cum+graph.x.shape[0]-1]= arr[cum]-1
                arr[cum:cum+graph.x.shape[0]] = arr[cum:cum+graph.x.shape[0]] +1

                #arr = fused_graph.arr

                embedding = embedding[arr]

                output_negative = model[0].encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor)
                output_negative = output_negative[cum:cum+graph.x.shape[0],:]
                summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
                discriminator_summary = model[1](summary_emb).T
                positive_score = output_positive @ discriminator_summary
                negative_score = output_negative @ discriminator_summary
                loss1 = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
                negative_score, torch.zeros_like(negative_score))


                output_positive = model[0].encode(graph.x, graph.edge_index, weights)
                arr = torch.arange(graph.x.shape[0]) + 1
                arr[-1:] = torch.arange(1)
                output_negative = model[0].encode(graph.x[arr], graph.edge_index, weights)
                summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
                discriminator_summary = model[1](summary_emb).T
                positive_score = output_positive @ discriminator_summary
                negative_score = output_negative @ discriminator_summary
                loss2 = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
                negative_score, torch.zeros_like(negative_score))

            cum+=graph.x.shape[0]
            loss+=(loss1+loss2)

        loss +=  2* fused_graph.adj_tensor.mean()  ###################################### 2,5
        optimizer.zero_grad()
        loss.backward()
        print(loss)
        optimizer.step()


    adj = match_graph(graphs)

    if task == 'dgi':

        original_adj = fused_graph.original_adj
        arr = fused_graph.arr
    
        fused_graph = Graph(fused_graph.x,match_graph.get_final_adj(adj,graphs,0.99),True) #0.95
        fused_graph.original_adj = original_adj
        fused_graph.arr = arr
    else:
        fused_graph = Graph(fused_graph.x,match_graph.get_final_adj(adj,graphs,0.99),True) #0.95



    

    return fused_graph

















def MDP_gradient_fuse(args, task, pretrain_model, auxiliary, graphs,fugraphs, optimizer,MLPs,wdiscriminator,optimizer_wd,match_graph, epoch, device, file):


    fused_graph = lambda_for_adj(args, task, pretrain_model, auxiliary,fugraphs, match_graph,device)
    #fused_graph = lambda_for_adj_new(args, task, pretrain_model, auxiliary,fugraphs, match_graph,device)



    task_losses = []

    torch.autograd.set_detect_anomaly(True)

    pretrain_model.train()
    if auxiliary: auxiliary.train()


    for index,name in enumerate(args.train_dataset):
        graphs[name] = graphs[name].transformation(MLPs[name],int(np.sum(args.dims[:index])))
        

    graphs['fuse'] = fused_graph
    if task == 'nmk':graphs['fuse'].original_x = graphs['fuse'].x.clone().detach()


    print(graphs['fuse'].x.shape)
    print(graphs['fuse'].edge_index.shape)

    weights = OrderedDict(pretrain_model.named_parameters())


    

    task_losses = []
    dis_losses = []

    for index,name in enumerate( graphs.keys()):

        Dis_loss = torch.tensor(0.0).to(device)
        graph= graphs[name]
        if name!='fuse':
            

            wdiscriminator_copy = copy.deepcopy(
                train_wdiscriminator(graph, graphs['fuse'], wdiscriminator,
                                     optimizer_wd, batch_d_per_iter=80))

            for p in wdiscriminator_copy.parameters(): p.requires_grad = False
            wdiscriminator_copy.to(device)


            w1 = wdiscriminator_copy( graphs['fuse'].x, graphs['fuse'].edge_index)
            w0 = wdiscriminator_copy(graph.x, graph.edge_index)

            Dis_loss = (torch.mean(w1) - torch.mean(w0))

        



        if task == 'edge':
            if name == 'fuse':
                graph.link_split(0.005, 0.005)
            else:graph.link_split(0.2, 0.2)
            print('graph.train_edge_index,',graph.train_edge_index.shape)
            z_val = pretrain_model.encode(graph.x, graph.train_edge_index, weights)
            loss = pretrain_model.recon_loss(z_val, graph.test_edge_index)

        elif task == 'nmk':
            
            graph.attr_split(0.2)
            z_val = pretrain_model.encode(graph.x, graph.edge_index, weights)
            z_val = auxiliary(z_val)
            if name == 'fuse':
                
                for ind,k in enumerate(args.train_dataset):
                    z_val[ind*args.fuse_scale:(1+ind)*args.fuse_scale,0:int(np.sum(args.dims[:ind]))]=0
                    z_val[ind*args.fuse_scale:(1+ind)*args.fuse_scale,args.dims[ind]+int(np.sum(args.dims[:ind])):]=0
                    graph.original_x[ind*args.fuse_scale:(1+ind)*args.fuse_scale,0:int(np.sum(args.dims[:ind]))]=0
                    graph.original_x[ind*args.fuse_scale:(1+ind)*args.fuse_scale,args.dims[ind]+int(np.sum(args.dims[:ind])):]=0

                    
                loss = torch.nn.MSELoss()(z_val[graph.mask], graph.original_x[graph.mask])
            else:loss = torch.nn.MSELoss()(z_val[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]], graph.original_x[graph.mask])

            
        elif task == 'sim':
            vice_model = gen_ran_output(pretrain_model)
            output_positive = pretrain_model.encode(graph.x, graph.edge_index, weights)
            output_negative = vice_model.encode(graph.x, graph.edge_index, weights)
            if name == 'fuse': loss = loss_cal2(output_positive,output_negative,auxiliary)
            else:loss = loss_cal2(output_positive,output_negative,auxiliary)
    
        else:
            


            if name == 'fuse':
                arr = torch.arange(graph.x.shape[0])#graph.arr
                output_positive = pretrain_model.encode(graph.x, graph.original_adj, weights)
            else:
                output_positive = pretrain_model.encode(graph.x, graph.edge_index, weights)
            

                arr = torch.arange(graph.x.shape[0]) + 1
                arr[-1:] = torch.arange(1)


            
            output_negative = pretrain_model.encode(graph.x[arr], graph.edge_index, weights)
            summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
            discriminator_summary = auxiliary(summary_emb).T
            positive_score = output_positive @ discriminator_summary
            negative_score = output_negative @ discriminator_summary


            if name == 'fuse': loss = 0.1*(torch.nn.BCEWithLogitsLoss()(positive_score, #0.1
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
            negative_score, torch.zeros_like(negative_score)))
            else:loss = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
            negative_score, torch.zeros_like(negative_score))


            
        dis_losses.append(Dis_loss)
        task_losses.append(loss)
        print(Dis_loss.item(),loss.item())

    file.write('{},{},{},'.format(epoch, '-'.join(args.train_dataset), ','.join([str(lo.item()) for lo in task_losses])))
    file.flush()


    if len(task_losses) != 0:
        optimizer.zero_grad()
        stack = torch.stack(task_losses)
        pretrain_batch_loss = stack.mean()+torch.stack(dis_losses).mean()
        pretrain_batch_loss.backward()

        optimizer.step()

    for p in pretrain_model.parameters():
        p.data.clamp_(-0.1, 0.1)

    return pretrain_model, pretrain_batch_loss.item()




    


        

         











    




def fused_split_new(fused_graph,graphs,task):

    #fused_graph.x = torch.cat([graphs[name].x for name in graphs.keys()]+[fused_graph.super_nodes],dim =0).detach()  #这个detach()要放在前面，以后再说
    fused_graph.x = torch.cat([graphs[name].x.detach() for name in graphs.keys()]+[fused_graph.super_nodes],dim =0)  #这个detach()要放在前面，以后再说


    if task == 'edge':
        cum = 0
        test_index = []

        for key in graphs.keys():  #### adj的排序是.keys()
            graph = graphs[key]
            
            graph.link_split(0.2, 0.2)
            test_index.append(cum+graph.test_edge_index)

            fused_graph.adj[cum:cum+graph.x.shape[0],cum:cum+graph.x.shape[0]] = 0.0
            fused_graph.adj[graph.train_edge_index[0]+cum,graph.train_edge_index[1]+cum] = 1.0

            cum+=graph.x.shape[0]

        fused_graph.test_index = test_index
        fused_graph.adj_tensor = fused_graph.adj
    
    elif task == 'nmk':
        
        cum = 0
        test_index = []
        for key in graphs.keys():
            graph = graphs[key]
            graph.attr_split2(0.2)
            test_index.append(graph.mask+cum)
            fused_graph.adj[cum:cum+graph.x.shape[0],cum:cum+graph.x.shape[0]] = 0.0
            fused_graph.adj[graph.edge_index[0]+cum,graph.edge_index[1]+cum] = 1.0
            cum+=graph.x.shape[0]

        
        fused_graph.test_index = test_index
        fused_graph.adj_tensor = fused_graph.adj
        
        fused_graph.xx = torch.cat([graphs[name].xx for name in graphs.keys()]+[fused_graph.super_nodes],dim =0)
        
    
    elif task == 'sim':


        cum = 0
        for key in graphs.keys():
            graph = graphs[key]
            fused_graph.adj[cum:cum+graph.x.shape[0],cum:cum+graph.x.shape[0]] = 0.0
            fused_graph.adj[graph.edge_index[0]+cum,graph.edge_index[1]+cum] = 1.0
            cum+=graph.x.shape[0]

        fused_graph.adj_tensor = fused_graph.adj


    else:
        
        all_edge_index = []
        
        cum = 0

        for key in graphs.keys():
            graph = graphs[key]

            all_edge_index.append(graph.edge_index+cum)

            fused_graph.adj[cum:cum+graph.x.shape[0],cum:cum+graph.x.shape[0]] = 0.0
            fused_graph.adj[graph.edge_index[0]+cum,graph.edge_index[1]+cum] = 1.0
            cum+=graph.x.shape[0]


        fused_graph.adj_tensor = fused_graph.adj
        fused_graph.original_adj = torch.cat(all_edge_index,dim=1)


        tmp = []
        node_nums = np.cumsum([0]+[graphs[name].x.shape[0] for name in graphs.keys()])
        for i in range(len(node_nums)-1):
            tmp.append(np.arange(node_nums[i],node_nums[i+1])+1)
            tmp[-1][-1] = node_nums[i]
        tmp.append(np.arange(node_nums[-1],fused_graph.x.shape[0]))
        arr = np.concatenate(tmp)
        
        fused_graph.arr = arr
    return fused_graph,graphs




def lambda_for_adj_new(args, task, base_model, base_auxiliary,graphs,match_graph,device):

   
    optimizer= torch.optim.Adam(match_graph.parameters(), lr=0.001,weight_decay=5e-4) ############0.005  dgi:0.0005


    

    for loop in range(5): #20      #edge:5
        

        adj = match_graph(graphs)

        fused_graph = Graph(None,None)
        fused_graph.adj = adj
        fused_graph.super_nodes = match_graph.super_nodes
        fused_graph,graphs = fused_split_new(fused_graph,graphs,task)

        print(fused_graph.adj_tensor.sum(),fused_graph.adj_tensor)

        models = lambda_training_new(args, task, base_model, base_auxiliary, fused_graph,graphs,device)

  
        cum = 0
        loss = 0
        for index,key in enumerate(graphs.keys()):
            model=models[index]
            graph = graphs[key]
            weights = OrderedDict(model[0].named_parameters())
            if task == 'edge':
                z_val1 = model[0].encode(x = fused_graph.x, edge_index = None,
                                           weights = weights, edge_weight = fused_graph.adj_tensor)

                loss1 = model[0].recon_loss(z_val1, fused_graph.test_index[index])

                z_val2 = model[0].encode(graph.x.detach(), graph.train_edge_index, weights)
                loss2 = model[0].recon_loss(z_val2, graph.test_edge_index)
                #original_test = torch.cat(fused_graph.test_index,dim=1)

            elif task == 'sim':
                vice_model = gen_ran_output(model[0])

                embedding = fused_graph.x
                output_positive = model[0].encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor)
                #output_positive = output_positive[cum:cum+graph.x.shape[0],:]

                output_negative = vice_model.encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor)
                #output_negative = output_negative[cum:cum+graph.x.shape[0],:]

                loss1 = loss_cal2(output_positive,output_negative,model[1])


                output_positive = model[0].encode(graph.x, graph.edge_index, weights)
                output_negative = vice_model.encode(graph.x, graph.edge_index, weights)

                loss2 = loss_cal2(output_positive,output_negative,model[1])




            elif task == 'nmk':
                z_val = model[0].encode(x = fused_graph.xx, edge_index = None,
                                           weights = weights, edge_weight = fused_graph.adj_tensor)
                z_val1 = model[1](z_val)
                loss1 = torch.nn.MSELoss()(z_val1[fused_graph.test_index[index],int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]], 
                                          graph.x[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]])
                
                z_val = model[0].encode(graph.xx, graph.edge_index, weights)
                z_val2 = model[1](z_val)
                loss2 = torch.nn.MSELoss()(z_val2[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]], 
                                          graph.x[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]])

            else:
                embedding = fused_graph.x
                output_positive = model[0].encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor)
                output_positive = output_positive[cum:cum+graph.x.shape[0],:]
                
                arr = np.arange(embedding.shape[0])
                arr[cum+graph.x.shape[0]-1]= arr[cum]-1
                arr[cum:cum+graph.x.shape[0]] = arr[cum:cum+graph.x.shape[0]] +1

                #arr = fused_graph.arr

                embedding = embedding[arr]

                output_negative = model[0].encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor)
                output_negative = output_negative[cum:cum+graph.x.shape[0],:]
                summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
                discriminator_summary = model[1](summary_emb).T
                positive_score = output_positive @ discriminator_summary
                negative_score = output_negative @ discriminator_summary
                loss1 = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
                negative_score, torch.zeros_like(negative_score))


                output_positive = model[0].encode(graph.x, graph.edge_index, weights)
                arr = torch.arange(graph.x.shape[0]) + 1
                arr[-1:] = torch.arange(1)
                output_negative = model[0].encode(graph.x[arr], graph.edge_index, weights)
                summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
                discriminator_summary = model[1](summary_emb).T
                positive_score = output_positive @ discriminator_summary
                negative_score = output_negative @ discriminator_summary
                loss2 = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
                negative_score, torch.zeros_like(negative_score))

            cum+=graph.x.shape[0]
            loss+=(loss1+loss2)
        optimizer.zero_grad()
        loss.backward()
        print(loss)
        optimizer.step()


    adj = match_graph(graphs)

    
    if task == 'dgi':
        original_adj = fused_graph.original_adj
        fused_graph = Graph(fused_graph.x,match_graph.get_final_adj(adj,0.8),True) #0.95
        fused_graph.original_adj = original_adj
    # elif task == 'sim':
    #     edge_index = match_graph.get_final_adj(adj,0.8)
    #     if (edge_index>fused_graph




    elif task == 'edge':
        
        # adj[:adj.shape[0]-match_graph.proto_num,:adj.shape[0]-match_graph.proto_num]=fused_graph.adj_tensor[:adj.shape[0]-match_graph.proto_num,:adj.shape[0]-match_graph.proto_num]
        # test_index = torch.cat(fused_graph.test_index,1)
        # fused_graph = Graph(fused_graph.x,match_graph.get_final_adj(adj,0.8),True) #0.95 //0.8
        # fused_graph.test_edge_index = test_index
        # fused_graph.train_edge_index = fused_graph.edge_index
        adj_tensor = fused_graph.adj_tensor
        test_index = torch.cat(fused_graph.test_index,1)
        fused_graph = Graph(fused_graph.x,match_graph.get_final_adj(adj,0.8),True) #0.95 //0.8
        fused_graph.test_edge_index = test_index
        adj[:adj.shape[0]-match_graph.proto_num,:adj.shape[0]-match_graph.proto_num]=adj_tensor[:adj.shape[0]-match_graph.proto_num,:adj.shape[0]-match_graph.proto_num]
        fused_graph.train_edge_index = match_graph.get_final_adj(adj,0.8)
    else:
        fused_graph = Graph(fused_graph.x,match_graph.get_final_adj(adj,0.8),True) #0.95
        
    # print('mmmm1',(adj[-match_graph.proto_num:,:4096]>0.8).sum())
    # print('mmmm2',(adj[:4096,-match_graph.proto_num:]>0.8).sum())
    # print('mmmm3',(adj[:4096,:4096]>0.8).sum())
        
    fused_graph.num = fused_graph.x.shape[0]-match_graph.proto_num
    if (fused_graph.edge_index>=fused_graph.num).sum()==0:
        print('000000')
        fused_graph.x = fused_graph.x[:fused_graph.num ]
    return fused_graph





def lambda_training_new(args, task, base_model, base_auxiliary, fused_graph,graphs,device):
    
    models = []
    #fused_graph,graphs = fused_split(fused_graph,graphs,task)
    cum = 0
    for index,key in enumerate(graphs.keys()):
        graph = graphs[key]
        model = deepcopy(base_model),deepcopy(base_auxiliary)
        model_params = []
        model_params.extend(model[0].parameters())
        if model[1] is not None: model_params.extend(model[1].parameters())


        optimizer= torch.optim.Adam([{'params': model_params}], lr=0.001,weight_decay=5e-4)   ########args.model_lr  dgi:0.001
        
        for epoch in range(10): #20
            model[0].train()
            if model[1] is not None:model[1].train()

            weights = OrderedDict(model[0].named_parameters())

            if task == 'edge':
                z_val1 = model[0].encode(x = fused_graph.x, edge_index = None,
                                           weights = weights, edge_weight = fused_graph.adj_tensor.detach())

                loss1 = model[0].recon_loss(z_val1, fused_graph.test_index[index])

                z_val2 = model[0].encode(graph.x.detach(), graph.train_edge_index, weights)
                loss2 = model[0].recon_loss(z_val2, graph.test_edge_index)

            elif task == 'sim':


                
                vice_model = gen_ran_output(model[0])

                embedding = fused_graph.x
                output_positive = model[0].encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor.detach())
                #output_positive = output_positive[cum:cum+graph.x.shape[0],:]

                output_negative = vice_model.encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor.detach())
                #output_negative = output_negative[cum:cum+graph.x.shape[0],:]

                loss1 = loss_cal2(output_positive,output_negative,model[1])


                output_positive = model[0].encode(graph.x, graph.edge_index, weights)
                output_negative = vice_model.encode(graph.x, graph.edge_index, weights)

                loss2 = loss_cal2(output_positive,output_negative,model[1])
                

            elif task == 'nmk':

                z_val = model[0].encode(x = fused_graph.xx, edge_index = None,
                                           weights = weights, edge_weight = fused_graph.adj_tensor.detach())
                z_val1 = model[1](z_val)
                loss1 = torch.nn.MSELoss()(z_val1[fused_graph.test_index[index],int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]], 
                                          graph.x[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]])
                
                z_val = model[0].encode(graph.xx, graph.edge_index, weights)
                z_val2 = model[1](z_val)
                loss2 = torch.nn.MSELoss()(z_val2[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]], 
                                          graph.x[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]])

            else:
                embedding = fused_graph.x
                output_positive = model[0].encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor.detach())
                output_positive = output_positive[cum:cum+graph.x.shape[0],:]

                arr = np.arange(embedding.shape[0])
                arr[cum+graph.x.shape[0]-1]= arr[cum]-1
                arr[cum:cum+graph.x.shape[0]] = arr[cum:cum+graph.x.shape[0]] +1
                

                #arr = fused_graph.arr

                embedding = embedding[arr]

                output_negative = model[0].encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor.detach())
                output_negative = output_negative[cum:cum+graph.x.shape[0],:]
                summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
                discriminator_summary = model[1](summary_emb).T
                positive_score = output_positive @ discriminator_summary
                negative_score = output_negative @ discriminator_summary
                loss1 = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
                negative_score, torch.zeros_like(negative_score))


                output_positive = model[0].encode(graph.x, graph.edge_index, weights)
                arr = torch.arange(graph.x.shape[0]) + 1
                arr[-1:] = torch.arange(1)
                output_negative = model[0].encode(graph.x[arr], graph.edge_index, weights)
                summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
                discriminator_summary = model[1](summary_emb).T
                positive_score = output_positive @ discriminator_summary
                negative_score = output_negative @ discriminator_summary
                loss2 = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
                negative_score, torch.zeros_like(negative_score))

            
            loss = loss1+loss2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        print(loss1,loss2)
        cum+=graph.x.shape[0]


        for p in model[0].parameters(): p.requires_grad = False
        if model[1] is not None: 
            for p in model[1].parameters(): p.requires_grad = False
        models.append(model)


        

    return models






def MDP_gradient_fuse_new(args, task, pretrain_model, auxiliary, graphs,fugraphs, optimizer,MLPs,wdiscriminator,optimizer_wd,match_graph, epoch, device, file):


    #fused_graph = lambda_for_adj(args, task, pretrain_model, auxiliary,fugraphs, match_graph,device)
    fused_graph = lambda_for_adj_new(args, task, pretrain_model, auxiliary,fugraphs, match_graph,device)



    task_losses = []

    torch.autograd.set_detect_anomaly(True)

    pretrain_model.train()
    if auxiliary: auxiliary.train()


    for index,name in enumerate(args.train_dataset):
        graphs[name] = graphs[name].transformation(MLPs[name],int(np.sum(args.dims[:index])))
        

    graphs['fuse'] = fused_graph
    if task == 'nmk':graphs['fuse'].original_x = graphs['fuse'].x.clone().detach()


    print(graphs['fuse'].x.shape)
    print(graphs['fuse'].edge_index.shape)


    weights = OrderedDict(pretrain_model.named_parameters())


    

    task_losses = []
    dis_losses = []

    for index,name in enumerate( graphs.keys()):

        Dis_loss = torch.tensor(0.0).to(device)
        graph= graphs[name]
        if name!='fuse':
            

            wdiscriminator_copy = copy.deepcopy(
                train_wdiscriminator(graph, graphs['fuse'], wdiscriminator,
                                     optimizer_wd, batch_d_per_iter=80))

            for p in wdiscriminator_copy.parameters(): p.requires_grad = False
            wdiscriminator_copy.to(device)


            w1 = wdiscriminator_copy( graphs['fuse'].x, graphs['fuse'].edge_index)
            w0 = wdiscriminator_copy(graph.x, graph.edge_index)

            Dis_loss = (torch.mean(w1) - torch.mean(w0))

        



        if task == 'edge':
            if name != 'fuse':
                graph.link_split(0.2, 0.2)

            print('graph.train_edge_index,',graph.train_edge_index.shape)
            z_val = pretrain_model.encode(graph.x, graph.train_edge_index, weights)
            loss = pretrain_model.recon_loss(z_val, graph.test_edge_index)
            

        elif task == 'nmk':
            
            graph.attr_split(0.2)
            z_val = pretrain_model.encode(graph.x, graph.edge_index, weights)
            z_val = auxiliary(z_val)
            if name == 'fuse':
                
                for ind,k in enumerate(args.train_dataset):
                    z_val[ind*args.fuse_scale:(1+ind)*args.fuse_scale,0:int(np.sum(args.dims[:ind]))]=0
                    z_val[ind*args.fuse_scale:(1+ind)*args.fuse_scale,args.dims[ind]+int(np.sum(args.dims[:ind])):]=0
                    graph.original_x[ind*args.fuse_scale:(1+ind)*args.fuse_scale,0:int(np.sum(args.dims[:ind]))]=0
                    graph.original_x[ind*args.fuse_scale:(1+ind)*args.fuse_scale,args.dims[ind]+int(np.sum(args.dims[:ind])):]=0

                    
                loss = torch.nn.MSELoss()(z_val[graph.mask], graph.original_x[graph.mask])
            else:loss = torch.nn.MSELoss()(z_val[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]], graph.original_x[graph.mask])

            
        elif task == 'sim':
            if name == 'fuse': 
                vice_model = gen_ran_output(pretrain_model)
                output_positive = pretrain_model.encode(graph.x, graph.edge_index, weights)[:graph.num]
                output_negative = vice_model.encode(graph.x, graph.edge_index, weights)[:graph.num]
                loss = loss_cal2(output_positive,output_negative,auxiliary)
            else:
                vice_model = gen_ran_output(pretrain_model)
                output_positive = pretrain_model.encode(graph.x, graph.edge_index, weights)
                output_negative = vice_model.encode(graph.x, graph.edge_index, weights)
                loss = loss_cal2(output_positive,output_negative,auxiliary)
    
        else:
            


            if name == 'fuse':
                arr = torch.arange(graph.x.shape[0])#graph.arr
                output_positive = pretrain_model.encode(graph.x, graph.original_adj, weights)[:graph.num]
                output_negative = pretrain_model.encode(graph.x[arr], graph.edge_index, weights)[:graph.num]
            else:
                output_positive = pretrain_model.encode(graph.x, graph.edge_index, weights)
            

                arr = torch.arange(graph.x.shape[0]) + 1
                arr[-1:] = torch.arange(1)
                output_negative = pretrain_model.encode(graph.x[arr], graph.edge_index, weights)


            
            


            summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
            discriminator_summary = auxiliary(summary_emb).T
            positive_score = output_positive @ discriminator_summary
            negative_score = output_negative @ discriminator_summary


            if name == 'fuse': loss = (torch.nn.BCEWithLogitsLoss()(positive_score, #0.1
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
            negative_score, torch.zeros_like(negative_score)))
            else:loss = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
            negative_score, torch.zeros_like(negative_score))


            
        dis_losses.append(Dis_loss)
        task_losses.append(loss)
        print(Dis_loss.item(),loss.item())

    file.write('{},{},{},{}\n'.format(epoch, '-'.join(args.train_dataset), ','.join([str(lo.item()) for lo in task_losses]),','.join([str(lo.item()) for lo in dis_losses[:4]])))
    file.flush()


    if len(task_losses) != 0:
        
        optimizer.zero_grad()
        pretrain_batch_loss = torch.stack(task_losses).mean() +torch.stack(dis_losses).mean() * 0.01
        pretrain_batch_loss.backward()

        optimizer.step()

    for p in pretrain_model.parameters():
        p.data.clamp_(-0.1, 0.1)

    return pretrain_model, pretrain_batch_loss.item()
