import torch
import numpy as np
import random
from dataset import Data
from my_parser import parse
from tqdm import tqdm
from model import ReED
import os
import copy
from evaluate import evaluate

OMP_NUM_THREADS = 8

torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(8)
torch.cuda.empty_cache()

args = parse()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False

data_path = args.data_path + args.dataset_name
dataset = Data(data_path)

my_model = ReED(hid_ent = args.dimension, hid_rel = args.dimension, dim_final = args.dimension, 
                 num_RAMPLayer = args.num_RAMPLayer, num_ent = len(dataset.ent2id), num_rel = 2*len(dataset.rel2id)+1,
                 decoder = args.decoder, aggr_method = args.aggr).cuda()

loss_fn = torch.nn.MarginRankingLoss(margin = args.margin, reduction = 'mean')
optimizer = torch.optim.Adam(my_model.parameters(), lr = args.learning_rate)


train_pos_size = len(dataset.train_pos)
train_neg_size = len(dataset.train_neg)
batch_size = (train_pos_size + train_neg_size)//args.num_batch
train_pos = torch.tensor(dataset.train_pos, dtype = torch.long).cuda()
train_neg = torch.tensor(dataset.train_neg, dtype = torch.long).cuda()
valid_pos = torch.tensor(dataset.valid_pos, dtype = torch.long).cuda()
valid_neg = torch.tensor(dataset.valid_neg, dtype = torch.long).cuda()
test_pos = torch.tensor(dataset.test_pos, dtype = torch.long).cuda()
test_neg = torch.tensor(dataset.test_neg, dtype = torch.long).cuda()
train_pos_inv = torch.fliplr(torch.tensor(dataset.train_pos, dtype = torch.long).cuda())
train_pos_inv[:,1] += len(dataset.rel2id)
train_pos_self = 2*len(dataset.rel2id)*torch.ones((len(dataset.ent2id), 3), dtype = torch.long).cuda()
train_pos_self[:,0] = torch.arange(len(dataset.ent2id))
train_pos_self[:,2] = torch.arange(len(dataset.ent2id))
msg = torch.cat([train_pos[:train_pos_size//2], train_pos_inv[:train_pos_size//2], train_pos_self])

tqdm_bar = tqdm(range(args.num_epoch))
num_ent = len(dataset.ent2id)
num_rel = len(dataset.rel2id)
for epoch in tqdm_bar:

    train = torch.cat([train_pos, train_neg])
    label = torch.cat([torch.ones(train_pos_size).cuda(), -torch.ones(train_neg_size).cuda()])
    rand_idx = torch.randperm(train_pos_size + train_neg_size)
    train = train[rand_idx]
    label = label[rand_idx]
    epoch_loss = 0
    for batch, batch_label in zip(torch.split(train, batch_size), torch.split(label, batch_size)):
        optimizer.zero_grad()
        emb_ent, emb_rel = my_model(msg)
        scores_pos, scores_neg = my_model.score(emb_ent, emb_rel, batch)
        
        loss = loss_fn(scores_pos, scores_neg, batch_label)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for name, param in my_model.named_parameters():
                if not param.requires_grad:
                    continue
                if "proj_msg" not in name and "decoder_" not in name:
                    param_norm = torch.linalg.norm(param)
                    param.div_(param_norm/args.norm)
                else:
                    dim1 = 2 * num_rel + 1
                    dim2 = param.shape[0]//dim1
                    dim3 = param.shape[1]
                    param_div = torch.tile(torch.linalg.norm(param.reshape(dim1, dim2, dim3), dim = (1,2), keepdim = True), (1, dim2, dim3)).reshape(-1, dim3)
                    param.div_(param_div/args.norm)
                        
        epoch_loss += loss.item()
        tqdm_bar.set_description(f"loss {loss.item()}")

emp_loss = evaluate(my_model, msg, train_pos, train_neg, margin = args.margin)
exp_loss = evaluate(my_model, msg, torch.cat([train_pos, valid_pos, test_pos], dim = 0), torch.cat([train_neg, valid_neg, test_neg], dim = 0))
print(f"Empirical Loss:{emp_loss}")
print(f"Expected Loss: {exp_loss}")
print(f"Generalization Error: {exp_loss-emp_loss}")