# -*- coding: utf-8 -*-

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from load_data import load_data
from prepro_data import prepro_data_train, prepro_data_dev, prepro_data_test
from torch.optim.lr_scheduler import StepLR, MultiStepLR 
from transformers import BertTokenizer
from parameter import parse_args

from model_BERT import BERT_DA_WP_GATED 

torch.cuda.empty_cache() #清除GPU缓存
args = parse_args() #加载参数

# 设置随机数种子
def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
setup_seed(args.seed)

# load BERT model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)

# load data tsv file
train_data, dev_data, test_data = load_data()

# get arg_1 arg_2 label from data
train_arg_1, train_arg_2, train_label, = prepro_data_train(train_data)
dev_arg_1, dev_arg_2, dev_label, = prepro_data_dev(dev_data)
test_arg_1, test_arg_2, test_label, = prepro_data_test(test_data)

label_tr = torch.LongTensor(train_label)
label_de = torch.LongTensor(dev_label)
label_te = torch.LongTensor(test_label)
print('Data loaded')

def get_batch(text_data, indices):
    indices_ids = []
    indices_mask = []
    for idx in indices:    
        encode_dict = tokenizer.encode_plus(
            text_data[idx], 
            add_special_tokens=True, 
            padding='max_length',
            max_length=args.len_arg, 
            truncation=True, 
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt')
        indices_ids.append(encode_dict['input_ids'])
        indices_mask.append(encode_dict['attention_mask'])
    batch_ids = torch.cat(indices_ids, dim=0)
    batch_mask = torch.cat(indices_mask, dim=0)

    return batch_ids, batch_mask

# ---------- network ----------
net = BERT_DA_WP_GATED(args).cuda()
optimizer = optim.Adam(net.parameters(), args.lr, betas=(args.momentum, 0.999), weight_decay=args.wd)
criterion = nn.CrossEntropyLoss().cuda()

# creat file to save model and result 
file_out = open("./out.txt", "w")

# learning rate optimizer
if args.stepLR == 1:
    scheduler = MultiStepLR(optimizer, milestones=[args.milestones], gamma=args.gamma)

##################################  epoch  #################################
for epoch in range(args.num_epoch):
    print('Epoch: ', epoch+1)
    print('Epoch: ', epoch+1, file=file_out)
    all_indices = torch.randperm(args.train_size).split(args.batch_size)
    loss_epoch = 0.0
    acc = 0.0
    f1_pred = torch.IntTensor([]).cuda()
    f1_truth = torch.IntTensor([]).cuda()
    start = time.time()
    
    ############################################################################
    ##################################  train  #################################
    ############################################################################
    net.train()
    for i, batch_indices in enumerate(all_indices, 1):
        # get a batch of wordvecs
        batch_arg1, mask_arg1 = get_batch(train_arg_1, batch_indices)        
        batch_arg2, mask_arg2 = get_batch(train_arg_2, batch_indices)
        batch_arg1, batch_arg2, mask_arg1, mask_arg2 = batch_arg1.cuda(), batch_arg2.cuda(), mask_arg1.cuda(), mask_arg2.cuda()
        
        y = label_tr[batch_indices].cuda()
        
        # fed data into network
        out = net(batch_arg1, mask_arg1, batch_arg2, mask_arg2)
        _, pred = torch.max(out, dim=1)
        _, truth = torch.max(y, dim=1)
        
        # calculate acc and F1 
        num_correct = (pred==truth).sum()
        acc += num_correct.item()
        f1_pred = torch.cat((f1_pred, pred), 0)
        f1_truth = torch.cat((f1_truth, truth), 0)
        
        # loss
        loss = criterion(out, truth)
        
        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # report
        loss_epoch += loss.item()
        if i%(3000//args.batch_size) == 0:
            print('loss={:.4f}, acc={:.4f}, F1_score={:.4f}'.format(loss_epoch/(3000//args.batch_size), acc/3000, f1_score(f1_truth.cpu(), f1_pred.cpu(), average='macro')), file = file_out)
            print('loss={:.4f}, acc={:.4f}, F1_score={:.4f}'.format(loss_epoch/(3000//args.batch_size), acc/3000, f1_score(f1_truth.cpu(), f1_pred.cpu(), average='macro')))
            loss_epoch = 0.0
            acc = 0.0
            f1_pred = torch.IntTensor([]).cuda()
            f1_truth = torch.IntTensor([]).cuda()
    end = time.time()
    print('Training Time: {:.2f}s'.format(end-start))
    if args.stepLR == 1:
        scheduler.step()

    ############################################################################
    ##################################  dev  ###################################
    ############################################################################
    all_indices = torch.randperm(args.dev_size).split(args.batch_size)
    acc = 0.0
    f1_pred = torch.IntTensor([]).cuda()
    f1_truth = torch.IntTensor([]).cuda()
    net.eval()
    for batch_indices in all_indices:
        # get a batch of wordvecs
        batch_arg1, mask_arg1 = get_batch(dev_arg_1, batch_indices)        
        batch_arg2, mask_arg2 = get_batch(dev_arg_2, batch_indices)
        batch_arg1, batch_arg2, mask_arg1, mask_arg2 = batch_arg1.cuda(), batch_arg2.cuda(), mask_arg1.cuda(), mask_arg2.cuda()
        
        y = label_de[batch_indices].cuda()

        # fed data into network
        out = net(batch_arg1, mask_arg1, batch_arg2, mask_arg2)
        _, pred = torch.max(out, dim=1)
        _, truth = torch.max(y, dim=1)
        
        # calculate acc and F1 
        num_correct = (pred==truth).sum()
        acc += num_correct.item()
        f1_pred = torch.cat((f1_pred, pred), 0)
        f1_truth = torch.cat((f1_truth, truth), 0)
            
    # report
    print('Dev Acc={:.4f}, Dev F1_score={:.4f}'.format(acc/args.dev_size, f1_score(f1_truth.cpu(), f1_pred.cpu(), average='macro')), file  = file_out)
    print('Dev Acc={:.4f}, Dev F1_score={:.4f}'.format(acc/args.dev_size, f1_score(f1_truth.cpu(), f1_pred.cpu(), average='macro')))
            
    ############################################################################
    ##################################  test  ##################################
    ############################################################################
    all_indices = torch.randperm(args.test_size).split(args.batch_size)
    acc = 0.0
    f1_pred = torch.IntTensor([]).cuda()
    f1_truth = torch.IntTensor([]).cuda()
    net.eval()
    for batch_indices in all_indices:
        # get a batch of wordvecs
        batch_arg1, mask_arg1 = get_batch(test_arg_1, batch_indices)        
        batch_arg2, mask_arg2 = get_batch(test_arg_2, batch_indices)
        batch_arg1, batch_arg2, mask_arg1, mask_arg2 = batch_arg1.cuda(), batch_arg2.cuda(), mask_arg1.cuda(), mask_arg2.cuda()
        
        y = label_te[batch_indices].cuda()
        
        # fed data into network
        out = net(batch_arg1, mask_arg1, batch_arg2, mask_arg2)
        _, pred = torch.max(out, dim=1)
        _, truth = torch.max(y, dim=1)
        
        # calculate acc and F1 
        num_correct = (pred==truth).sum()
        acc += num_correct.item()
        f1_pred = torch.cat((f1_pred, pred), 0)
        f1_truth = torch.cat((f1_truth, truth), 0)
                      
    print('Test Acc={:.4f}, Test F1_score={:.4f}'.format(acc/args.test_size, f1_score(f1_truth.cpu(), f1_pred.cpu(), average='macro')), file = file_out)
    print('Test Acc={:.4f}, Test F1_score={:.4f}'.format(acc/args.test_size, f1_score(f1_truth.cpu(), f1_pred.cpu(), average='macro')))

file_out.close()
