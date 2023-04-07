
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy
from model import ScoreNN
from config import Config
import numpy as np
import torch.nn.functional as F

use_gpu = False
config = Config()

loss = {}

def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

def pretrain(iscontinue = 0):
    batch_size = 16
    epoch_num = 10
    learning_rate = 0.01

    dataset_feature_dim = 1900
    pipeline_feature_dim = 30
    iscontinue = iscontinue
    if iscontinue == 0:
        model = ScoreNN(dataset_input_dim =dataset_feature_dim, dataset_output_dim = 300, pipeline_input_dim =pipeline_feature_dim, seq_hidden_size=100, output_dim = 200, seq_embedding_dim = 30, seq_num_layers = 5, prim_nums = len(config.ope2id)+2 )
    else:
        model = torch.load('model/'+str(iscontinue))
    loss_func = torch.nn.L1Loss(reduction='mean')
    opt = torch.optim.Adam(model.parameters(),lr=learning_rate)

    dataset_features = np.load('train_dataset_features.npy', allow_pickle=True)
    pipeline_features = np.load('train_pipeline_features.npy', allow_pickle=True)
    y = np.load('train_y.npy', allow_pickle=True)

    train_dataset = TensorDataset(torch.FloatTensor(dataset_features), torch.LongTensor(pipeline_features), torch.FloatTensor(y))
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    
    scheduler = ReduceLROnPlateau(opt, 'min', verbose=True, patience = 1)

    loss_log = {}
    if use_gpu:
        model.cuda()
        loss_func.cuda()

    for epoch in range(iscontinue + 1, iscontinue + epoch_num + 1):
        print('epoch', epoch, epoch_num)
        loss_log[epoch] = []
        adjust_learning_rate(opt)

        for i,(dataset_feature,pipeline_feature,y) in enumerate(train_loader):
            dataset_feature = Variable(dataset_feature)
            pipeline_feature = Variable(pipeline_feature)
            y = Variable(y)
            if use_gpu:
                dataset_feature,pipeline_feature,y = dataset_feature.cuda(), pipeline_feature.cuda(), y.cuda()
            print('i', i)
            
            out = model(dataset_feature, pipeline_feature)
            out = out.flatten()
            loss = loss_func(out, y)
            print('loss', loss)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_log[epoch].append(loss.detach())

        torch.save(model, 'model/'+str(epoch), )
        np.save('loss.npy', loss_log)
if __name__ == '__main__':
    pretrain(iscontinue = 9)