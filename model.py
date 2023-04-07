import torch
from torch import nn
from haipipe.core.al.config import Config
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable

import numpy as np
torch.set_printoptions(threshold=10)
config = Config()
class ScoreNN(nn.Module):
    def __init__(self, dataset_input_dim, dataset_output_dim, pipeline_input_dim, seq_hidden_size, output_dim, prim_nums, seq_embedding_dim, seq_num_layers):
        super(ScoreNN, self).__init__()
        self.prim_nums = prim_nums
        self.dataset_nn = nn.Sequential(
            nn.Linear(dataset_input_dim, int(dataset_input_dim/2)),
            nn.BatchNorm1d(int(dataset_input_dim/2)),
            nn.LeakyReLU(),
            nn.Linear(int(dataset_input_dim/2), int(dataset_input_dim/2)),
            nn.BatchNorm1d(int(dataset_input_dim/2)),
            nn.LeakyReLU(),
            nn.Linear(int(dataset_input_dim/2), dataset_output_dim),
            nn.BatchNorm1d(dataset_output_dim),
            nn.LeakyReLU(),
            nn.Linear(dataset_output_dim, dataset_output_dim),
            nn.BatchNorm1d(dataset_output_dim),
            nn.LeakyReLU(),
            
        )

        self.seq_embedding = nn.Embedding(prim_nums, seq_embedding_dim)
        self.seq_lstm = nn.LSTM(input_size=seq_embedding_dim, hidden_size=seq_hidden_size,num_layers=seq_num_layers,
                        bias=True,batch_first=False,dropout=0.5,bidirectional=False)

        inp_size = dataset_output_dim + seq_hidden_size*config.seq_len
        self.end_mlp = nn.Sequential(
            nn.Linear(inp_size,int(inp_size / 2)),
            nn.LeakyReLU(),
            nn.Linear(int(inp_size / 2), output_dim),
            nn.LeakyReLU(),
            nn.Linear(output_dim, output_dim),
            nn.LeakyReLU(),
            nn.Linear(output_dim, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1),
            nn.BatchNorm1d(1, track_running_stats = False),
            nn.Tanh(),
        )


    def forward(self, dataset_feature, pipeline_feature):
        dataset_emb = self.dataset_nn(dataset_feature)
        seq_embed_feature = self.seq_embedding(pipeline_feature) # (batch_size , seq_len , seq_embedding_dim)
        seq_embed_feature = seq_embed_feature.permute(1,0,2) # (seq_len , batch_size , seq_embedding_dim)
        seq_hidden_feature,(h_1,c_1) = self.seq_lstm(seq_embed_feature) # (6 , batch_size , seq_hidden_size)
        seq_hidden_feature = seq_hidden_feature.permute(1,0,2) # (batch_size , 6 , seq_hidden_size)
        seq_hidden_feature = torch.flatten(seq_hidden_feature, start_dim=1) # (batch_size , 6*seq_hidden_size)
        input_feature = torch.cat((dataset_emb, seq_hidden_feature), 1)
        return self.end_mlp(input_feature)


    def latent(self, pipeline_feature):

        seq_embed_feature = self.seq_embedding(pipeline_feature) # (batch_size , seq_len , seq_embedding_dim)
        seq_embed_feature = seq_embed_feature.permute(1,0,2) # (seq_len , batch_size , seq_embedding_dim)
        seq_hidden_feature,(h_1,c_1) = self.seq_lstm(seq_embed_feature) # (6 , batch_size , seq_hidden_size)
        seq_hidden_feature = seq_hidden_feature.permute(1,0,2) # (batch_size , 6 , seq_hidden_size)
        seq_hidden_feature = torch.flatten(seq_hidden_feature, start_dim=1) # (batch_size , 6*seq_hidden_size)
        return seq_hidden_feature

    