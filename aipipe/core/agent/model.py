import torch
from torch import nn
from aipipe.core.config import Config

class DQN(nn.Module):
    def __init__(self, num_inputs, actions_dim):
        super(DQN, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(num_inputs, 1907),
            nn.LeakyReLU(),
            nn.Linear(1907, 1907),
            nn.LeakyReLU(),
            # nn.Linear(1907, 1907),
            # nn.LeakyReLU(),
            nn.Linear(1907, 950),
            nn.LeakyReLU(),
            # nn.Linear(950, 950),
            # nn.LeakyReLU(),
            nn.Linear(950, 950),
            nn.LeakyReLU(),
            nn.Linear(950, 387),
            nn.LeakyReLU(),
            nn.Linear(387, 387),
            nn.LeakyReLU(),
            nn.Linear(387, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64,32),
            nn.LeakyReLU(),
            nn.Linear(32,actions_dim),
            nn.Tanh(),
        )

    def forward1(self, x):
        return self.nn(x)
    def forward(self, x):
        return self.nn(x)

class RnnDQN(nn.Module):
    def __init__(self, actions_dim, config: Config = Config()):
        super(RnnDQN, self).__init__()

        self.config = config

        # input dim
        self.data_feature_dim = self.config.data_dim
        self.seq_feature_dim = len(self.config.logic_pipeline_1)

        # seq_embedding_param
        prim_nums = len(set(self.config.imputernums) | set(self.config.encoders) | set(self.config.fpreprocessings) | set(self.config.fengines) | set(self.config.fselections)) + 1 + 1
        # print('prim_nums', prim_nums)
        seq_embedding_dim = self.config.seq_embedding_dim
        # seq_lstm param
        seq_hidden_size = self.config.seq_hidden_size
        seq_num_layers = self.config.seq_num_layers
        # predictor param
        predictor_nums = len(self.config.classifier_predictor_list)
        predictor_embedding_dim = self.config.predictor_embedding_dim
        # lpip param
        lpipeline_nums = len(self.config.lpipelines)
        self.lpipeline_nums = lpipeline_nums
        lpipeline_embedding_dim = self.config.lpipeline_embedding_dim    

        # sequence networks
        self.seq_embedding = nn.Embedding(prim_nums, seq_embedding_dim)
        self.seq_lstm = nn.LSTM(input_size=seq_embedding_dim, hidden_size=seq_hidden_size,num_layers=seq_num_layers,
                        bias=True,batch_first=False,dropout=0.5,bidirectional=False)

        # predictor
        self.predictor_embedding = nn.Embedding(predictor_nums, predictor_embedding_dim)

        # logic pipeline
        self.lpipeline_embedding = nn.Embedding(lpipeline_nums, lpipeline_embedding_dim)

        self.nn = nn.Sequential(
            nn.Linear(self.data_feature_dim + seq_hidden_size * self.seq_feature_dim + predictor_embedding_dim + lpipeline_embedding_dim, self.data_feature_dim + seq_hidden_size + predictor_embedding_dim + lpipeline_embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(self.data_feature_dim + seq_hidden_size + predictor_embedding_dim + lpipeline_embedding_dim, self.data_feature_dim + seq_hidden_size + predictor_embedding_dim + lpipeline_embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(self.data_feature_dim + seq_hidden_size + predictor_embedding_dim + lpipeline_embedding_dim, 1907),
            nn.LeakyReLU(),
            nn.Linear(1907, 1907),
            nn.LeakyReLU(),
            nn.Linear(1907, 950),
            nn.LeakyReLU(),
            nn.Linear(950, 950),
            nn.LeakyReLU(),
            nn.Linear(950, 387),
            nn.LeakyReLU(),
            nn.Linear(387, 387),
            nn.LeakyReLU(),
            nn.Linear(387, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64,32),
            nn.LeakyReLU(),
            nn.Linear(32, actions_dim),
            nn.Tanh()
        )

    def forward1(self, x):
        data_feature = x[:, 0: self.data_feature_dim] # (batch_size , data_dim)
        # print('seq', x[:, self.data_feature_dim: self.data_feature_dim + self.seq_feature_dim])
        seq_feature = x[:, self.data_feature_dim: self.data_feature_dim + self.seq_feature_dim].type(torch.LongTensor) # (batch_size , 6)
        predictor_feature = x[:, self.data_feature_dim + self.seq_feature_dim: self.data_feature_dim + self.seq_feature_dim + 1].type(torch.LongTensor) # (batch_size )
        lpipeline_feature = x[:, self.data_feature_dim + self.seq_feature_dim + 1: self.data_feature_dim+self.seq_feature_dim + 2].type(torch.LongTensor) # (batch_size )

        # print('seq_feature', seq_feature)
        # print('predictor_feature', predictor_feature)
        # print('lpipeline_feature', lpipeline_feature)
        # print('self.lpipeline_nums', self.lpipeline_nums)
        # print(seq_feature)
        seq_embed_feature = self.seq_embedding(seq_feature) # (batch_size , 6 , seq_embedding_dim)
        # batch_first=False
        seq_embed_feature = seq_embed_feature.permute(1,0,2) # (6 , batch_size , seq_embedding_dim)
        seq_hidden_feature,(h_1,c_1) = self.seq_lstm(seq_embed_feature) # (6 , batch_size , seq_hidden_size)
        # batch_first=True
        seq_hidden_feature = seq_hidden_feature.permute(1,0,2) # (batch_size , 6 , seq_hidden_size)
        seq_hidden_feature = torch.flatten(seq_hidden_feature, start_dim=1) # (batch_size , 6*seq_hidden_size)
 
        predictor_embed_feature = self.predictor_embedding(predictor_feature) # (batch_size, 1, predictor_embedding_dim)
        lpipeline_embed_deature = self.lpipeline_embedding(lpipeline_feature) # (batch_size, 1, predictor_embedding_dim)

        predictor_embed_feature = torch.flatten(predictor_embed_feature, start_dim=1)
        lpipeline_embed_deature = torch.flatten(lpipeline_embed_deature, start_dim=1)    

        # print('data_feature_size', data_feature.size())
        # print('seq_hidden_feature', seq_hidden_feature.size())
        # print('predictor_embed_feature', predictor_embed_feature.size())
        # print('lpipeline_embed_deature', lpipeline_embed_deature.size())
        
        input_feature = torch.cat((data_feature, seq_hidden_feature, predictor_embed_feature, lpipeline_embed_deature), 1)
        # print('input_feature', input_feature.size())
        # print('model,', self.data_feature_dim + seq_hidden_size + predictor_embedding_dim + lpipeline_embedding_dim)
        for i in range(len(self.nn)):
            input_feature = self.nn[i](input_feature)
            if i == len(self.nn)-2:
                return input_feature

    def forward(self, x): # x batch_size * state
        data_feature = x[:, 0: self.data_feature_dim] # (batch_size , data_dim)
        # print('seq', x[:, self.data_feature_dim: self.data_feature_dim + self.seq_feature_dim])
        seq_feature = x[:, self.data_feature_dim: self.data_feature_dim + self.seq_feature_dim].type(torch.LongTensor) # (batch_size , 6)
        predictor_feature = x[:, self.data_feature_dim + self.seq_feature_dim: self.data_feature_dim + self.seq_feature_dim + 1].type(torch.LongTensor) # (batch_size )
        lpipeline_feature = x[:, self.data_feature_dim + self.seq_feature_dim + 1: self.data_feature_dim+self.seq_feature_dim + 2].type(torch.LongTensor) # (batch_size )

        # print('seq_feature', seq_feature)
        # print('predictor_feature', predictor_feature)
        # print('lpipeline_feature', lpipeline_feature)
        # print('self.lpipeline_nums', self.lpipeline_nums)
        # print(seq_feature)
        seq_embed_feature = self.seq_embedding(seq_feature) # (batch_size , 6 , seq_embedding_dim)
        # batch_first=False
        seq_embed_feature = seq_embed_feature.permute(1,0,2) # (6 , batch_size , seq_embedding_dim)
        seq_hidden_feature,(h_1,c_1) = self.seq_lstm(seq_embed_feature) # (6 , batch_size , seq_hidden_size)
        # batch_first=True
        seq_hidden_feature = seq_hidden_feature.permute(1,0,2) # (batch_size , 6 , seq_hidden_size)
        seq_hidden_feature = torch.flatten(seq_hidden_feature, start_dim=1) # (batch_size , 6*seq_hidden_size)
 
        predictor_embed_feature = self.predictor_embedding(predictor_feature) # (batch_size, 1, predictor_embedding_dim)
        lpipeline_embed_deature = self.lpipeline_embedding(lpipeline_feature) # (batch_size, 1, predictor_embedding_dim)

        predictor_embed_feature = torch.flatten(predictor_embed_feature, start_dim=1)
        lpipeline_embed_deature = torch.flatten(lpipeline_embed_deature, start_dim=1)    

        # print('data_feature_size', data_feature.size())
        # print('seq_hidden_feature', seq_hidden_feature.size())
        # print('predictor_embed_feature', predictor_embed_feature.size())
        # print('lpipeline_embed_deature', lpipeline_embed_deature.size())
        
        input_feature = torch.cat((data_feature, seq_hidden_feature, predictor_embed_feature, lpipeline_embed_deature), 1)
        # print('input_feature', input_feature.size())
        # print('model,', self.data_feature_dim + seq_hidden_size + predictor_embedding_dim + lpipeline_embedding_dim)
        return self.nn(input_feature)

class CnnDQN(nn.Module):
    def __init__(self, inputs_shape, num_actions):
        super(CnnDQN, self).__init__()

        self.inut_shape = inputs_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(inputs_shape[0], 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.features_size(), 512),
            nn.LeakyReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def features_size(self):
        return self.features(torch.zeros(1, *self.inut_shape)).view(1, -1).size(1)
