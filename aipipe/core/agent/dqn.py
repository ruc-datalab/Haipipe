import argparse
import random
import torch
from torch.optim import Adam
from .buffer import ReplayBuffer
from aipipe.core.config import Config
from .model import DQN, RnnDQN
from aipipe.core.trainer import Trainer
import warnings
import numpy as np
import os
warnings.filterwarnings('ignore')
class DQNAgent:
    def __init__(self, process_id, config: Config):
        self.config = config
        self.is_training = True
        self.process_id = process_id
        self.buffer = ReplayBuffer(self.config.max_buff, self.process_id)
        if self.config.use_cuda:
            self.imputernum_model = RnnDQN(self.config.imputernum_action_dim).cuda()
            self.encoder_model = RnnDQN(self.config.encoder_action_dim).cuda()
            self.fpreprocessing_model = RnnDQN(self.config.fpreprocessing_action_dim).cuda()
            self.fengine_model = RnnDQN(self.config.fegine_action_dim).cuda()
            self.fselection_model = RnnDQN(self.config.fselection_action_dim).cuda()
            self.lpipeline_model = DQN(self.config.lpip_state_dim, self.config.lpipeline_action_dim).cuda()
        else:
            self.imputernum_model = RnnDQN(self.config.imputernum_action_dim)
            self.encoder_model = RnnDQN(self.config.encoder_action_dim)
            self.fpreprocessing_model = RnnDQN(self.config.fpreprocessing_action_dim)
            self.fengine_model = RnnDQN(self.config.fegine_action_dim)
            self.fselection_model = RnnDQN(self.config.fselection_action_dim)
            self.lpipeline_model = DQN(self.config.lpip_state_dim, self.config.lpipeline_action_dim)
        
        self.imputernum_model_optim = Adam(self.imputernum_model.parameters(), lr=self.config.learning_rate)
        self.encoder_model_optim = Adam(self.encoder_model.parameters(), lr=self.config.learning_rate)
        self.fpreprocessing_model_optim = Adam(self.fpreprocessing_model.parameters(), lr=self.config.learning_rate)
        self.fengine_model_optim = Adam(self.fengine_model.parameters(), lr=self.config.learning_rate)
        self.fselection_model_optim = Adam(self.fselection_model.parameters(), lr=self.config.learning_rate)
        self.lpipeline_model_optim = Adam(self.lpipeline_model.parameters(), lr=self.config.learning_rate)

        if self.config.use_cuda:
            self.cuda()

    def act(self, state, index, tryed_list=None, epsilon=None, not_random=False, taskid=None):
        if tryed_list is None:
            tryed_list = []
        if epsilon is None: epsilon = self.config.epsilon_min
    
        if index == 'ImputerNum':
            model = self.imputernum_model
            action_dim = self.config.imputernum_action_dim

        elif index == 'Encoder':
            model = self.encoder_model
            action_dim = self.config.encoder_action_dim

        elif index == 'FeaturePreprocessing':
            model = self.fpreprocessing_model
            action_dim = self.config.fpreprocessing_action_dim
        elif index == 'FeatureEngine':
            model = self.fengine_model
            action_dim = self.config.fegine_action_dim
        elif index == 'FeatureSelection':
            model = self.fselection_model
            action_dim = self.config.fselection_action_dim
        elif index == 'LogicPipeline':
            model = self.lpipeline_model
            action_dim = self.config.lpipeline_action_dim
        
        randnum = random.random()
        if ((randnum > epsilon or not self.is_training)) or not_random:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            if self.config.use_cuda:
                state = state.cuda()
            q_value = model.forward(state)

            action_index_list = [i for i in range(action_dim)]
            action_index_list = np.array([i for i in list(action_index_list) if i not in tryed_list])
            q_value_temp = np.array([i for index, i in enumerate(list(q_value[0])) if index not in tryed_list])

            action = action_index_list[q_value_temp.argmax()]
           
        else:
            q_value_temp = np.array([index for index, i in enumerate(np.zeros(action_dim)) if index not in tryed_list])
            action_index_list = [i for i in range(action_dim)]
            action_index_list = list(np.array([i for i in list(action_index_list) if i not in tryed_list]))
            action = random.sample(action_index_list, 1)[0]

        return action, randnum > epsilon 

    def learn_lp(self):
        if os.path.exists(self.config.lp_loss_log_file_name):
            loss_log = list(np.load(self.config.lp_loss_log_file_name, allow_pickle=True))
        else:
            loss_log = []
        s0, a, r = self.buffer.lp_sample(self.config.logic_batch_size)
        a = torch.tensor(np.array(a), dtype=torch.long)
        r = torch.tensor(np.array(r), dtype=torch.float)
        s0 = torch.tensor(s0, dtype=torch.float)
        if self.config.use_cuda: # need modify
            s0 = s0.cuda()
            s1 = s1.cuda()
            a = a.cuda()
            r = r.cuda()
            done = done.cuda()

            q_values = self.lpipeline_model(s0).cuda()
        else:
            result = []
            q_values = self.lpipeline_model(s0)

        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        loss = (q_value - r.detach()).pow(2).mean()
        self.lpipeline_model_optim.zero_grad()
        loss.backward()
        self.lpipeline_model_optim.step()
        loss_log.append(loss.item())
        np.save(self.config.lp_loss_log_file_name, loss_log)
        return loss.item()

    def learning(self, test=False):
        if not test:
            s0, a, r, s1, done, index, logic_pipeline_id = self.buffer.sample(self.config.batch_size)
        else:
            s0, a, r, s1, done, index, logic_pipeline_id = self.buffer.sample(5)
        
        if os.path.exists(self.config.loss_log_file_name):
            loss_log = np.load(self.config.loss_log_file_name, allow_pickle=True).item()
        else:
            loss_log = {0:[], 2:[], 3:[], 4:[], 5:[]}

        s0_imputernum = []
        a_imputernum = []
        r_imputernum = []
        s1_imputernum = []
        done_imputernum = []

        s0_encoder = []
        a_encoder = []
        r_encoder = []
        s1_encoder = []
        done_encoder = []
        sfi_encoder = []

        s0_fpreprocessing = []
        a_fpreprocessing = []
        r_fpreprocessing = []
        s1_fpreprocessing = []
        done_fpreprocessing = []
        sfi_fpreprocessing = []

        s0_fengine = []
        a_fengine = []
        r_fengine = []
        s1_fengine = []
        done_fengine = []
        sfi_fengine = []

        s0_fselection = []
        a_fselection = []
        r_fselection = []
        s1_fselection = []
        done_fselection = []
        sfi_fselection = []
        for i in range(len(index)):
            if index[i] == 'ImputerNum':   
                if a[i] != len(self.config.imputernums):
                    a_imputernum.append(a[i])
                    r_imputernum.append(r[i])
                    s1_imputernum.append(s1[i])
                    done_imputernum.append(done[i])
                    s0_imputernum.append(s0[i])
            if index[i] == 'Encoder':
                if a[i] != len(self.config.encoders):
                    a_encoder.append(a[i])
                    r_encoder.append(r[i])
                    s1_encoder.append(s1[i])
                    done_encoder.append(done[i])
                    s0_encoder.append(s0[i])
                    sfi_encoder.append(logic_pipeline_id[i])
            if index[i] == 'FeaturePreprocessing':
                s0_fpreprocessing.append(s0[i])
                a_fpreprocessing.append(a[i])
                r_fpreprocessing.append(r[i])
                s1_fpreprocessing.append(s1[i])
                done_fpreprocessing.append(done[i])
                sfi_fpreprocessing.append(logic_pipeline_id[i])
            if index[i] == 'FeatureEngine':
                s0_fengine.append(s0[i])
                a_fengine.append(a[i])
                r_fengine.append(r[i])
                s1_fengine.append(s1[i])
                done_fengine.append(done[i])
                sfi_fengine.append(logic_pipeline_id[i])
            if index[i] == 'FeatureSelection':
                s0_fselection.append(s0[i])
                a_fselection.append(a[i])
                r_fselection.append(r[i])
                s1_fselection.append(s1[i])
                done_fselection.append(done[i]) 
                sfi_fselection.append(logic_pipeline_id[i])

        a_imputernum = torch.tensor(np.array(a_imputernum), dtype=torch.long)
        a_encoder = torch.tensor(np.array(a_encoder), dtype=torch.long)
        a_fpreprocessing = torch.tensor(np.array(a_fpreprocessing), dtype=torch.long)
        a_fengine = torch.tensor(np.array(a_fengine), dtype=torch.long)
        a_fselection = torch.tensor(np.array(a_fselection), dtype=torch.long)
        r_imputernum = torch.tensor(np.array(r_imputernum), dtype=torch.float)
        r_encoder = torch.tensor(np.array(r_encoder), dtype=torch.float)
        r_fpreprocessing = torch.tensor(np.array(r_fpreprocessing), dtype=torch.float)
        r_fengine = torch.tensor(np.array(r_fengine), dtype=torch.float)
        r_fselection = torch.tensor(np.array(r_fselection), dtype=torch.float)
        done_imputernum = torch.tensor(np.array(done_imputernum), dtype=torch.float)
        done_encoder = torch.tensor(np.array(done_encoder), dtype=torch.float)
        done_fpreprocessing = torch.tensor(np.array(done_fpreprocessing), dtype=torch.float)
        done_fengine = torch.tensor(np.array(done_fengine), dtype=torch.float)
        done_fselection = torch.tensor(np.array(done_fselection), dtype=torch.float)
        s0_imputernum = torch.tensor(s0_imputernum, dtype=torch.float)
        s0_encoder = torch.tensor(s0_encoder, dtype=torch.float)
        s0_fpreprocessing = torch.tensor(s0_fpreprocessing, dtype=torch.float)
        s0_fengine = torch.tensor(s0_fengine, dtype=torch.float)
        s0_fselection = torch.tensor(s0_fselection, dtype=torch.float)
        s1_imputernum = torch.tensor(s0_imputernum, dtype=torch.float)
        s1_encoder = torch.tensor(s1_encoder, dtype=torch.float)
        s1_fpreprocessing = torch.tensor(s1_fpreprocessing, dtype=torch.float)
        s1_fengine = torch.tensor(s1_fengine, dtype=torch.float)
        s1_fselection = torch.tensor(s1_fselection, dtype=torch.float)

        if self.config.use_cuda: # need modify
            s0 = s0.cuda()
            s1 = s1.cuda()
            a = a.cuda()
            r = r.cuda()
            done = done.cuda()

            a_imputernum = torch.tensor(np.array(a_imputernum), dtype=torch.long).cuda()
            a_encoder = torch.tensor(np.array(a_encoder), dtype=torch.long).cuda()
            a_fpreprocessing = torch.tensor(np.array(a_fpreprocessing), dtype=torch.long).cuda()
            a_fengine = torch.tensor(np.array(a_fengine), dtype=torch.long).cuda()
            a_fselection = torch.tensor(np.array(a_fselection), dtype=torch.long).cuda()
            r_imputernum = torch.tensor(np.array(r_imputernum), dtype=torch.float).cuda()
            r_encoder = torch.tensor(np.array(r_encoder), dtype=torch.float).cuda()
            r_fpreprocessing = torch.tensor(np.array(r_fpreprocessing), dtype=torch.float).cuda()
            r_fengine = torch.tensor(np.array(r_fengine), dtype=torch.float).cuda()
            r_fselection = torch.tensor(np.array(r_fselection), dtype=torch.float).cuda()
            done_imputernum = torch.tensor(np.array(done_imputernum), dtype=torch.float).cuda()
            done_encoder = torch.tensor(np.array(done_encoder), dtype=torch.float).cuda()
            done_fpreprocessing = torch.tensor(np.array(done_fpreprocessing), dtype=torch.float).cuda()
            done_fengine = torch.tensor(np.array(done_fengine), dtype=torch.float).cuda()
            done_fselection = torch.tensor(np.array(done_fselection), dtype=torch.float).cuda()
            s0_imputernum = torch.tensor(s0_imputernum, dtype=torch.float).cuda()
            s0_encoder = torch.tensor(s0_encoder, dtype=torch.float).cuda()
            s0_fpreprocessing = torch.tensor(s0_fpreprocessing, dtype=torch.float).cuda()
            s0_fengine = torch.tensor(s0_fengine, dtype=torch.float).cuda()
            s0_fselection = torch.tensor(s0_fselection, dtype=torch.float).cuda()
            s1_imputernum = torch.tensor(s0_imputernum, dtype=torch.float).cuda()
            s1_encoder = torch.tensor(s1_encoder, dtype=torch.float).cuda()
            s1_fpreprocessing = torch.tensor(s1_fpreprocessing, dtype=torch.float).cuda()
            s1_fengine = torch.tensor(s1_fengine, dtype=torch.float).cuda()
            s1_fselection = torch.tensor(s1_fselection, dtype=torch.float).cuda()

            result = []
            if len(s0_imputernum) > 0:
                q_values_0 = self.imputernum_model(s0_imputernum).cuda()
                next_q_values_0 = self.encoder_model(s1_imputernum).cuda()
                next_q_value_0 = next_q_values_0.max(1)[0]
                q_value_0 = q_values_0.gather(1, a_imputernum.unsqueeze(1)).squeeze(1)

                expected_q_value_0 = r_imputernum + next_q_value_0 * (1 - done_imputernum)
                loss_0 = (q_value_0 - expected_q_value_0.detach()).pow(2).mean()
                self.imputernum_model_optim.zero_grad()
                loss_0.backward()
                self.imputernum_model_optim.step()
                result.append(loss_0.item())
                loss_log[0].append(loss_0.item())
            else:
                result.append(-1)
            if len(s0_encoder) > 0:
                q_values_2 = self.encoder_model(s0_encoder).cuda()
                for logiclineid in sfi_encoder:
                    if logiclineid in [0,1]:
                        next_q_values_2 = self.fpreprocessing_model(s1_encoder).cuda()
                    elif logiclineid in [2,3]:
                        next_q_values_2 = self.fengine_model(s1_encoder).cuda()
                    if logiclineid in [4,5]:
                        next_q_values_2 = self.fselection_model(s1_encoder).cuda()
                next_q_value_2 = next_q_values_2.max(1)[0]
                q_value_2 = q_values_2.gather(1, a_encoder.unsqueeze(1)).squeeze(1)
                expected_q_value_2 = r_encoder + next_q_value_2 * (1 - done_encoder)
                loss_2 = (q_value_2 - expected_q_value_2.detach()).pow(2).mean()
                self.encoder_model_optim.zero_grad()
                loss_2.backward()
                self.encoder_model_optim.step()
                result.append(loss_2.item())
                loss_log[2].append(loss_2.item())
            else:
                result.append(-1)
            if len(s0_fpreprocessing) > 0:
                q_values_3 = self.fpreprocessing_model(s0_fpreprocessing).cuda()
                for logiclineid in sfi_fpreprocessing:
                    if logiclineid in [0,5]:
                        next_q_values_3 = self.fengine_model(s1_fpreprocessing).cuda()
                    elif logiclineid in [1,3]:
                        next_q_values_3 = self.fselection_model(s1_fpreprocessing).cuda()
                    elif logiclineid in [2,4]:
                        next_q_values_3 = self.fpreprocessing_model(s1_fpreprocessing).cuda()
                next_q_value_3 = next_q_values_3.max(1)[0]
                q_value_3 = q_values_3.gather(1, a_fpreprocessing.unsqueeze(1)).squeeze(1)
                expected_q_value_3 = r_fpreprocessing + next_q_value_3 * (1 - done_fpreprocessing)
                loss_3 = (q_value_3 - expected_q_value_3.detach()).pow(2).mean()
                self.fpreprocessing_model_optim.zero_grad()
                loss_3.backward()
                self.fpreprocessing_model_optim.step()
                result.append(loss_3.item())
                loss_log[3].append(loss_3.item())
            else:
                result.append(-1)

            if len(s0_fengine) > 0:
                q_values_4 = self.fengine_model(s0_fengine).cuda()
                for logiclineid in sfi_fengine:
                    if logiclineid in [0,2]:
                        next_q_values_4 = self.fselection_model(s1_fengine).cuda()
                    elif logiclineid in [1,5]:
                        next_q_values_4 = self.fengine_model(s1_fengine).cuda()
                    elif logiclineid in [3,4]:
                        next_q_values_4 = self.fpreprocessing_model(s1_fengine).cuda()
                next_q_value_4 = next_q_values_4.max(1)[0]
                q_value_4 = q_values_4.gather(1, a_fengine.unsqueeze(1)).squeeze(1)
                expected_q_value_4 = r_fengine + next_q_value_4 * (1 - done_fengine)
                loss_4 = (q_value_4 - expected_q_value_4.detach()).pow(2).mean()
                self.fengine_model_optim.zero_grad()
                loss_4.backward()
                self.fengine_model_optim.step()
                result.append(loss_4.item())
                loss_log[4].append(loss_4.item())
            else:
                result.append(-1)
            if len(s0_fselection) > 0:
                q_values_5 = self.fselection_model(s0_fselection).cuda()
                for logiclineid in sfi_fselection:
                    if logiclineid in [0,3]:
                        next_q_values_5 = self.fselection_model(s1_fselection).cuda()
                    elif logiclineid in [1,4]:
                        next_q_values_5 = self.fengine_model(s1_fselection).cuda()
                    elif logiclineid in [2,5]:
                        next_q_values_5 = self.fpreprocessing_model(s1_fselection).cuda()
                next_q_value_5 = next_q_values_5.max(1)[0]     
                q_value_5 = q_values_5.gather(1, a_fselection.unsqueeze(1)).squeeze(1)
                expected_q_value_5 = r_fselection + next_q_value_5 * (1 - done_fselection)
                loss_5 = (q_value_5 - expected_q_value_5.detach()).pow(2).mean()
                self.fselection_model_optim.zero_grad()
                loss_5.backward()
                self.fselection_model_optim.step()
                result.append(loss_5.item())
                loss_log[5].append(loss_5.item())
            else:
                result.append(-1)
            result = tuple(result)
        else:
            result = []
            if len(s0_imputernum) > 0:
                q_values_0 = self.imputernum_model(s0_imputernum)
                next_q_values_0 = self.encoder_model(s1_imputernum)
                next_q_value_0 = next_q_values_0.max(1)[0]
                q_value_0 = q_values_0.gather(1, a_imputernum.unsqueeze(1)).squeeze(1)
                expected_q_value_0 = r_imputernum + next_q_value_0 * (1 - done_imputernum)
                loss_0 = (q_value_0 - expected_q_value_0.detach()).pow(2).mean()
                self.imputernum_model_optim.zero_grad()
                loss_0.backward()
                self.imputernum_model_optim.step()
                result.append(loss_0.item())
                loss_log[0].append(loss_0.item())
            else:
                result.append(-1)
            if len(s0_encoder) > 0:
                q_values_2 = self.encoder_model(s0_encoder)
                for logiclineid in sfi_encoder:
                    if logiclineid in [0,1]:
                        next_q_values_2 = self.fpreprocessing_model(s1_encoder)
                    elif logiclineid in [2,3]:
                        next_q_values_2 = self.fengine_model(s1_encoder)
                    if logiclineid in [4,5]:
                        next_q_values_2 = self.fselection_model(s1_encoder)
                next_q_value_2 = next_q_values_2.max(1)[0]
                q_value_2 = q_values_2.gather(1, a_encoder.unsqueeze(1)).squeeze(1)
                expected_q_value_2 = r_encoder + next_q_value_2 * (1 - done_encoder)

                loss_2 = (q_value_2 - expected_q_value_2.detach()).pow(2).mean()
                self.encoder_model_optim.zero_grad()
                loss_2.backward()
                self.encoder_model_optim.step()
                result.append(loss_2.item())
                loss_log[2].append(loss_2.item())
            else:
                result.append(-1)
            if len(s0_fpreprocessing) > 0:
                q_values_3 = self.fpreprocessing_model(s0_fpreprocessing)
                for logiclineid in sfi_fpreprocessing:
                    if logiclineid in [0,5]:
                        next_q_values_3 = self.fengine_model(s1_fpreprocessing)
                    elif logiclineid in [1,3]:
                        next_q_values_3 = self.fselection_model(s1_fpreprocessing)
                    elif logiclineid in [2,4]:
                        next_q_values_3 = self.fpreprocessing_model(s1_fpreprocessing)
                next_q_value_3 = next_q_values_3.max(1)[0]
                q_value_3 = q_values_3.gather(1, a_fpreprocessing.unsqueeze(1)).squeeze(1)
                expected_q_value_3 = r_fpreprocessing + next_q_value_3 * (1 - done_fpreprocessing)
                loss_3 = (q_value_3 - expected_q_value_3.detach()).pow(2).mean()
                self.fpreprocessing_model_optim.zero_grad()
                loss_3.backward()
                self.fpreprocessing_model_optim.step()
                result.append(loss_3.item())
                loss_log[3].append(loss_3.item())
            else:
                result.append(-1)

            if len(s0_fengine) > 0:
                q_values_4 = self.fengine_model(s0_fengine)
                for logiclineid in sfi_fengine:
                    if logiclineid in [0,2]:
                        next_q_values_4 = self.fselection_model(s1_fengine)
                    elif logiclineid in [1,5]:
                        next_q_values_4 = self.fengine_model(s1_fengine)
                    elif logiclineid in [3,4]:
                        next_q_values_4 = self.fpreprocessing_model(s1_fengine)
                next_q_value_4 = next_q_values_4.max(1)[0]
                q_value_4 = q_values_4.gather(1, a_fengine.unsqueeze(1)).squeeze(1)
                expected_q_value_4 = r_fengine + next_q_value_4 * (1 - done_fengine)
                loss_4 = (q_value_4 - expected_q_value_4.detach()).pow(2).mean()
                self.fengine_model_optim.zero_grad()
                loss_4.backward()
                self.fengine_model_optim.step()
                result.append(loss_4.item())
                loss_log[4].append(loss_4.item())
            else:
                result.append(-1)
            if len(s0_fselection) > 0:
                q_values_5 = self.fselection_model(s0_fselection)
                for logiclineid in sfi_fselection:
                    if logiclineid in [0,3]:
                        next_q_values_5 = self.fselection_model(s1_fselection)
                    elif logiclineid in [1,4]:
                        next_q_values_5 = self.fengine_model(s1_fselection)
                    elif logiclineid in [2,5]:
                        next_q_values_5 = self.fpreprocessing_model(s1_fselection)
                next_q_value_5 = next_q_values_5.max(1)[0]     
                q_value_5 = q_values_5.gather(1, a_fselection.unsqueeze(1)).squeeze(1)
                expected_q_value_5 = r_fselection + next_q_value_5 * (1 - done_fselection)
                loss_5 = (q_value_5 - expected_q_value_5.detach()).pow(2).mean()
                self.fselection_model_optim.zero_grad()
                loss_5.backward()
                self.fselection_model_optim.step()
                result.append(loss_5.item())
                loss_log[5].append(loss_5.item())
            else:
                result.append(-1)
            result = tuple(result)
        np.save(self.config.loss_log_file_name, loss_log)
        return result

    def cuda(self):
        self.model.cuda()

    def load_weights(self, output, tag=''):
        if tag =='':
            if output is None: return
            if os.path.exists(output+'/imputernum_model.pkl'):
                self.imputernum_model.load_state_dict(torch.load(output+'/imputernum_model.pkl'))
            if os.path.exists(output+'/encoder_model.pkl'):
                self.encoder_model.load_state_dict(torch.load(output+'/encoder_model.pkl'))
            if os.path.exists(output+'/fpreprocessing_model.pkl'):
                self.fpreprocessing_model.load_state_dict(torch.load(output+'/fpreprocessing_model.pkl'))
            if os.path.exists(output+'/fengine_model.pkl'):
                self.fengine_model.load_state_dict(torch.load(output+'/fengine_model.pkl'))
            if os.path.exists(output+'/fselection_model.pkl'):
                self.fselection_model.load_state_dict(torch.load(output+'/fselection_model.pkl'))
            if os.path.exists(output+'/logical_pipeline.pkl'):
                self.lpipeline_model.load_state_dict(torch.load(output+'/logical_pipeline.pkl'))
        else:
            if output is None: return
            if os.path.exists(output+'/' + str(tag) + '_imputernum_model.pkl'):
                self.imputernum_model.load_state_dict(torch.load(output+'/' + str(tag) + '_imputernum_model.pkl'))
            if os.path.exists(output+'/' + str(tag) + '_encoder_model.pkl'):
                self.encoder_model.load_state_dict(torch.load(output+'/' + str(tag) + '_encoder_model.pkl'))
            if os.path.exists(output+'/' + str(tag) + '_fpreprocessing_model.pkl'):
                self.fpreprocessing_model.load_state_dict(torch.load(output+'/' + str(tag) + '_fpreprocessing_model.pkl'))
            if os.path.exists(output+'/' + str(tag) + '_fengine_model.pkl'):
                self.fengine_model.load_state_dict(torch.load(output+'/' + str(tag) + '_fengine_model.pkl'))
            if os.path.exists(output+'/' + str(tag) + '_fselection_model.pkl'):
                self.fselection_model.load_state_dict(torch.load(output+'/' + str(tag) + '_fselection_model.pkl'))
            if os.path.exists(output+'/' + str(tag) + '_logical_pipeline.pkl'):
                self.lpipeline_model.load_state_dict(torch.load(output+'/' + str(tag) + '_logical_pipeline.pkl'))

    def save_model(self, output, tag=''):
        if tag == '':
            torch.save(self.imputernum_model.state_dict(), '%s/%s.pkl' % (output, 'imputernum_model'))
            torch.save(self.encoder_model.state_dict(), '%s/%s.pkl' % (output, 'encoder_model'))
            torch.save(self.fpreprocessing_model.state_dict(), '%s/%s.pkl' % (output, 'fpreprocessing_model'))
            torch.save(self.fengine_model.state_dict(), '%s/%s.pkl' % (output, 'fengine_model'))
            torch.save(self.fselection_model.state_dict(), '%s/%s.pkl' % (output, 'fselection_model'))
            torch.save(self.lpipeline_model.state_dict(), '%s/%s.pkl' % (output, 'logical_pipeline'))
        else:
            torch.save(self.imputernum_model.state_dict(), '%s/%s_%s.pkl' % (output, tag, 'imputernum_model'))
            torch.save(self.encoder_model.state_dict(), '%s/%s_%s.pkl' % (output, tag, 'encoder_model'))
            torch.save(self.fpreprocessing_model.state_dict(), '%s/%s_%s.pkl' % (output, tag, 'fpreprocessing_model'))
            torch.save(self.fengine_model.state_dict(), '%s/%s_%s.pkl' % (output, tag, 'fengine_model'))
            torch.save(self.fselection_model.state_dict(), '%s/%s_%s.pkl' % (output, tag,'fselection_model'))
            torch.save(self.lpipeline_model.state_dict(), '%s/%s_%s.pkl' % (output, tag,'logical_pipeline'))



