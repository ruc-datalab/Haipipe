import math
from copy import deepcopy
import numpy as np
from aipipe.core.config import Config
import os
from aipipe.core.env.primitives.primitive import Primitive
from aipipe.core.env.primitives.imputercat import ImputerCatPrim
from aipipe.core.tester import Tester
from aipipe.core.env.enviroment import Environment
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class Trainer:
    def __init__(self, agent, env, test_pred, config: Config):
        self.agent = agent
        self.env = env
        self.config = config
        self.test_pred = test_pred
        self.imputernum_state = None
        self.imputernum_action = None

        epsilon_final = self.config.epsilon_min
        epsilon_start = self.config.epsilon
        epsilon_decay = self.config.eps_decay
        self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
            -1. * frame_idx / epsilon_decay)

        self.outputdir = self.config.model_dir

    def get_five_items_from_pipeline(self, fr, state, reward_dic, one_pip_ismodel, one_pip_sample, result_log, seq):
        epsilon = self.epsilon_by_frame(fr)
        pipeline_index = self.env.pipeline.get_index()
        has_num_nan, has_cat_nan = self.env.has_nan()
        isModel = False
        tryed_list = []

        #### select an action by epsilon-greedy
        if self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index]== 'ImputerNum': # imputernum   ----pipeline_index == 0:
            if has_num_nan:
                try:
                    action, isModel = self.agent.act(state, self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index], tryed_list, epsilon)
                except:
                    return
                temp = self.config.imputernums[action]
                step = deepcopy(temp)
            else:
                action = len(self.config.imputernums)
                step = Primitive()
        elif self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index]== 'ImputerCat': #pipeline_index == 1: # imputercat
            action = -1
            if has_cat_nan:
                step = ImputerCatPrim()
            else:
                step = Primitive()
        elif self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index]== 'Encoder': #pipeline_index == 2: # encoder
            if self.env.has_cat_cols():
                action, isModel = self.agent.act(state, self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index], tryed_list, epsilon)
                temp = self.config.encoders[action]
                step = deepcopy(temp)
            else:
                action = len(self.config.encoders)
                step = Primitive()
        elif self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index] in ['FeaturePreprocessing', 'FeatureEngine', 'FeatureSelection']: # elif pipeline_index in [3,4,5]: # fpreprocessin
            action, isModel = self.agent.act(state, self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index], tryed_list, epsilon)
            if self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index] == 'FeaturePreprocessing':
                temp = self.config.fpreprocessings[action]
                step = deepcopy(temp)
            elif self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index] == 'FeatureEngine':
                temp = self.config.fengines[action]
                step = deepcopy(temp)
            elif self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index] == 'FeatureSelection':
                temp = self.config.fselections[action]
                step = deepcopy(temp)

        #### execute the action
        step_result = self.env.step(step)
        tryed_list.append(action)
        repeat_time = 0
        #### if the action is not valid, then select another action
        while step_result==0 or step_result == 1:
            if self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index] == 'ImputerNum': # imputernum   ----pipeline_index == 0:
                if has_num_nan:
                    try:
                        action, isModel = self.agent.act(state, self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index], tryed_list, epsilon)
                    except:
                        # print('error state:', state)
                        # return
                        action = -1
                        break
                    temp = self.config.imputernums[action]
                    step = deepcopy(temp)
                else:
                    action = len(self.config.imputernums)
                    step = Primitive()
            elif self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index] == 'ImputerCat': #pipeline_index == 1: # imputercat
                action = -1
                if has_cat_nan:
                    step = ImputerCatPrim()
                else:
                    step = Primitive()
            elif self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index]== 'Encoder': #pipeline_index == 2: # encoder
                if self.env.has_cat_cols():
                    action, isModel = self.agent.act(state, self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index], tryed_list, epsilon)
                    temp = self.config.encoders[action]
                    step = deepcopy(temp)
                else:
                    action = len(self.config.encoders)
                    step = Primitive()
            elif self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index] in ['FeaturePreprocessing', 'FeatureEngine', 'FeatureSelection']: # elif pipeline_index in [3,4,5]: # fpreprocessin
                action, isModel = self.agent.act(state, self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index], tryed_list, epsilon)
                if self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index] == 'FeaturePreprocessing':
                    temp = self.config.fpreprocessings[action]
                    step = deepcopy(temp)
                elif self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index] == 'FeatureEngine':
                    temp = self.config.fengines[action]
                    step = deepcopy(temp)
                elif self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index] == 'FeatureSelection':
                    temp = self.config.fselections[action]
                    step = deepcopy(temp)
            if action in tryed_list:
                repeat_time += 1
                continue
            tryed_list.append(action)
            step_result = self.env.step(step)

        seq.append(action)
        state, reward, next_state, done = step_result
        if self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index] not in ['ImputerNum','ImputerCat']:
            self.agent.buffer.add(state, action, reward, next_state, done, self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index], self.env.pipeline.logic_pipeline_id)
        elif self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index] == 'ImputerNum':
            self.imputernum_state = state
            self.imputernum_action = action
        elif self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index] == 'ImputerCat':
            self.agent.buffer.add(self.imputernum_state, self.imputernum_action, reward, next_state, False, 'ImputerNum',self.env.pipeline.logic_pipeline_id)
        state = next_state
        loss = 0

        ### save checkpoint
        if fr % self.config.test_interval == 0:
            self.agent.save_model(self.outputdir, tag=str(fr))
        
        ### update model
        if self.agent.buffer.size() > self.config.batch_size and fr % self.config.train_interval == 0:
            loss_0, loss_2, loss_3, loss_4, loss_5 = self.agent.learning()
            lp_loss = self.agent.learn_lp()
            result_log['learn_time'] += 1

        if done:
            # add sample
            self.agent.buffer.lp_add(self.env.lpip_state, self.env.pipeline.logic_pipeline_id, reward)
            for i in range(len(one_pip_ismodel)):
                if one_pip_ismodel[i] == True:
                    if self.env.pipeline.taskid not in result_log['max_reward']:
                        result_log['max_reward'][self.env.pipeline.taskid] = {}
                    if i not in result_log['max_reward'][self.env.pipeline.taskid]:
                        result_log['max_reward'][self.env.pipeline.taskid][i] = []

                            
                    if i not in result_log['max_action']:
                        result_log['max_action'][i] = []
                    result_log['max_reward'][self.env.pipeline.taskid][i].append(reward)
                    result_log['max_action'][i].append(seq[4])

            if self.env.pipeline.taskid not in result_log['seq_log']:
                result_log['seq_log'][self.env.pipeline.taskid] = []
            result_log['seq_log'][self.env.pipeline.taskid].append((result_log['learn_time'], [i.name for i in self.env.pipeline.sequence], reward, one_pip_ismodel))
            if self.env.pipeline.taskid not in result_log['reward_dic']:
                result_log['reward_dic'][self.env.pipeline.taskid] = []
            result_log['reward_dic'][self.env.pipeline.taskid].append(reward)
            np.save(self.config.result_log_file_name, result_log)
            # self.agent.save_model(self.outputdir, 'best')

            self.env.reset()
            one_pip_ismodel = []
            one_pip_sample = []
            self.env.pipeline.logic_pipeline_id, _ = self.agent.act(self.env.lpip_state, 'LogicPipeline', epsilon = epsilon)
            seq = []

        return reward_dic, one_pip_ismodel, one_pip_sample, result_log, seq, state

    def train(self, pre_fr=0):
        if os.path.exists(self.config.result_log_file_name):
            result_log = np.load(self.config.result_log_file_name, allow_pickle=True).item()
        else:
            result_log = {}
        if 'reward_dic' not in result_log:
            result_log['reward_dic'] = {}
        if 'max_action' not in result_log:
            result_log['max_action'] = {}
        if 'max_reward' not in result_log:
            result_log['max_reward'] = {}
        if 'seq_log' not in result_log:
            result_log['seq_log'] = {}
        if 'learn_time' not in result_log:
            result_log['learn_time'] = 0
        

        self.agent.load_weights(self.outputdir, tag=str(pre_fr))
        one_pip_ismodel = []
        one_pip_sample = []
        seq = []
        self.env.reset()
        self.env.pipeline.logic_pipeline_id, _ = self.agent.act(self.env.lpip_state, 'LogicPipeline', epsilon = self.epsilon_by_frame(0))
        state = self.env.get_state()
        for fr in range(pre_fr + 1, self.config.frames + 1):     
            reward_dic, one_pip_ismodel, one_pip_sample, result_log, seq, state = self.get_five_items_from_pipeline(fr, state, result_log, one_pip_ismodel, one_pip_sample, result_log, seq)
