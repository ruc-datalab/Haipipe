import math
from copy import deepcopy
import numpy as np
from aipipe.core.config import Config
import os
from aipipe.core.env.primitives.primitive import Primitive
from aipipe.core.env.primitives.imputercat import ImputerCatPrim
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class Tester:
    def __init__(self, agent, env, test_pred, config: Config):
        self.agent = agent
        self.env = env
        self.config = config
        self.test_pred = test_pred

        epsilon_final = self.config.epsilon_min
        epsilon_start = self.config.epsilon
        epsilon_decay = self.config.eps_decay
        self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
            -1. * frame_idx / epsilon_decay)

        self.outputdir = self.config.model_dir

        
    def get_five_items_from_pipeline(self, fr, state, reward_dic, seq, taskid, need_save=True):
        tryed_list = []
        epsilon = self.epsilon_by_frame(fr)
        pipeline_index = self.env.pipeline.get_index()
        has_num_nan, has_cat_nan = self.env.has_nan()

        #### predict action by epsilon-greedy
        if self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index]== 'ImputerNum': # imputernum   ----pipeline_index == 0:
            if has_num_nan:
                action, isModel = self.agent.act(state, self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index], tryed_list, epsilon, not_random=True, taskid=taskid)
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
                action, isModel = self.agent.act(state, self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index], tryed_list, epsilon, not_random=True, taskid=taskid)
                temp = self.config.encoders[action]
                step = deepcopy(temp)
            else:
                action = len(self.config.encoders)
                step = Primitive()
        elif self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index] in ['FeaturePreprocessing', 'FeatureEngine', 'FeatureSelection']: # elif pipeline_index in [3,4,5]: # fpreprocessin
            action, isModel = self.agent.act(state, self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index], tryed_list, epsilon, not_random=True, taskid=taskid)
            if self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index] == 'FeaturePreprocessing':
                temp = self.config.fpreprocessings[action]
                step = deepcopy(temp)
            elif self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index] == 'FeatureEngine':
                temp = self.config.fengines[action]
                step = deepcopy(temp)
            elif self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index] == 'FeatureSelection':
                temp = self.config.fselections[action]
                step = deepcopy(temp)


        ### execute action
        step_result = self.env.step(step)
        tryed_list.append(action)
        repeat_time = 0
        ### if execute fail, try again
        while step_result==0 or step_result == 1:
            if self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index]== 'ImputerNum': 
                if has_num_nan:
                    try:
                        action, isModel = self.agent.act(state, self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index], tryed_list, epsilon, not_random=True, taskid=taskid)
                    except:
                        # print('error state:', state)
                        return
                    temp = self.config.imputernums[action]
                    step = deepcopy(temp)
                else:
                    action = len(self.config.imputernums)
                    step = Primitive()
            elif self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index]== 'ImputerCat':  # imputercat
                action = -1
                if has_cat_nan:
                    step = ImputerCatPrim()
                else:
                    step = Primitive()
            elif self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index]== 'Encoder': # encoder
                if self.env.has_cat_cols():
                    action, isModel = self.agent.act(state, self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index], tryed_list, epsilon, not_random=True, taskid=taskid)
                    temp = self.config.encoders[action]
                    step = deepcopy(temp)
                else:
                    action = len(self.config.encoders)
                    step = Primitive()
            elif self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index] in ['FeaturePreprocessing', 'FeatureEngine', 'FeatureSelection']:  # fpreprocessin
                action, isModel = self.agent.act(state, self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index], tryed_list, epsilon, not_random=True, taskid=taskid)
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


        #### get (st, r, st+1, done) for this execute
        state, reward, next_state, done = step_result
        seq.append(step.name)
        state = next_state
        loss = 0

        ### if done, evaluate and save result
        if done:
            self.end_time = self.env.end_time
            self.env.reset(taskid=taskid, default=False, metric=self.config.metric_list[0], predictor=self.config.classifier_predictor_list[self.test_pred])
            self.env.pipeline.logic_pipeline_id, _ = self.agent.act(self.env.lpip_state, 'LogicPipeline', epsilon = self.epsilon_by_frame(0), not_random=True)
            if self.env.pipeline.taskid not in reward_dic:
                reward_dic[self.env.pipeline.taskid] = {'reward':{}, 'seq': {}, 'time': {}}
            reward_dic[self.env.pipeline.taskid]['reward'][self.pre_fr] = reward
            reward_dic[self.env.pipeline.taskid]['seq'][self.pre_fr] = seq
            reward_dic[self.env.pipeline.taskid]['time'][self.pre_fr] = self.end_time-self.start_time
            if need_save:
                np.save(self.config.test_reward_dic_file_name, reward_dic)
        return state, reward_dic, seq, reward, done


 
    def inference(self, data_path):
        self.agent.load_weights(self.outputdir, tag='56000')
        self.pre_fr = 0
        score = 0
        reward_dic = {}
        datasetname = data_path.split("/")[-2]
        for taskid in self.config.classification_task_dic:
            if datasetname == self.config.classification_task_dic[taskid]['dataset']:
                i = taskid
        seq = []
        select_cl = 0
        for cid, cl in enumerate(self.config.classifier_predictor_list):
            if cl.name == self.config.classification_task_dic[i]['model']:
                select_cl = cl
        # print('taskid', i)
        self.start_time = time.time()
        self.env.reset(taskid=i, default=False, metric=self.config.metric_list[0], predictor=select_cl)
        self.env.pipeline.logic_pipeline_id, _ = self.agent.act(self.env.lpip_state, 'LogicPipeline', epsilon = self.epsilon_by_frame(0), not_random=True)
        state = self.env.get_state()
        for fr in range(self.pre_fr + 1, self.pre_fr + 7):
            state, reward_dic, seq, reward, done = self.get_five_items_from_pipeline(fr, state, reward_dic, seq, taskid=i,need_save=False)
        score = reward
        return seq, score
        # return score/14