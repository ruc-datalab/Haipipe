from .pipeline import Pipeline
import random
from aipipe.core.config import Config
import numpy as np
import pandas as pd

from collections import defaultdict, OrderedDict, deque
import copy
import sys

import scipy.stats
from scipy.linalg import LinAlgError
import scipy.sparse
import sklearn

import sklearn.metrics
import sklearn.model_selection
from sklearn.utils import check_array
from sklearn.multiclass import OneVsRestClassifier

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

import time
class Environment:
    def __init__(self, config: Config, train=True):
        self.config = config
        self.pipeline = None
        self.column_num = self.config.column_num
        self.state = None
        self.reward = None
        self.action = None
        self.next_state = None
        self.done = False
        self.train = train
        self.lpip_state = None

    def set_pipeline(self, taskid, predictor, metric, train=True):
        if self.pipeline:
            self.pipeline.reset_data()
            del self.pipeline
        self.pipeline = Pipeline(taskid, predictor, metric, self.config, train=train)

    def reset(self, taskid=None, predictor=None, metric=None, default=True):
        if default or taskid is None or predictor is None or metric is None :
            taskid = random.choice(list(self.config.train_index)) 
            if taskid in self.config.train_index:
                predictor_id = random.randrange(1,len(self.config.classifier_predictor_list)+1)
                predictor = [i for i in self.config.classifier_predictor_list if i.id==predictor_id][0]
                metric = [i for i in self.config.metric_list if i.id==self.config.classification_metric_id][0]
            else:
                predictor_id = random.randrange(len(self.config.classifier_predictor_list)+1,len(self.config.classifier_predictor_list)+self.config.regression_dim+1)
                predictor = [i for i in self.config.classifier_predictor_list if i.id==predictor_id][0]
                metric = [i for i in self.config.metric_list if i.id==self.config.regression_metric_id][0]

        self.set_pipeline(taskid, predictor, metric, train=self.train)
        self.column_num = self.config.column_num
        self.lpip_state = self.get_lpip_state()
        self.reward = None
        self.action = None
        self.next_state = None
        self.done = False

    def step(self, step):
        if self.pipeline is None:
            self.reset()
        self.prim_state = self.get_state()
        step_result = self.pipeline.add_step(step)
        if step_result == -1:
            return -1
        elif step_result == 0:
            return 0

        self.next_prim_state = self.get_state()
        self.get_reward()
        self.is_done()
        self.action = step

        return self.prim_state, self.reward, self.next_prim_state, self.done


    def get_data_feature(self):
        inp_data = pd.DataFrame(self.pipeline.train_x)
        if len(self.pipeline.train_y.shape) > 1:
            train_y = self.pipeline.train_y[0]
        else:
            train_y = self.pipeline.train_y
        categorical = list(self.pipeline.train_x.dtypes == object)
        column_info = {}
     
        for i in range(len(inp_data.columns)):
            col = inp_data.iloc[:,i]
            if i >= self.column_num:
                break
            s_s = col

            column_info[i] = {}
            column_info[i]['col_name'] = 'unknown_' + str(i)
            column_info[i]['dtype'] = str(s_s.dtypes) # 1
            column_info[i]['length'] = len(s_s.values) # 2
            column_info[i]['null_ratio'] = s_s.isnull().sum() / len(s_s.values) # 3
            column_info[i]['ctype'] = 1 if inp_data.columns[i] in self.pipeline.num_cols else 2 # 4
            column_info[i]['nunique'] = s_s.nunique() # 5
            column_info[i]['nunique_ratio'] = s_s.nunique() / len(s_s.values) # 6
        
            if 'mean' not in s_s.describe():
                column_info[i]['ctype'] = 2
            if column_info[i]['ctype'] == 1:  # numeric
                
                column_info[i]['mean'] = 0 if np.isnan(s_s.describe()['mean']) or abs(s_s.describe()['mean'])==np.inf else s_s.describe()['mean'] # 7
                column_info[i]['std'] = 0 if np.isnan(s_s.describe()['std']) or abs(s_s.describe()['std'])==np.inf else s_s.describe()['std'] # 8

                column_info[i]['min'] = 0 if np.isnan(s_s.describe()['min']) or abs(s_s.describe()['min'])==np.inf else s_s.describe()['min'] # 9
                column_info[i]['25%'] = 0 if np.isnan(s_s.describe()['25%']) or abs(s_s.describe()['25%'])==np.inf else s_s.describe()['25%']
                column_info[i]['50%'] = 0 if np.isnan(s_s.describe()['50%']) or abs(s_s.describe()['50%'])==np.inf else s_s.describe()['50%']
                column_info[i]['75%'] = 0 if np.isnan(s_s.describe()['75%']) or abs(s_s.describe()['75%'])==np.inf else s_s.describe()['75%']
                column_info[i]['max'] = 0 if np.isnan(s_s.describe()['max']) or abs(s_s.describe()['max'])==np.inf else s_s.describe()['max']
                column_info[i]['median'] = 0 if np.isnan(s_s.median()) or abs(s_s.median())==np.inf else s_s.median()
                if len(s_s.mode()) != 0:
                    column_info[i]['mode'] = 0 if np.isnan(s_s.mode().iloc[0]) or abs(s_s.mode().iloc[0])==np.inf else s_s.mode().iloc[0]
                else:
                    column_info[i]['mode'] = 0
                column_info[i]['mode_ratio'] = 0 if np.isnan(s_s.astype('category').describe().iloc[3] / column_info[i]['length']) or abs(s_s.astype('category').describe().iloc[3] / column_info[i]['length'])==np.inf else s_s.astype('category').describe().iloc[3] / column_info[i]['length']
                column_info[i]['sum'] = 0 if np.isnan(s_s.sum()) or abs(s_s.sum())==np.inf else s_s.sum()
                column_info[i]['skew'] = 0 if np.isnan(s_s.skew()) or abs(s_s.skew())==np.inf else s_s.skew()
                column_info[i]['kurt'] = 0 if np.isnan(s_s.kurt()) or abs(s_s.kurt())==np.inf else s_s.kurt()

            elif column_info[i]['ctype'] == 2:  # category
                column_info[i]['mean'] = 0
                column_info[i]['std'] = 0
                column_info[i]['min'] = 0
                column_info[i]['25%'] = 0
                column_info[i]['50%'] = 0
                column_info[i]['75%'] = 0
                column_info[i]['max'] = 0
                column_info[i]['median'] = 0
                column_info[i]['mode'] = 0
                column_info[i]['mode_ratio'] = 0
                column_info[i]['sum'] = 0
                column_info[i]['skew'] = 0
                column_info[i]['kurt'] = 0
        data_feature = []
        for index in column_info.keys():
            one_column_feature = []
            column_dic = column_info[index]
            for kw in column_dic.keys():
                if kw == 'col_name' or kw == 'content':
                    continue
                elif kw == 'dtype':
                    content = self.config.dtype_dic[column_dic[kw]]
                else:
                    content = column_dic[kw]
                one_column_feature.append(content)
            data_feature.append(one_column_feature)
        if len(column_info) < self.column_num:
            for index in range(len(column_info),self.column_num):
                one_column_feature = np.zeros(self.config.column_feature_dim)
                data_feature.append(one_column_feature)
        data_feature = np.ravel(np.array(data_feature))

        del inp_data
        del column_info
        return data_feature

    def get_lpip_state(self):
        data_feature = self.get_data_feature()
        predictor = np.array([self.pipeline.predictor.id])
        state = np.concatenate((data_feature, predictor))
        return state

    def get_state(self):
        data_feature = self.get_data_feature()
        sequence = np.array(self.pipeline.gsequence)
        predictor = np.array([self.pipeline.predictor.id-1])
        logic_pipeline_id = np.array([self.pipeline.logic_pipeline_id])
        state = np.concatenate((data_feature, sequence, predictor, logic_pipeline_id))
        return state

    def get_reward(self):
        if len(self.pipeline.sequence) < 6:
            self.reward = 0
        else:
            self.end_time = time.time()
            self.reward = self.pipeline.evaluate()
            

    def is_done(self):
        if len(self.pipeline.sequence) < 6:
            self.done = False
        elif len(self.pipeline.sequence) == 6:
            self.done = True
        else:
            return

    def has_nan(self):
        has_num_nan = False
        has_cat_nan = False
        def catch_num(data):
            num_cols = [col for col in data.columns if str(data[col].dtypes) != 'object']
            cat_cols = [col for col in data.columns if col not in num_cols]
            cat_train_x = data[cat_cols]
            num_train_x = data[num_cols]
            return cat_train_x, num_train_x

        
        with pd.option_context('mode.use_inf_as_null', True):
            cat_train_x, num_train_x = catch_num(self.pipeline.train_x)
            cat_test_x, num_test_x = catch_num(self.pipeline.test_x)
            if len(self.pipeline.cat_cols) != 0:
                if cat_train_x.isnull().any().any():
                    has_cat_nan = True
                if cat_test_x.isnull().any().any():
                    has_cat_nan = True
            if len(self.pipeline.num_cols) != 0:
                if num_train_x.isnull().any().any():
                    has_num_nan = True
                if num_test_x.isnull().any().any():
                    has_num_nan = True
        return has_num_nan, has_cat_nan

    def has_cat_cols(self):
        if not len(self.pipeline.cat_cols) == 0:
            return True
        else:
            return False