import json
import numpy as np
import os
from haipipe.core.preprocessing import *
from haipipe.core.graph import profile_hipipe
from haipipe.core.merge import *
from haipipe.core.al.preprocessing import generate_one_test_features
from aipipe.AIPipe import AIPipe
import pandas as pd
import shutil
from haipipe.core.al.finetune import *
from model import ScoreNN

class HAIPipe:
    def __init__(self, notebook_path, data_path, label_index, model):
        """
        notebook_path: the path of the notebook
        data_path: the path of the dataset
        label_index: the index of the label column
        model: the model type
        """
        if not os.path.exists('haipipe/core/tmpdata'):
            os.mkdir('haipipe/core/tmpdata')
        notebook_id = notebook_path.split('/')[-1].split('.')[0]
        self.notebook_id = notebook_id
        self.data_path = data_path
        self.label_index = label_index
        self.model = model
        self.update_info()

    def update_info(self):
        """
        add information of current task and notebook in information files.
        """
        # "notebookinfo.npy" saves (notebook_id, datasetname, model) of each notebook
        info_triple = np.load('haipipe/core/notebookinfo.npy',allow_pickle=True).item()
        # "datasetinfo.json" saves (datasetname, column names, label index) of each notebook
        with open("haipipe/core/datasetinfo.json",'r') as f:
            datasetinfo = json.load(f)
        # "classification_task_dic.json" saves (dataset, data file, label index, model, task) of each task for AIPipe
        with open("aipipe/core/jsons/classification_task_dic.json",'r') as f:
            classification_task_dic = json.load(f)
        # "test_index.json" saves online tasks for AIPipe
        with open("aipipe/core/jsons/test_index.json",'r') as f:
            test_index = json.load(f)

        #### check if current task is in saved information file
        task_id = str(len(classification_task_dic))
        test_index.append(task_id)
        task_name = self.data_path.split('/')[-2]+'_'+self.model+'_'+str(self.label_index)
        exist_task = False
        for task in classification_task_dic:
            if task_name == classification_task_dic[task]['task_name']:
                exist_task = True
                break
        if not exist_task:
            classification_task_dic[task_id] = {}
            classification_task_dic[task_id]['dataset'] = self.data_path.split('/')[-2]
            classification_task_dic[task_id]['csv_file'] = self.data_path.split('/')[-1]
            classification_task_dic[task_id]['label'] = str(self.label_index)
            classification_task_dic[task_id]['model'] = self.model
            classification_task_dic[task_id]['task_name'] = task_name
            
        with open("aipipe/core/jsons/classification_task_dic.json",'w') as f:
            json.dump(classification_task_dic, f)


        #### check if current notebook is in saved information file    
        if self.notebook_id not in info_triple:
            info_triple[self.notebook_id] = {}
            info_triple[self.notebook_id]['dataset_name'] = self.data_path
            info_triple[self.notebook_id]['dataset_id'] = self.data_path
            info_triple[self.notebook_id]['model_type'] = self.model
        if self.notebook_id not in datasetinfo:
            data = pd.read_csv(self.data_path)
            columns = data.columns
            column_index = {}
            for index,col in enumerate(columns):
                column_index[col] = index
            datasetinfo[self.notebook_id] = {}
            datasetinfo[self.notebook_id]['dataset'] = self.data_path
            datasetinfo[self.notebook_id]['column_index'] = column_index
            datasetinfo[self.notebook_id]['index'] = [self.label_index]

        ### update information files
        np.save('haipipe/core/notebookinfo.npy', info_triple)
        with open("haipipe/core/datasetinfo.json",'w') as f:
            json.dump(datasetinfo, f)
        with open("aipipe/core/jsons/classification_task_dic.json",'w') as f:
            json.dump(classification_task_dic, f)
        with open("aipipe/core/jsons/test_index.json",'w') as f:
            json.dump(test_index, f)

    def combine(self):
        """
        HAI-pipeline Enumeration with a given HI-program and a generated Ai-pipeline
        """

        print("\033[0;33;40mstart run HAIPipe:\033[0m")

        ### HI-program -> HI-pipe
        profile_hipipe(self.notebook_id)

        ### mkdir for candidate HAI-programs
        if not os.path.exists('haipipe/core/tmpdata/rl_test_merge_code'):
            os.mkdir('haipipe/core/tmpdata/rl_test_merge_code')
        if not os.path.exists('haipipe/core/tmpdata/rl_test_merge_code_py'):
            os.mkdir('haipipe/core/tmpdata/rl_test_merge_code_py')
        if not os.path.exists('haipipe/core/tmpdata/rl_cross_validation_code'):
            os.mkdir('haipipe/core/tmpdata/rl_cross_validation_code')
        if not os.path.exists('haipipe/core/tmpdata/rl_cross_validation_code_py'):
            os.mkdir('haipipe/core/tmpdata/rl_cross_validation_code_py')

        ### HAI-pipeline enumeration
        merger = Merger()
        merger.merging_one_notebook_rl(self.notebook_id, self.ai_sequence)
        transform_one_validation_rl(self.notebook_id)
        transform_one_validation_rl_origin(self.notebook_id)

    def generate_aipipe(self):
        """
        Generte AIPipe for current task with our RL-based approach.
        """
        print("\033[0;33;40mstart run AIPipe:\033[0m")
        aipipe = AIPipe(self.data_path, self.label_index)
        self.ai_sequence, self.ml_score = aipipe.inference()
        print("\033[0;32;40msucceed\033[0m")
        print('\n')


    def evaluate_hi(self):
        pro = Preprocessing()
        if not os.path.exists('haipipe/core/tmpdata/prenotebook_code'):
            os.mkdir('haipipe/core/tmpdata/prenotebook_code')
        if not os.path.exists('haipipe/core/tmpdata/runned_notebook'):
            os.mkdir('haipipe/core/tmpdata/runned_notebook')
        if not os.path.exists('haipipe/core/tmpdata/prenotebook_res'):
            os.mkdir('haipipe/core/tmpdata/prenotebook_res')
        if not os.path.exists('haipipe/core/tmpdata/prenotebook_varibles_index'):
            os.mkdir('haipipe/core/tmpdata/prenotebook_varibles_index')

        ### index the start and end of data preparation
        res = pro.profiling_code(self.notebook_id, need_remove_model=1)
      
        ### execute HI-pipeline
        print("\033[0;33;40mstart run HIPipe:\033[0m")
        if res == True:
            pro.run_origin_test(self.notebook_id, need_try_again=2)
        else:
            return 
        print("\033[0;32;40msucceed\033[0m")
        print('\n')

    def select_best_hai_by_al(self, K = 20, T = 7):
        """
        HAI-pipeline selection by active learning

        Parameters
        ----------
        K : int
            the number of HAI-pipelines to be selected
        T : int
            the number of training iterations
        """

        ### mkdir for execute results of candidate HAI-programs
        if not os.path.exists("haipipe/core/tmpdata/merge_max_result_rl/"):
            os.mkdir("haipipe/core/tmpdata/merge_max_result_rl/")
        if not os.path.exists("haipipe/core/tmpdata/rl_cross_val_res/"):
            os.mkdir("haipipe/core/tmpdata/rl_cross_val_res/")
        if not os.path.exists("haipipe/core/tmpdata/merge_max_result_rl/" + self.notebook_id):
            os.mkdir("haipipe/core/tmpdata/merge_max_result_rl/" + self.notebook_id)
        if not os.path.exists("haipipe/core/tmpdata/rl_cross_val_res/" + self.notebook_id):
            os.mkdir("haipipe/core/tmpdata/rl_cross_val_res/" + self.notebook_id)
    
        ### generate the features for AL
        generate_one_test_features(self.notebook_id)
        model = torch.load('model/10')
        
        k = int(K / T) # training batch size for each iteration
        V = K - k*T # the number of HAI-pipelines to be selected in the last iteration
        ft = Finetune(self.notebook_id, k, T, V, model)

        ### active learning training
        ft.finetune(i=True, r=True, d=True)

        ### select the best HAI-pipeline
        result, val_score, best_seq_id = ft.get_result()

        self.hai_score = result
        self.hai_index = best_seq_id
        print("\033[0;32;40msucceed\033[0m")
        print('\n')

    def output(self,hai_name,save_fig =False):
        """
        Output the results of HAIPipe
        """
        hi_score = np.load("haipipe/core/tmpdata/prenotebook_res/"+self.notebook_id+'.npy', allow_pickle=True).item()
        if self.hai_index!='origin':
            shutil.copyfile('haipipe/core/tmpdata/rl_test_merge_code_py/'+self.notebook_id +'/'+self.hai_index+'.py', hai_name)
        else:
            shutil.copyfile('haipipe/core/tmpdata/prenotebook_code/'+self.notebook_id +'.py', hai_name)
        print('notebook:',self.notebook_id)
        print('accuracy of HIPipe:',hi_score['accuracy_score'])
        print('accuracy of AIPipe:',self.ml_score)
        print('self.hai_score', self.hai_score)
        print('accuracy of HAIPipe',self.hai_index,':',self.hai_score['accuracy_score'])
     
        shutil.rmtree('haipipe/core/tmpdata/')
        os.mkdir('haipipe/core/tmpdata/')
