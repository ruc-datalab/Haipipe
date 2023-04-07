"""
input: Graph, Code, Operations
output: new code -> run result
"""
# step 1 : Add one operation on one Edge.
"""
Input: Graph, Code, Operation, Edge
output: new code
"""

from haipipe.core.read_ipynb import ipynb2py
import json
import pickle
from haipipe.core.operation_code import OperationCode, OperationType, EdgeOperationType
from haipipe.core.graph import NotebookGraph, Node, Edge 
import os
import pprint
import copy
import time
from func_timeout import func_set_timeout
import func_timeout

import numpy as np
import gc

cleaning = ["confusion_matrix", "sum", "remove", "drop", "unstack", "reshape", "replace", "drop_duplicates", "groupby", "merge", "reset_index", "join", "sort_values", "concat"]
fit_transform_ope = ["fit_transform", "transform","fit"]


res_data_path = 'hi_res_data/'
val_res_data_path = 'hi_val_res_data/'

def get_all_aiseq():
    seqs = [
    ['Scaler', 'FeatureEngine_simple', 'FeatureSelection'],
    ['FeatureEngine_simple', 'FeatureSelection', 'Scaler'],
    ['FeatureSelection', 'Scaler', 'FeatureEngine_simple'],
    ['FeatureSelection', 'FeatureEngine_simple', 'Scaler'],
    ]
    ai_sequences = []
    for seq in seqs:
        temp_list = []
        for index, lop in enumerate(seq):
            for pop in OperationType[lop]:
                if index == 0:
                    temp_list.append([pop])
                else:
                    for index1,temp_seq in enumerate(temp_list1):
                        temp_seq1 = temp_seq.copy()
                        temp_seq1.append(pop)
                        temp_list.append(temp_seq1)
            temp_list1 = temp_list.copy()
        ai_sequences += temp_list
    # pprint.pprint(ai_sequences)
    ai_sequences_new = [i for i in ai_sequences if len(i) ==3]
    # print(len(ai_sequences_new))
    return ai_sequences_new

class Step():
    def __init__(self, operator, edge_id):
        self.operator = operator
        self.edge_id = edge_id

    
class Merger():
    def __init__(self):
        self.code = ''
        self.origin_code = ''
        self.operations = []
        self.graph = None
        self.all_seq_num_dict = {}
        self.cuted_seq_num_dict = {}

        self.enum_num = []
        self.cut_num = []

    def load_origin_code(self, notebook_id):
        filepath = 'haipipe/core/tmpdata/prenotebook_code/' + str(notebook_id) + '.py'
        self.file_path = filepath
        output_path = filepath
        with open(output_path, 'r') as src_file:
            self.origin_code = src_file.read()
        self.code = self.origin_code
  

    def load_graph(self, notebook_id):     
        filepath = 'haipipe/core/tmpdata/prenotebook_graph/' + str(notebook_id) + ".pkl"
        with open(filepath, 'rb') as f:
            self.graph = pickle.load(f)  


    def enum_adding_haipipe(self, notebook_id, ai_seq):
        """
        Enumeration strategy.
        
        Parameters
        ----------
        notebook_id : str
        ai_seq : list
            The sequence of ML-operations geneerated by AIPipe.
        """
        def is_all_in(ope_id, seq):
            res = True
            need_operations = set()
            for index,ope in enumerate(self.operations):
                if index == ope_id:
                    break
                need_operations.add(ope)
            has_operations = set()
            for item in seq:
                has_operations.add(item.operator)
            if len(has_operations) < len(need_operations):
                return False
            else:
                return True
        
        ### step 1: load HI-pipeline and preprocess AI-pipeline
        self.load_graph(notebook_id)
        self.operations = [ i for i in ai_seq if i != 'blank']

        ### step 2: check the insertable edges of HI-pipeline
        self.need_add_edge_id = []
        added_sorted_id = []
        for edge in self.graph.result_edges:
            if edge.edge_type == 1 and edge.sorted_id not in added_sorted_id and 'sum' not in edge.func_name:
                self.need_add_edge_id.append(edge.edge_id)
                added_sorted_id.append(edge.sorted_id)

        ### step 3: enumerate all possible insertions
        self.all_seq = []
        if len(self.need_add_edge_id) != 0:
            self.all_seq = [[]]
            for ope_index,operation in enumerate(self.operations):
                new_seq = copy.copy(self.all_seq)
                for edge_index in self.need_add_edge_id:
                    for position in ['before', 'after']:
                        step = Step(operation, str(edge_index)+"---"+position)
                        for seq in self.all_seq:
                            last_step_id = 0
                            last_step_position = 'before'
                            if len(seq) != 0:
                                last_step = seq[-1]
                                last_step_id, last_step_position = last_step.edge_id.split("---")
                                last_step_id = int(last_step_id)
                            if edge_index < last_step_id: 
                                continue
                            if last_step_position == 'after' and position == 'before':
                                continue
                            temp_seq = copy.copy(seq)
                            temp_seq.append(step)
                            new_seq.append(temp_seq)
                    
                self.all_seq = new_seq
        if len(self.all_seq) != 0:
            if len(self.all_seq[0]) == 0:
                self.all_seq.pop(0)
   
        end_seqs = []
        def remove_ope(remove_num, seq):
            from itertools import combinations
            import copy
            new_seq = []
            combinations_items = list(combinations(seq, remove_num))
            for remove_items in combinations_items:
                seq_copy = copy.copy(seq)
                for remove_item in remove_items:
                    seq_copy.remove(remove_item)
                new_seq.append(seq_copy)
            return new_seq

        for remove_num in range(0, len(self.operations)):
            end_seqs += remove_ope(remove_num, self.operations)
       
        for seq in end_seqs:
            step_seq = []
            for operation in seq:
                step = Step(operation, 'end')
                step_seq.append(step)
            temp = copy.copy(step_seq)
            self.all_seq.append(temp)
        if len(self.all_seq) not in self.all_seq_num_dict:
            self.all_seq_num_dict[len(self.all_seq)] = 0
        self.all_seq_num_dict[len(self.all_seq)] += 1
        self.enum_num.append(len(self.all_seq))
    
    def pruning_by_rule(self):
        """
        rule0: imputer, encoder is not need.
        rule1: operations must be done after cleaning.
        rule2: no operation can be done between 'fit' and 'transform'.        
        """
        
        def check_rule0(seq):       
            
            for step in seq:
                if step.operator not in self.can_add_operaiton:
                    return False
            return True

        def check_rule1(seq, edge_dict):
            for step in seq:
                if "---" not in step.edge_id:
                    continue
                edge_id, position = step.edge_id.split("---")
                edge_id = int(edge_id)
                after_list = []
                for one_edge_id in edge_dict:
                    if position == 'before':
                        if one_edge_id < edge_id:
                            continue
                        after_list.append(edge_dict[one_edge_id])
                    elif position == 'after':
                        if one_edge_id <= edge_id:
                            continue
                        after_list.append(edge_dict[one_edge_id])
                for operation in after_list:
                    for ope in EdgeOperationType['Cleaning']:
                        if ope in operation:
                            return False
            return True

        def check_rule2(seq, edge_dict):
            for step in seq:
                if "---" not in step.edge_id:
                    continue
                edge_id, position = step.edge_id.split("---")
                edge_id = int(edge_id)
                before_fit_num = 0
                before_transform_num = 0
                for one_edge_id in edge_dict:
                    if position == 'before':
                        if one_edge_id >= edge_id:
                            continue
                        if edge_dict[one_edge_id] == 'fit':
                            before_fit_num += 1
                        if edge_dict[one_edge_id] == 'transform':
                            before_transform_num += 1
                    elif position == 'after':
                        if one_edge_id > edge_id:
                            continue
                        if edge_dict[one_edge_id] == 'fit':
                            before_fit_num += 1
                        if edge_dict[one_edge_id] == 'transform':
                            before_transform_num += 1
                if before_fit_num > 0 and before_transform_num and before_fit_num == before_transform_num:
                    return False
            return True
        def check_rule3(seq, edge_dict):
            found = False
            for one_edge_id in edge_dict:
                if edge_dict[one_edge_id] == 'train_test_split':
                   train_test_split_num  = one_edge_id
                   found = True

            for step in seq:
                if "---" not in step.edge_id:
                    continue
                edge_id, position = step.edge_id.split("---")
                edge_id = int(edge_id)
                before_list = []
                for one_edge_id in edge_dict:
                    if one_edge_id < edge_id:
                        before_list.append(one_edge_id)
                for one_edge_id in before_list:
                    if found:
                        if one_edge_id == train_test_split_num:
                            return False
            return True

        self.cuted_all_seq = []
        self.edge_dict = {}
        self.edge_all_dict = {}
        for edge in self.graph.result_edges:
            if edge.edge_id in self.need_add_edge_id:
                self.edge_dict[edge.edge_id] = edge.func_name
            if 'train_test_split(' in edge.original_code:
                edge.func_name = 'train_test_split'
            self.edge_all_dict[edge.edge_id] = edge.func_name
        cant_add_index = []
        cant_add_index3 = []
        if len(self.edge_dict) == 0:
            self.cuted_all_seq = copy.copy(self.all_seq)
        else:
            for index,seq in enumerate(self.all_seq):
                if check_rule0(seq) and check_rule1(seq, self.edge_dict) and check_rule2(seq, self.edge_dict) and check_rule3(seq, self.edge_all_dict):
                    self.cuted_all_seq.append(seq)
                else:
                    if check_rule0(seq) and check_rule1(seq, self.edge_dict) and check_rule2(seq, self.edge_dict) and check_rule3(seq, self.edge_all_dict) == False:
                        cant_add_index3.append(index)
                    cant_add_index.append(index)
    
        if len(self.cuted_all_seq) not in self.cuted_seq_num_dict:
            self.cuted_seq_num_dict[len(self.cuted_all_seq)] = 0
        self.cuted_seq_num_dict[len(self.cuted_all_seq)] += 1
        self.cut_num.append(len(self.cuted_all_seq))

        return cant_add_index, cant_add_index3
     
     
    def add_one_ope(self, notebook_id, edge_id, ope, position, varible):
        """
        An iteration step of generating HAI-program. --- add one operation code in HI-program to generate HAI-program.
        Parameters
        ----------
        notebook_id: str
        edge_id: int
            which HI-operation to insert AI-operation.
        ope: str
            which AI-operation to be inserted.
        position: str
            'before' or 'after'
        varible: str
            the dataset variable of the HI-operation.
        """

        ### check the dataset variable of the HI-operation.
        with open('haipipe/core/tmpdata/prenotebook_varibles_index/'+str(notebook_id)+'.json', 'r') as f:
            varible_index = json.load(f)
        code_list = self.code.split('\n')
        if ope not in list(OperationCode.keys()):
            return False
    
        if position != 'end': ### if the position of insert is after or before one HI-operation.
            ### generate the code of AI-operation.
            operation_code = OperationCode[ope]['pre_code'] + OperationCode[ope]['code']
            operation_code = operation_code.replace("-[PLACEHOLDER]-", varible) + '\n'
            ### find the line number of the HI-operation to be inserted.
            found = False
            for edge in self.graph.result_edges:
                if edge.edge_id == edge_id:
                    add_position = edge.line_id[0] + self.added_rows
                    found_edge_id = edge.line_id[0]
                    found = True
            
            ### add the code of AI-operation to the HI-program.
            if add_position == 0:
                pre_code_list = []
            else:
                pre_code_list = code_list[0:add_position]
                
            pre_code = ''
            for item in pre_code_list:
                pre_code += item
                pre_code += '\n'
    
            edge_code = code_list[add_position] + '\n'
            if add_position == len(code_list)-1:
                after_code_list = []
            else:
                after_code_list = code_list[add_position+1:]
            if found_edge_id > varible_index['end_idnex'] and found:
                return False
            after_code = ''
            for item in after_code_list:
                after_code += item
                after_code += '\n'
            if position == 'before': # before
                self.code = pre_code + operation_code + edge_code + after_code
            elif position == 'after':
                self.code = pre_code  + edge_code + operation_code + after_code
        else: ## if the position of insert is the end of the data prep process.
            ### generate the code of AI-operation.
            x_varible = varible_index['x_varible']
            operation_code = OperationCode[ope]['pre_code'] + OperationCode[ope]['code']
            operation_code = operation_code.replace("-[PLACEHOLDER]-", x_varible) + '\n'

            ### add the code of AI-operation to the HI-program.
            end_index = varible_index['end_idnex'] + self.added_rows
            code_list = self.code.split("\n")
            pre_code_list = code_list[0:end_index+1]
            after_code_list = code_list[end_index+1:]
            pre_code = ''
            for line in pre_code_list:
                pre_code += line
                pre_code += '\n'
            after_code = ''
            for line in after_code_list:
                after_code += line
                after_code += '\n'
            self.code = pre_code + operation_code + after_code    
            
        self.added_rows += len(operation_code.split('\n'))-1
        return True
   
   
    def merging_one_notebook_rl(self,notebook_id, ai_seq):
        """
        HAI-pipeline enumeration and HAI-program generation.
        Parameters
        ----------
        notebook_id: str
        ai_seq: list
            the AI-sequence to be merged.
        """

        ### enumearte all hai-pipelines.
        res = self.enum_adding_haipipe(notebook_id, ai_seq)

        ### generate all HAI-programs.
        seq_id = 0
        self.cuted_all_seq = self.all_seq

    
        for seq in self.cuted_all_seq:
            self.added_rows = 0
            self.load_origin_code(notebook_id)
            subtime1 = time.time()
            is_all_no_add = True
            for step in seq:
                ### added AI-operation.
                now_ope = step.operator

                ### check the HI-pipeline position for inserting AI-operation.
                if step.edge_id != 'end':
                    edge_id, position = step.edge_id.split("---")
                    edge_id = int(edge_id)
                else:
                    edge_id = 0
                    position = 'end'
                varible = ''

                ### find the variable of the dataset.
                for node in self.graph.result_nodes:
                    for edge_index,node_edge in enumerate(node.children_edges):
                        if node_edge.edge_id == edge_id:
                            varible = node.varible_name
                            assign_num = 0
                            temp_varible = ''
                            for child_index,child_edge in enumerate(node.childrens[edge_index].children_edges):
                                if child_edge.func_name == '-Assign-':
                                    assign_num += 1
                                    temp_varible = node.childrens[edge_index].childrens[child_index].varible_name
                            if assign_num == 1:
                                varible = temp_varible
  
                ### add one AI-operation in the HI-program.
                add_res = self.add_one_ope(notebook_id, edge_id, now_ope, position, varible)
                if add_res == True:
                    is_all_no_add = False

            if is_all_no_add == True:
                continue

            save_seq = []
            for item in seq:
                save_seq.append({"operator": item.operator, "edge_id": item.edge_id})
            self.code = self.code.replace('haipipe/core/tmpdata/prenotebook_res/'+ notebook_id, 'haipipe/core/tmpdata/merge_max_result_rl/' + notebook_id + '/' + str(seq_id))
            subtime3 = time.time()

            if not os.path.exists('haipipe/core/tmpdata/rl_test_merge_code/'+str(notebook_id)):
                os.mkdir('haipipe/core/tmpdata/rl_test_merge_code/'+str(notebook_id))
            with open('haipipe/core/tmpdata/rl_test_merge_code/' + str(notebook_id)+'/'+str(seq_id)+'.json', 'w') as f:
                json.dump({'seq':save_seq, 'code': self.code}, f)
            if not os.path.exists('haipipe/core/tmpdata/rl_test_merge_code_py/'+str(notebook_id)):
                os.mkdir('haipipe/core/tmpdata/rl_test_merge_code_py/'+str(notebook_id))
            with open('haipipe/core/tmpdata/rl_test_merge_code_py/'+str(notebook_id)+'/'+str(seq_id)+'.py', 'w') as f:
                f.write(self.code)
            seq_id += 1

        gc.collect()
    

def transform_one_validation_rl(notebook_id):
    """
    Transform the validation notebook for HAI-programs.
    Parameters
    ----------
    notebook_id: str
    """
    seq_files = os.listdir('haipipe/core/tmpdata/rl_test_merge_code/' + notebook_id)
    for seq_file in seq_files:
        seq_index = seq_file.split('.')[0]
        with open('haipipe/core/tmpdata/rl_test_merge_code/' + notebook_id + '/' + seq_file, 'r') as f:
            seq_code_dict = json.load(f)

        test_code = seq_code_dict['code']
        validation_code = ''
        test_code_list = test_code.split('\n')

        train_test_index = 0
        for index, line in enumerate(test_code_list):
            if '=train_test_split(' in line or ' train_test_split(' in line:
                if '=' not in line:
                    continue
                try:
                    x_train_varible , x_test_varible , y_train_varible , y_test_varible = line.split('=')[0].strip().split(',')
                    train_test_index = index
                except:
                    continue
                train_test_split_code = line.split('=')[1].strip()
                kuohao = train_test_split_code.find('(')
                train_test_split_code = train_test_split_code[kuohao+1:]
                arglist = train_test_split_code.split(",")
                try:
                    x_varible = arglist[0].strip()
                    y_varible = arglist[1].strip()
                except:
                    need_continue = True
            if "model.fit(" in line:
                ours_index = index

        for index, line in enumerate(test_code_list):
            if index == train_test_index-1:
                validation_code += line
                validation_code += '\n'

            if index == train_test_index:
                validation_code += line
                validation_code += '\n'
                try:
                    x_train_varible , x_test_varible , y_train_varible , y_test_varible = line.split('=')[0].strip().split(',')
                    x_train_varible = x_train_varible.strip()
                    not_null_index = line.find(x_train_varible)
                    start_null = line[0:not_null_index]
                except:
                    continue
         
                x_train_varible = x_train_varible.strip()
                x_test_varible = x_test_varible.strip()
                y_train_varible = y_train_varible.strip()
                y_test_varible = y_test_varible.strip()
                x_validation_varible = 'x_validation_varible'
                y_validation_varible = 'y_validation_varible'
    
                cross_validation_code = 'from sklearn.model_selection import cross_val_score\n'
                cross_validation_code += 'cross_score = cross_val_score(model, ' + x_train_varible +', ' + y_train_varible + ',cv=4)\n'
                
            elif 'model.fit(' in line  and index >= ours_index:
                validation_code += '#'+line.replace(x_test_varible, x_validation_varible)
                validation_code += '\n'
            elif 'model.predict(' in line and index >= ours_index:
                validation_code += '#'+line.replace(x_test_varible, x_validation_varible)
                validation_code += '\n'
            
            elif 'score = accuracy_score(' in line and index >= ours_index:
                validation_code += '#'+line.replace(y_test_varible, y_validation_varible)
                validation_code += '\n'
                validation_code += cross_validation_code
            elif 'haipipe/core/tmpdata/merge_max_result_rl/' in line:
                validation_code += '#'+line.replace(y_test_varible, y_validation_varible)
                validation_code += '\n'
                validation_code += line.replace('haipipe/core/tmpdata/merge_max_result_rl/', 'haipipe/core/tmpdata/rl_cross_val_res/').replace(': score }', ': cross_score }')
                validation_code += '\n'
            else:
                validation_code += line
                validation_code += '\n'
        seq_code_dict['validation_code'] = validation_code
    
        if not os.path.exists('haipipe/core/tmpdata/rl_cross_validation_code/' + notebook_id):
            os.mkdir('haipipe/core/tmpdata/rl_cross_validation_code/'+ notebook_id)
        if not os.path.exists('haipipe/core/tmpdata/rl_cross_validation_code_py/' + notebook_id):
            os.mkdir('haipipe/core/tmpdata/rl_cross_validation_code_py/' + notebook_id)
        with open('haipipe/core/tmpdata/rl_cross_validation_code/' + notebook_id + '/' + seq_file, 'w') as f:
            json.dump(seq_code_dict, f)
        with open('haipipe/core/tmpdata/rl_cross_validation_code_py/' + notebook_id + '/' + seq_file.replace('.json', '.py'), 'w') as f:
            f.write(validation_code)

def transform_one_validation_rl_origin(notebook_id):
    """
    Transform the validation notebook for HI-programs.
    Parameters
    ----------
    notebook_id: str
    """
    notebook_py = notebook_id + '.py'
    with open('haipipe/core/tmpdata/prenotebook_code/' + notebook_py, 'r') as f:
        test_code = f.read()
    validation_code = ''
    test_code_list = test_code.split('\n')
    seq_code_dict = {}
    need_continue=False
    train_test_index = 0
    start_time = time.time()
    for index, line in enumerate(test_code_list):
        if '=train_test_split(' in line or ' train_test_split(' in line:
            if '=' not in line:
                continue
            try:
                x_train_varible , x_test_varible , y_train_varible , y_test_varible = line.split('=')[0].strip().split(',')
                train_test_index = index
            except:
                continue
            train_test_split_code = line.split('=')[1].strip()
            kuohao = train_test_split_code.find('(')
            train_test_split_code = train_test_split_code[kuohao+1:]
            arglist = train_test_split_code.split(",")
            try:
                x_varible = arglist[0].strip()
                y_varible = arglist[1].strip()
            except:

                need_continue = True
        if "model.fit(" in line:
            ours_index = index

    for index, line in enumerate(test_code_list):
        if index == train_test_index-1:
            validation_code += line
            validation_code += '\n'
        if index == train_test_index:
            validation_code += line
            validation_code += '\n'
            try:
                x_train_varible , x_test_varible , y_train_varible , y_test_varible = line.split('=')[0].strip().split(',')
                x_train_varible = x_train_varible.strip()
                not_null_index = line.find(x_train_varible)
                start_null = line[0:not_null_index]
            except:
                continue
            x_train_varible = x_train_varible.strip()
            x_test_varible = x_test_varible.strip()
            y_train_varible = y_train_varible.strip()
            y_test_varible = y_test_varible.strip()
            x_validation_varible = 'x_validation_varible'
            y_validation_varible = 'y_validation_varible'
            cross_validation_code = 'from sklearn.model_selection import cross_val_score\n'
            cross_validation_code += 'cross_score = cross_val_score(model, ' + x_train_varible +', ' + y_train_varible + ',cv=4)\n'

        elif 'model.fit(' in line  and index >= ours_index:
            validation_code += '#'+line.replace(x_test_varible, x_validation_varible)
            validation_code += '\n'
        elif 'model.predict(' in line and index >= ours_index:
            validation_code += '#'+line.replace(x_test_varible, x_validation_varible)
            validation_code += '\n'
        elif 'score = accuracy_score(' in line  and index >= ours_index:
            validation_code += '#'+line.replace(y_test_varible, y_validation_varible)
            validation_code += '\n'
            validation_code += cross_validation_code
        elif 'haipipe/core/tmpdata/prenotebook_res/' in line:
            validation_code += '#'+line.replace(y_test_varible, y_validation_varible)
            validation_code += '\n'
            validation_code += line.replace('haipipe/core/tmpdata/prenotebook_res/', 'haipipe/core/tmpdata/rl_cross_val_res/'+notebook_id+'/').replace(': score }', ': cross_score }').replace(notebook_id +'.npy','origin.npy')
            validation_code += '\n'
        else:
            validation_code += line
            validation_code += '\n'
    seq_code_dict['code'] = validation_code
    if not os.path.exists('haipipe/core/tmpdata/rl_cross_validation_code/' + notebook_id):
        os.mkdir('haipipe/core/tmpdata/rl_cross_validation_code/' + notebook_id)
    if not os.path.exists('haipipe/core/tmpdata/rl_cross_validation_code_py/'  + notebook_id):
        os.mkdir('haipipe/core/tmpdata/rl_cross_validation_code_py/' + notebook_id)
    with open('haipipe/core/tmpdata/rl_cross_validation_code/' + notebook_id + '/' + 'origin.json', 'w') as f:
        json.dump(seq_code_dict, f)
    with open('haipipe/core/tmpdata/rl_cross_validation_code_py/' + notebook_id + '/origin.py', 'w') as f:
        f.write(validation_code)
