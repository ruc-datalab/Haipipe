import os
from haipipe.core.read_ipynb import ipynb2py
import ast
import astunparse
import numpy as np
import traceback
import sys
import json
import random
import time
from func_timeout import func_set_timeout
import func_timeout
from haipipe.core.remove_model import remove_model
from haipipe.core.remove_model import remove_model_2
import gc
import shutil

specific_split = 0
dataset_root_path = 'data/dataset/'
accuracy_import_code : dict= {
        'accuracy_score': 'from sklearn.metrics import accuracy_score',
        'f1_score': 'from sklearn.metrics import f1_score',
        'mean_absolute_error': 'from sklearn.metrics import mean_absolute_error',
    }

model_import_code : dict = {
        'LogisticRegression': 'from sklearn.linear_model.logistic import LogisticRegression',
        'RandomForestClassifier': 'from sklearn.ensemble import RandomForestClassifier',
        'KNeighborsClassifier': 'from sklearn.neighbors import KNeighborsClassifier',
        'SVC': 'from sklearn.svm import SVC',
        'DecisionTreeClassifier': 'from sklearn.tree import DecisionTreeClassifier',
    }


error_dict = {}

def exchange_code(code):
    if '.fit_sample(' in code:
        code = code.replace('.fit_sample(', '.fit_resample(')
    if 'from keras.utils import plot_model' in code:
        code = code.replace('from keras.utils import plot_model', "from keras.utils.vis_utils import plot_model")
    if 'from keras.utils import to_categorical' in code:
        code = code.replace('from keras.utils import to_categorical', 'from sklearn.model_selection import train_test_split')
    if 'sklearn.cross_validation' in code:
        code = code.replace('sklearn.cross_validation', 'sklearn.model_selection')
    if 'sklearn.grid_search' in code:
        code = code.replace('sklearn.grid_search', 'sklearn.model_selection')
    if 'from pandas.tools.plotting' in code:
        code = code.replace('from pandas.tools.plotting', 'from pandas.plotting')
    if 'convert_objects(convert_numeric=True)' in code:
        code = code.replace('convert_objects(convert_numeric=True)','apply(pd.to_numeric, errors="ignore")')
    if 'from plotly import plotly' in code:
        code = code.replace('from plotly import plotly','from chart_studio import plotly')
    if 'optimizers.SGD' in code:
        code = code.replace('optimizers.SGD','optimizers.gradient_descent_v2.SGD')
    if 'from sklearn.externals import joblib' in code:
        code = code.replace('from sklearn.externals import joblib','import joblib')
    if 'time.clock()' in code:
        code = code.replace("time.clock()","time.perf_counter()")
    if "plotly.plotly" in code:
        code = code.replace("plotly.plotly","chart_studio.plotly")
    if "sklearn.externals.six" in code:
        code = code.replace("sklearn.externals.six","six")
    if "from keras.utils import to_categorical" in code:
        code = code.replace("from keras.utils import to_categorical","from keras.utils.np_utils import to_categorical")
    if "from sklearn.preprocessing import Imputer" in code:
        code = code.replace("from sklearn.preprocessing import Imputer","from sklearn.impute import SimpleImputer as Imputer")
    if "from keras.optimizers import Adam" in code:
        code = code.replace("from keras.optimizers import Adam","from keras.optimizers import adam_v2 as Adam")
    if "from pandas.tools import plotting" in code:
        code = code.replace("from pandas.tools import plotting","from pandas import plotting")
    if "sklearn.externals.joblib" in code:
        code = code.replace("sklearn.externals.joblib","joblib")
    if ".as_matrix()" in code:
        code = code.replace(".as_matrix()",".values")
    if "jaccard_similarity_score" in code:
        code = code.replace("jaccard_similarity_score","jaccard_score")
    if "get_ipython()." in code:
        code = code.replace("get_ipython().","#get_ipython().")
    if "from_pandas_dataframe(" in code:
        code = code.replace("from_pandas_dataframe(","from_pandas_edgelist(")
    if "Perceptron(n_iter" in code:
        code = code.replace("Perceptron(n_iter","Perceptron(max_iter")
    if "pd.scatter_matrix(" in code:
        code = code.replace("pd.scatter_matrix(", "pd.plotting.scatter_matrix(")
    if "from keras.optimizers import SGD" in code:
        code = code.replace("from keras.optimizers import SGD", "from tensorflow.keras.optimizers import SGD")
    if "SMOTE" in code and "ratio" in code:
        code = code.replace("ratio", "sampling_strategy")
    return code

def load_code(path, test):
    with open(path, 'r') as f:
        dict_ = json.load(f)
        if test:
            code = dict_['code']
        else:
            code = dict_['validation_code']

    code = cleaning_origin(code)
    lines = code.split('\n')
    new_code = ""
    for index,line in enumerate(lines):
        line = exchange_code(line)  
        new_code += line
        new_code +="\n"
    return new_code


def cleaning_origin(code):
    """
    remove redundant annotations of HI-program.
    """
    lines = code.split('\n')
    res_str = ''
    for line in lines:
        line1 = line.strip()
        
        if '#' in line:
            num_p = 0
            num_pp = 0
            index = line.index("#")
            for char in line[0:index]:
                if char == "'":
                    num_p += 1
                if char == '"':
                    num_pp += 1
            if num_p %2 == 0 and num_pp %2 == 0:
                line = line[0:index]
        if len(line) != 0:
            line1 = line.strip()
            if line[-1] == '\\':
                res_str += line[0:-1]
            elif len(line1) > 0:
                if line1[-1] == ',':
                    res_str += line
                else:
                    res_str += line
                    res_str += '\n'
            else:
                res_str += line
                res_str += '\n'

    res_str = clean_brackets(res_str)
    return res_str

def clean_brackets(code):
    """
    fix the brackets errors caused by wrapping.
    """
    lines = code.split('\n')
    left_num = 0
    right_num = 0
    left_z_num = 0
    right_z_num = 0
    str_ = ''
    is_in_beizhu = False
    for line in lines:
        num_p = 0
        num_pp = 0
        if line.count('"""') % 2 == 1 or line.count("'''") %2 == 1:
            is_in_beizhu = not is_in_beizhu
        
        if not is_in_beizhu:
            for index,char in enumerate(line):
                if num_pp % 2 ==0 and num_p % 2 ==0:
                    if char == '#':
                        continue
                if char == '"':
                    if index > 0:
                        if line[index-1] != '\\':
                            if num_p % 2 ==0:
                                num_pp += 1
                    else:
                        if num_p % 2 ==0:
                                num_pp += 1
                if char == "'":
                    if index > 0:
                        if line[index-1] != '\\':
                            if num_pp % 2 ==0:
                                num_p += 1
                    else:
                        if num_pp % 2 ==0:
                                num_p += 1
                if char == '(':
                    if num_pp %2 ==0 and num_p %2 == 0:
                        left_num += 1
                if char == ')':
                    if num_pp %2 ==0 and num_p %2 == 0:
                        right_num += 1
                if char == '[':
                    if num_pp %2 ==0 and num_p %2 == 0:
                        left_z_num += 1
                if char == ']':
                    if num_pp %2 ==0 and num_p %2 == 0:
                        right_z_num += 1
                    
        if left_num == right_num and left_z_num == right_z_num:
            str_ += line
            str_ += '\n'
            left_num =0
            right_num = 0
        else:
            str_ += line
    return str_

class Preprocessing:
    def __init__(self):
        self.origin_code = ''
        self.code = ''
        self.train_test_split_line = None
        self.info_triple = np.load('haipipe/core/notebookinfo.npy', allow_pickle=True).item()
        self.suc = 0
        self.change_suc = 0
        self.train_test_split_no_label = 0
        self.model_index = 0
        self.run_faile = 0
        self.res_data_path = 'haipipe/core/hi_res_data/'
        self.val_res_data_path = 'haipipe/core/hi_val_res_data/'

    def load_code(self, ipypath):
        '''input a ipynb path,
        create a python file from this ipynb file'''
        with open(ipypath, 'r', encoding='utf-8') as f:
            cells = json.load(f)['cells']
        wstr = ''
        for i in range(len(cells)):
            if cells[i]['cell_type'] == 'markdown':
                for j in cells[i]['source']:
                    wstr += ('# ' + j)
                wstr += '\n\n'
            elif cells[i]['cell_type'] == 'code':
                if type(cells[i]['source']).__name__ == 'str':
                    code_list = cells[i]['source'].split("\n")
                    for line in code_list:
                        if len(line) != 0:
                            if line[0] == '%':
                                line = '#' + line[0]
                        wstr += line
                        wstr += '\n'
                    wstr += '\n'
                else:
                    for line in cells[i]['source']:
                        if len(line) != 0:
                            if line[-1] == '\n':
                                line = line[0:-1]
                        if len(line) != 0:
                            if line[0] == '%':
                                line = '#' + line[0]
                        wstr += line
                        wstr += '\n'
                    wstr += '\n'
        return wstr

    def load_origin_code(self, notebook_id, need_remove_model):
        """
        Load the code from the notebook.
        """
        self.train_test_split_index = None
        self.code = ''
        filepath = "data/notebook/" + str(notebook_id) +'.ipynb'
        self.file_path = filepath
        self.notebook_id = notebook_id
        if '.ipynb' in filepath:
            self.origin_code = self.load_code(filepath)
        else:
            output_path = filepath
            with open(output_path, 'r') as src_file:
                self.origin_code = src_file.read()
        self.origin_code = cleaning_origin(self.origin_code)
        if need_remove_model==1:
            try:
                temp = remove_model(self.origin_code)
                self.origin_code = temp
            except:
                pass
        if need_remove_model==2:
            try:
                temp = remove_model_2(self.origin_code)
                self.origin_code = temp
            except:
                pass

        self.code = self.origin_code

    def find_train_test_split(self):
        '''find the train_test_split line'''
        self.origin_code = self.origin_code.replace("\\\n", "")
        code_lines = self.origin_code.split("\n")
        
        not_found = True
        for index,line in enumerate(code_lines):
            if "train_test_split(" in line and '=' in line:
                self.indent = 0
                train_test_index = index
                not_found = False
                for char in line:
                    if ord(char) != 32:
                        break
                    self.indent += 1
        if not_found:
            return 0
        for index,line in enumerate(code_lines):
            if index == train_test_index:
                try:
                    line1 = line.strip()
                    r_node = ast.parse(line1)
                    func_node = r_node.body[0].value
                    target_node_tuple_node = r_node.body[0].targets[0]
                    target_node_list = [item for item in target_node_tuple_node.elts]
                except Exception as e:       
                    if str(e) == 'unexpected EOF while parsing (<unknown>, line 1)':
                        suc_run = False
                        add_num = 1
                        new_line = line.strip()
                        need_strip = False
                        while suc_run == False:
                            new_line = new_line + (code_lines[index+add_num].strip())
                            try:
                                r_node = ast.parse(new_line)
                                target_node_tuple_node = r_node.body[0].targets[0]
                                target_node_list = [item for item in target_node_tuple_node.elts]
                                func_node = r_node.body[0].value
                                suc_run=True
                            except:
                                add_num += 1
                    else:
                        continue
                if len(func_node.args) == 2:
                    self.X_varible = astunparse.unparse(func_node.args[0])[:-1]
                    self.y_varible = astunparse.unparse(func_node.args[1])[:-1]
                    if len(target_node_list) == 4:
                        self.x_train_varible = astunparse.unparse(target_node_list[0])[:-1]
                        self.x_test_varible = astunparse.unparse(target_node_list[1])[:-1]
                        self.y_train_varible = astunparse.unparse(target_node_list[2])[:-1]
                        self.y_test_varible = astunparse.unparse(target_node_list[3])[:-1]
                        self.train_test_split_index = index
                        return 1
                else:
                    self.train_test_split_no_label+=1
                    self.data_varible = astunparse.unparse(func_node.args[0])[:-1]
                    if len(target_node_list) == 2:
                        self.train_varible = astunparse.unparse(target_node_list[0])[:-1]
                        self.test_varible = astunparse.unparse(target_node_list[1])[:-1]
                        self.train_test_split_index = index
                        return 2
 
        return 0
    def change_train_test_split(self, result_id, ratio=0.8, split_random_state=0):
        """
        Uniformly change the train_test_split function to the specified ratio and random_state.
        Parameters
        ----------
        result_id: int
        ratio: float
        split_random_state: int
        """
        if self.train_test_split_index == None:
                return False
        indentline = ''
        for i in range(0,self.indent):
            indentline += ' '
        if result_id == 2:
            with open('../statsklearn/new_notebook.json', 'r') as f:
                dataset_label = json.load(f)
        
            if str(self.notebook_id) not in dataset_label:
                return False
            for name in dataset_label[str(self.notebook_id)]['column_index']:
                if dataset_label[str(self.notebook_id)]['index'][0] == dataset_label[str(self.notebook_id)]['column_index'][name]:
                    label_name = name

            self.x_train_varible = 'x_train_varible'
            self.y_train_varible = 'y_train_varible'
            self.x_test_varible = 'x_test_varible'
            self.y_test_varible = 'y_test_varible'

            self.X_varible = 'x_varible'
            self.y_varible = 'y_varible'
            
            split_x_y_code = self.y_varible + ' = ' + self.data_varible + '["' + label_name + '"]\n'
            split_x_y_code += self.X_varible + ' = ' + self.data_varible + '.drop(["' + label_name + '"], 1)\n'
            train_test_split_line = split_x_y_code + 'from sklearn.model_selection import train_test_split\n'

        elif result_id == 1:
            train_test_split_line = indentline + 'from sklearn.model_selection import train_test_split\n'
        train_test_split_line += indentline
        train_test_split_line += self.x_train_varible + ', ' + self.x_test_varible + ', ' +  self.y_train_varible + ', ' +  self.y_test_varible
        train_test_split_line = train_test_split_line + " = train_test_split(" + self.X_varible + ', ' + self.y_varible + ', train_size='+str(ratio)+', test_size=1-'+str(ratio)+', random_state='+str(split_random_state)+')' + '\n'
        code_lines = self.origin_code.split("\n")
        self.code = ''
        for index,line in enumerate(code_lines):
            if index == self.train_test_split_index and result_id==1:
                self.code += train_test_split_line
                continue
            elif index == self.train_test_split_index and result_id==2:
                self.code += train_test_split_line
                self.code += line
                self.code += '\n'
                continue
            if line == '#print(os.listdir("../input"))':
                self.code = self.code + '#' + line
            else: 
                self.code += line
            self.code += '\n'
        code_list = self.code.split('\n')
        self.change_suc += 1
        if result_id == 2:
            self.end_index = self.train_test_split_index +2
        else:
            self.end_index = self.train_test_split_index+1
        
        with open("haipipe/core/tmpdata/prenotebook_varibles_index/"+str(self.notebook_id)+".json", 'w') as f:
            json.dump( {'x_varible':self.X_varible, 'end_idnex': self.end_index}, f)

    def save_code(self, root_path):
        with open(root_path+ str(self.notebook_id) + '.py', 'w') as f:
            f.write(self.code)
    
         
    def add_model_code(self, metric_type='accuracy_score'):
        """
        add the specific model code to the HI-program if need.
        Parameters
        ----------
        metric_type : str
        """
        save_file_path = 'prenotebook_res/'+ str(self.notebook_id) + '.npy'
        model_type = self.info_triple[self.notebook_id]['model_type']
        if model_type not in model_import_code:
            return False
        if self.code == '':
            return False
        

        self.code = self.code+'import pandas as pd\n'+ accuracy_import_code[metric_type] + '\n'
        self.code = self.code + model_import_code[model_type] + '\n'
        self.code += '#print("start running model training........")\n'
        if model_type in ['LinearRegression','KNeighborsClassifier']:
            self.code += 'model = ' + model_type + '()\n'
        elif model_type == "LogisticRegression":
            self.code += "model = LogisticRegression(solver='liblinear', random_state=0)\n"
        else:
            self.code += 'model = ' + model_type + '(random_state=0)\n'
        self.code += 'model.fit(' + self.x_train_varible + ', ' + self.y_train_varible + ')\n'
        self.code += 'y_pred = model.predict(' + self.x_test_varible +')\n'
        self.code += 'score = ' +  metric_type +'(' + self.y_test_varible +', y_pred)\n'
        self.code += 'import numpy as np\n'
        self.code += 'np.save("haipipe/core/tmpdata/' + save_file_path +'", { "' + metric_type+'": score })\n'
        self.suc += 1
        return True
        

    def found_dataset(self, old_path, notebook_id, root_path, origin_code):
        """
        :param old_path:
        :param notebook_id:
        :param root_path:
        :param origin_code:
        :return:
        if found the error of missing dataset path, check the error path.
        """
        old_root_path = ''
        if '/' not in old_path:
            result = root_path + '/' + old_path
            old_root_path = old_path
        else:
            for index, i in enumerate(old_path.split('/')):
                if index != len(old_path.split('/')) - 1:
                    old_root_path = old_root_path + i + '/'
                else:
                    if '.' not in i:
                        old_root_path = old_root_path + i
                    if '/' == old_root_path[-1]:
                        old_root_path = old_root_path[0:-1]

            result = root_path
        return origin_code.replace(old_root_path, result)

    def run_one_code(self, notebook_id, origin_code, new_path, try_time, found=False):
        """
        :param origin_code: code
        :param new_path: replaced dataset path
        :param try_time: try executing time
        :return: the seccessfully executed code
        Execute the code in the notebook
        """
        try:
            if '/kaggle/input' in origin_code:
                origin_code = origin_code.replace('/kaggle/input', new_path)
            cm = compile(origin_code, '<string>', 'exec')
        except Exception as e:
            return "compile fail"
    
        can_run = False
        try:
            ns = {}
            exec(cm,ns)
            can_run = True
        except Exception as e:
            error_str = str(e)

            new_code = origin_code
            foun = 0
            if "[Errno 2] No such file or directory: " in error_str:
                error_path = error_str.replace("[Errno 2] No such file or directory: " , "")
                error_path = error_path[1:-1]
                new_code = self.found_dataset(error_path, 1, new_path, origin_code)
                foun=1

            elif "\"['Unnamed: 0'] not found in axis\"" in error_str:
                new_code = origin_code.replace("'Unnamed: 0'", "'index'")
            elif "does not exist:" in error_str and '[Errno 2] File ' in error_str:
                error_path = error_str.split(':')[-1].strip()
                error_path = error_path[1:-1]
                new_code = self.found_dataset(error_path, 1, new_path, origin_code)
                foun=1
            elif "No module named " in error_str and '_tkinter' not in error_str:
                package = error_str.replace("No module named ", "")
                package = package[1:-1]
                command = ' pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ' + package.split('.')[0]
                if 'sklearn' in command or 'scikit_learn' in command:
                    command = 'pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit_learn==0.23.2'
                os.system(command)
            elif  ": No such file or directory" in error_str:
                index1 = error_str.find("'")
                index2 = error_str.find("'", index1+1)
                error_path = error_str[index1+1:index2]
                new_code = self.found_dataset(error_path, 1, new_path, origin_code)
            elif "Command '['ls'," in error_str:
                index1 = error_str.find('ls')
                el_line = error_str[index1+6:]
                right_index  = el_line.find('\'')
                error_path = el_line[0:right_index]
                new_code = self.found_dataset(error_path, 1, new_path, origin_code)
                foun = 1
            elif "File b" in error_str:
                index1 = error_str.find("'")
                index2 = error_str.find("'", index1 + 1)
                error_path = error_str[index1 + 1:index2]
                new_code = self.found_dataset(error_path, 1, new_path, origin_code)
                foun = 1
            elif "'DataFrame' object has no attribute 'ix'" in error_str or "'Series' object has no attribute 'ix'" in error_str:
                new_code = origin_code.replace('.ix', '.iloc')
            elif "'DataFrame' object has no attribute 'sort'" in error_str:
                new_code = origin_code.replace('.sort(', '.sort_values(')
            else:
                return "False"
            if try_time < 7:
                if foun ==1:
                    found = True
                res = self.run_one_code(notebook_id, new_code, new_path, try_time + 1,found)
                if res == 'compile fail': 
                    return res
                elif  res== 'False':
                    return res
            else:
                return "False"
        return origin_code

    def deal_with_split_no_var(self):
        code_lines = self.code.split('\n')
        for index,line in enumerate(code_lines):
            if 'train_test_split(' in line:
                train_test_split_index = index
        try:
            r_node = ast.parse(code_lines[train_test_split_index])
            arg0_node = r_node.body[0].value.args[0]
            arg1_node = r_node.body[0].value.args[1]
            if type(arg0_node).__name__ != 'Name':
                add_line1 = 'X = '+ astunparse.unparse(arg0_node)[0:-1]
                code_lines[train_test_split_index] = code_lines[train_test_split_index].replace(astunparse.unparse(arg0_node)[0:-1], 'X')
                code_lines = code_lines[0:train_test_split_index] + [add_line1] + code_lines[train_test_split_index:]
                train_test_split_index += 1
            if type(arg1_node).__name__ != 'Name':
                add_line2 = 'y = '+ astunparse.unparse(arg1_node)
                code_lines[train_test_split_index] = code_lines[train_test_split_index].replace(astunparse.unparse(arg1_node)[0:-1], 'y')
                code_lines = code_lines[0:train_test_split_index] + [add_line2] + code_lines[train_test_split_index:]
                train_test_split_index += 1
            str_ = ''
            for line in code_lines:
                str_ += line
                str_ += '\n'
            self.origin_code = str_
            self.code = str_
        except Exception as e:
            return 

    def profiling_code(self, notebook_id, need_remove_model):
        """
        Index the start and end of data preparation of HI-program.
        Parameters
        ----------
        notebook_id: str
        need_remove_model: bool
        """
        self.load_origin_code(notebook_id, need_remove_model)
        self.deal_with_split_no_var()
        result_id = self.find_train_test_split()
        if result_id == 0:
            self.cant_find_train_test += 1
            if need_remove_model == 1:
                self.load_origin_code(notebook_id, need_remove_model=2)
                result_id = self.find_train_test_split()
                if result_id == 0:
                    return -1
            else:
                return -1
        self.change_train_test_split(result_id)
        res = self.add_model_code()
        self.save_code('haipipe/core/tmpdata/prenotebook_code/')
        return res

    def run_origin_test(self, notebook_id, need_try_again):
        """
        Execute HI-program.
        Parameters
        ----------
        notebook_id: str
        need_try_again: int
        """
        start = time.time()
        self.code = exchange_code(self.code)
        self.code = self.run_one_code(notebook_id, self.code, dataset_root_path + self.info_triple[notebook_id]['dataset_name'], 0)
        if self.code != '' and self.code != "False" and self.code != "compile fail":
            self.save_code('haipipe/core/tmpdata/runned_notebook/')
        else:
            if need_try_again == 2:
                res = self.profiling_code(notebook_id, need_remove_model=2)
                self.run_origin_test(notebook_id, need_try_again-1)
            elif need_try_again == 1:
                res = self.profiling_code(notebook_id, need_remove_model=0)
                self.run_origin_test(notebook_id, need_try_again-1)
            else:
                self.run_faile += 1
