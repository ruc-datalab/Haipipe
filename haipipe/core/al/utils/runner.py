from haipipe.core.al.utils.load_code import py2str
import time
import os
import json
import numpy as np
from func_timeout import func_set_timeout
import traceback

class Runner:
    def found_dataset(self, old_path, notebook_id, root_path, origin_code):
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
        
    def run_one_code(self, origin_code, new_path, try_time=0, found=False):
        if '/kaggle/input' in origin_code:
            origin_code = origin_code.replace('/kaggle/input', new_path)

        try:
            cm = compile(origin_code, '<string>', 'exec')
        except Exception as e:
            print("compile fail", e)
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
                res = self.run_one_code(new_code, new_path, try_time + 1,found)
                if res == 'compile fail': 
                    return res
                elif  res== 'False':
                    return res
            else:
                return "False"
        return origin_code

    def exchange_code(self, code):
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
    def run_one_case(self, code_path, dataset_path):
        start = time.time()
        self.code = py2str(code_path)
        self.code = self.exchange_code(self.code)
        self.code = self.run_one_code(self.code, dataset_path)
        if self.code != '' and self.code != "False" and self.code != "compile fail":
            end = time.time()
            return end-start
        else:
            return 0