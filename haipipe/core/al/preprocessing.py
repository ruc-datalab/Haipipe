from haipipe.core.al.config import Config
import json
import os
import copy
import numpy as np
import pandas as pd

config = Config()

def generate_hi_seq(notebook_id, traintest):
    if traintest == 'test':
        json_path = config.param_candidate_hai_val_json_save_root_path
    else:
        json_path = config.param_candidate_offline_hai_val_json_save_root_path
    edge_ids = []
    for seq_id in os.listdir(json_path + notebook_id):
        seq_id = seq_id.replace('.json', '')
        if seq_id == 'origin':
            continue
        with open(json_path + notebook_id + '/' + seq_id + '.json', 'r') as f:
            pipeline_json = json.load(f)
            pipeline_json = pipeline_json['seq']
        
        for ope_dic in pipeline_json:
            ope = ope_dic['operator']
            if ope_dic['edge_id'] == 'end':
                edge_id = 'end'
                pos = 'end'
            else:
                edge_id, pos = ope_dic['edge_id'].split("---")
                if int(edge_id) not in edge_ids:
                    edge_ids.append(int(edge_id))
    
    edge_ids = sorted(edge_ids)
    edge_ids.append('end')

    return edge_ids

def get_data_feature(train_x, train_y):
    inp_data = pd.DataFrame(train_x)
    if len(train_y.shape) > 1:
        train_y = train_y[0]
    else:
        train_y = train_y
    categorical = list(train_x.dtypes == object)
    column_info = {}

    num_cols = [col for col in train_x.columns if str(train_x[col].dtypes) != 'object']

    for i in range(min(len(inp_data.columns), config.dataset_feature_colnum)):
        col = inp_data.iloc[:,i]
        if i >= config.dataset_feature_colnum:
            break
        s_s = col
        column_info[i] = {}
        column_info[i]['col_name'] = 'unknown_' + str(i)
        column_info[i]['dtype'] = str(s_s.dtypes) # 1
        column_info[i]['length'] = len(s_s.values) # 2
        column_info[i]['null_ratio'] = s_s.isnull().sum() / len(s_s.values) # 3
        column_info[i]['ctype'] = 1 if inp_data.columns[i] in num_cols else 2 # 4
        column_info[i]['nunique'] = s_s.nunique() # 5
        column_info[i]['nunique_ratio'] = s_s.nunique() / len(s_s.values) # 6

        if 'mean' not in s_s.describe():
            column_info[i]['ctype'] = 2
        if column_info[i]['ctype'] == 1:  # 如果是数字列
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

        elif column_info[i]['ctype'] == 2:  # category列
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
                content = config.dtype_dic[column_dic[kw]]
            else:
                content = column_dic[kw]
            one_column_feature.append(content)
        data_feature.append(one_column_feature)
    if len(column_info) < config.dataset_feature_colnum:
        for index in range(len(column_info), config.dataset_feature_colnum):
            one_column_feature = np.zeros(config.column_feature_dim)
            data_feature.append(one_column_feature)
    data_feature = np.ravel(np.array(data_feature))
    
    del inp_data
    del column_info
    return data_feature

def generate_pipeline_feature(notebook_id, seq_id, hipipe, traintest):
    if traintest == 'test':
        json_path = config.param_candidate_hai_val_json_save_root_path
    else:
        json_path = config.param_candidate_offline_hai_val_json_save_root_path
    if seq_id == 'origin':
        hipipe_copy = copy.deepcopy(hipipe)
    else:
        with open(json_path + notebook_id + '/' + seq_id + '.json', 'r') as f:
            pipeline_json = json.load(f)
            pipeline_json = pipeline_json['seq']
        hipipe_copy = copy.deepcopy(hipipe)
        after_num = {}
        for ope_dic in pipeline_json:
            operator = ope_dic['operator']
            if ope_dic['edge_id'] == 'end':
                edge_id = 'end'
                if edge_id not in after_num:
                    after_num[edge_id] = 0
                hipipe_copy.insert(hipipe_copy.index(edge_id)+after_num[edge_id]+1, operator)
                after_num[edge_id] += 1
            else:
                edge_id, pos = ope_dic['edge_id'].split('---')
                edge_id = int(edge_id)
                if pos == 'before':
                    hipipe_copy.insert(hipipe_copy.index(edge_id), operator)
                elif pos == 'after':
                    if edge_id not in after_num:
                        after_num[edge_id] = 0
                    
                    hipipe_copy.insert(hipipe_copy.index(edge_id)+after_num[edge_id] + 1, operator)
                    after_num[edge_id] += 1
        
    res = []
    for ope in hipipe_copy:
        if ope not in config.ope2id:
            res.append(len(config.ope2id)+1)
        else:
            res.append(config.ope2id[ope])

    while len(res) < config.seq_len:
        res.append(0)

    return res

def generate_dataset_feature(notebook_id):
    info_triple = np.load(config.info_triple_path, allow_pickle = True).item()
    
    with open(config.dataset_label_path, 'r') as f:
        dataset_label = json.load(f)
    dataset = dataset_label[notebook_id]['dataset']
    
    for col in dataset_label[notebook_id]['column_index']:
        if dataset_label[notebook_id]['column_index'][col] == dataset_label[notebook_id]['index'][0]:
            label_column = col

    dataset_path = config.origin_dataset + dataset + '/'
    csv_name = [item for item in list(os.listdir(dataset_path)) if '.csv' in item][0]


    data = pd.read_csv(dataset_path + '/' + csv_name)
    y = data[label_column]
    X = data.drop([label_column], axis = 1)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=1-0.8, random_state=0)
    return get_data_feature(X_train, y_train)

def generate_dataset_pipeline_features(notebook_id, traintest):
    dataset_features = []
    pipeline_features = []

    dataset_feature = generate_dataset_feature(notebook_id)
    

    edge_ids = generate_hi_seq(notebook_id, traintest)

    seq_ids = []
    if traintest == 'test':
        json_path = config.param_candidate_hai_val_json_save_root_path
    else:
        json_path = config.param_candidate_offline_hai_val_json_save_root_path

    for seq_id in os.listdir(json_path + notebook_id):
        seq_id = seq_id.replace('.json', '')
        seq_ids.append(seq_id)
        hipipe_copy = generate_pipeline_feature(notebook_id, seq_id, edge_ids, traintest)
        pipeline_features.append(hipipe_copy)
        dataset_features.append(dataset_feature)

    return np.array(dataset_features), np.array(pipeline_features), np.array(seq_ids)
    
def generate_one_test_features(notebook_id):
    save_path = config.test_features

    dataset_features, pipeline_features, seq_ids = generate_dataset_pipeline_features(notebook_id, 'test')

    res = {
        'dataset_features': dataset_features,
        'pipeline_features': pipeline_features,
        'seq_ids': seq_ids
    } 

    np.save(save_path + notebook_id + '.npy', res)

def batch_generate_features(traintest, thread_id, thread_num):
    if traintest == 'test':
        notebook_path = config.param_candidate_hai_val_json_save_root_path
        save_path = config.test_features
    else:
        notebook_path = config.param_candidate_offline_hai_val_json_save_root_path
        save_path = config.train_features
    notebooks = os.listdir(notebook_path)

    batch_len = len(notebooks)/thread_num
    for index,notebook_id in enumerate(notebooks):
        if notebook_id != 'datascientist25_gender-recognition-by-voice-using-machine-learning':
            continue
        print(index, len(notebooks))
        if os.path.exists(save_path + notebook_id + '.npy'):
            print('exists')
            continue
        try:
            dataset_features, pipeline_features, seq_ids = generate_dataset_pipeline_features(notebook_id, traintest)
        except:
            continue
        res = {
            'dataset_features': dataset_features,
            'pipeline_features': pipeline_features,
            'seq_ids': seq_ids
        } 

        np.save(save_path + notebook_id + '.npy', res)

def generate_training_data():
    notebook_path = param_candidate_offline_hai_val_json_save_root_path
    save_path = config.train_features

    notebooks = os.listdir(notebook_path)

    dataset_features = []
    pipeline_features = []
    
    y = []


    notebook_score = {}
    seq_num = 0
    for index,notebook_id in enumerate(notebooks):
        print(index, len(notebooks))
        if not os.path.exists(config.train_features + notebook_id + '.npy'):
            continue
        temp = []
        features = np.load(config.train_features + notebook_id + '.npy', allow_pickle=True).item()
        for index, seq_id in enumerate(features['seq_ids']):
            sv_path = config.param_offline_hai_val_result_save_root_path + notebook_id + '/' + seq_id + '.npy'
            if not os.path.exists(sv_path):
                continue
            accuracy = np.load(sv_path, allow_pickle=True).item()['accuracy'].mean()
            if accuracy != accuracy:
                continue
            temp.append(accuracy)
        if len(temp) > 1:
            notebook_score[notebook_id] = np.array(temp).mean()

        else:
            continue
        for index, seq_id in enumerate(features['seq_ids']):
            sv_path = config.param_offline_hai_val_result_save_root_path + notebook_id + '/' + seq_id + '.npy'
            if not os.path.exists(sv_path):
                continue
            origin_score = np.load(sv_path, allow_pickle=True).item()['accuracy'].mean()
            if origin_score != origin_score:
                continue
            dataset_feature = features['dataset_features'][index]
            pipeline_feature = features['pipeline_features'][index]

            dataset_features.append(dataset_feature)
            pipeline_features.append(pipeline_feature)
            y.append(origin_score - notebook_score[notebook_id])

    print(len(dataset_features), len(pipeline_features), len(y))
    dataset_features = np.array(dataset_features)
    pipeline_features = np.array(pipeline_features)
    y = np.array(y)
    np.save('train_dataset_features.npy', dataset_features)
    np.save('train_pipeline_features.npy', pipeline_features)
    np.save('train_y.npy', y)

def para_gen_features(traintest):
    import multiprocessing
    p1 = multiprocessing.Process(target=batch_generate_features, args=(traintest, 1, 6)) 
    p2 = multiprocessing.Process(target=batch_generate_features, args=(traintest, 2, 6))
    p3 = multiprocessing.Process(target=batch_generate_features, args=(traintest, 3, 6))
    p4 = multiprocessing.Process(target=batch_generate_features, args=(traintest, 4, 6))
    p5 = multiprocessing.Process(target=batch_generate_features, args=(traintest ,5, 6))
    p6 = multiprocessing.Process(target=batch_generate_features, args=(traintest ,6, 6))

    p1.start() 
    p2.start()
    p3.start() 
    p4.start()
    p5.start()
    p6.start()

    p1.join() 
    p2.join()
    p3.join() 
    p4.join()
    p5.join()
    p6.join()

if __name__ == '__main__':
    para_gen_features('test')
    