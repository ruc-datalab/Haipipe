import numpy as np
from sklearn.cluster import KMeans
import torch
import copy
import random
class Sampler:
    def __init__(self, k, dataset_features, pipeline_features, model):
        self.k = k
        self.samples = []
        self.tryed = []
        self.emcm = EMCM(model)
        self.dataset_features = dataset_features
        self.pipeline_features = pipeline_features
        self.model = model
        self.samples_label = {}


    def representative_sample(self, latent_features, k, random_state=0):
        other_latent_feataures = []
        id_list = []
        for id_, feature in enumerate(latent_features):
            if id_ in self.tryed or id_ in self.samples:
                continue
            id_list.append(id_)
            other_latent_feataures.append(feature)

        if len(other_latent_feataures) == 0:
            return [],[],[]
        if len(other_latent_feataures) == 1:
            candidate_dataset_features = []
            candidate_pipeline_features = []
            for i, feature in enumerate(other_latent_feataures):
                candidate_pipeline_features.append(torch.LongTensor(self.pipeline_features[id_list[i]]))
                candidate_dataset_features.append(torch.FloatTensor(self.dataset_features[id_list[i]]))
            candidate_dataset_features = torch.stack(candidate_dataset_features, 0)
            candidate_pipeline_features = torch.stack(candidate_pipeline_features, 0)
            return id_list, candidate_dataset_features, candidate_pipeline_features
        
        kms_model = KMeans(n_clusters=min(2*k, len(other_latent_feataures)), random_state=random_state)
        
        kms_model.fit(other_latent_feataures)

        center_points = []
        for cluster_index,cluster_center in enumerate(kms_model.cluster_centers_):
            min_dis = np.inf
            min_feature_index = -1
            for feature_index,feature in enumerate(other_latent_feataures):
                if kms_model.labels_[feature_index] != cluster_index:
                    continue
                distance = np.linalg.norm(feature-cluster_center)
                if distance < min_dis:
                    min_dis = distance
                    min_feature_index = feature_index
            center_points.append(min_feature_index)

        candidate_ids = []

        candidate_dataset_features = []
        candidate_pipeline_features = []
        for i, feature in enumerate(other_latent_feataures):
            if i in center_points and id_list[i] not in self.samples:
                candidate_ids.append(id_list[i])
                candidate_pipeline_features.append(torch.LongTensor(self.pipeline_features[id_list[i]]))
                candidate_dataset_features.append(torch.FloatTensor(self.dataset_features[id_list[i]]))
        candidate_dataset_features = torch.stack(candidate_dataset_features, 0)
        candidate_pipeline_features = torch.stack(candidate_pipeline_features, 0)
        return candidate_ids, candidate_dataset_features, candidate_pipeline_features
    
    def diversity_sort(self, candidate_ids, latent_features):
        distance_dic = {}
    
        for index,id_ in enumerate(candidate_ids):
            cand_feature = latent_features[index]
            min_dis = np.inf
            for sample_id in self.samples:
                feature = latent_features[sample_id]
                distance = np.linalg.norm(feature-cand_feature)
                if distance < min_dis:
                    min_dis = distance
            distance_dic[id_] = min_dis

        distance_dic = sorted(distance_dic.items(), key=lambda x: x[1])

        return distance_dic

    def informative_sort(self, candidate_ids, candidate_dataset_features, candidate_pipeline_features):
        if len(candidate_ids) == 0:
            return {}
        change_dic = self.emcm.update_labeled(self.samples, candidate_ids, candidate_dataset_features, candidate_pipeline_features, self.samples_label)
        return change_dic
            
    def select_topk(self, k, i, r, d):
        self.i, self.r, self.d = i, r, d
        all_pipeline_features = torch.LongTensor(self.pipeline_features)
        model_copy = copy.deepcopy(self.model)
        model_copy.eval()
        latent_features = model_copy.latent(all_pipeline_features)
        latent_features = latent_features.detach().numpy()
        
        if self.r == True:
            candidate_ids, candidate_dataset_features, candidate_pipeline_features = self.representative_sample(latent_features, k)
        else:
            id_list = []
            other_latent_feataures = []
            for id_, feature in enumerate(latent_features):
                if id_ in self.tryed or id_ in self.samples:
                    continue
                id_list.append(id_)
                other_latent_feataures.append(feature)
            if len(other_latent_feataures) == 0:
                return [],[],[]
            candidate_dataset_features = []
            candidate_pipeline_features = []
            for ind, feature in enumerate(other_latent_feataures):
                candidate_pipeline_features.append(torch.LongTensor(self.pipeline_features[id_list[ind]]))
                candidate_dataset_features.append(torch.FloatTensor(self.dataset_features[id_list[ind]]))
            candidate_dataset_features = torch.stack(candidate_dataset_features, 0)
            candidate_pipeline_features = torch.stack(candidate_pipeline_features, 0)
            candidate_ids, candidate_dataset_features, candidate_pipeline_features = id_list, candidate_dataset_features, candidate_pipeline_features
  
        if self.d == True:
            distance_dic = self.diversity_sort(candidate_ids, latent_features)


        if self.i == True:
            change_dic = self.informative_sort(candidate_ids, candidate_dataset_features, candidate_pipeline_features)

        next_sample_id = []
        i=0
        j = 0
   
        while len(next_sample_id) < min(len(candidate_ids), k):
            if self.d == True:
                while distance_dic[i][0] in next_sample_id and i < len(distance_dic):
                    i += 1
                if i < len(distance_dic):
                    next_sample_id.append(distance_dic[i][0])

            if len(next_sample_id) >= min(len(candidate_ids), k):
                break

            if self.i == True:
                while change_dic[j][0] in next_sample_id and j < len(change_dic):
                    j += 1
                if j < len(change_dic):
                    next_sample_id.append(change_dic[j][0])

            if self.d == False and self.i == False:
                next_sample_id.append(candidate_ids[i])
                i += 1
   
        self.samples += next_sample_id
        self.tryed += next_sample_id
        
        return next_sample_id

    
            


class EMCM:
    def __init__(self, model):
        self.qbc_models = []
        self.num_committee = 3
        self.model = model
        for i in range(self.num_committee):
            new_model = copy.deepcopy(self.model)
            self.qbc_models.append(new_model)


    def update_labeled(self, samples, candidate_ids, dataset_features, pipeline_features, samples_label):
        eq_24 = {}

        unlabeled_pos_list = candidate_ids
        res = {}

        x = torch.cat((dataset_features, pipeline_features), 1)
        y = {}

        model_copy = copy.deepcopy(self.model)
        if len(dataset_features) == 1:
            model_copy.eval()
        else:
            model_copy.train()
        model_copy.seq_lstm.dropout = 0
        fx = model_copy(dataset_features, pipeline_features)
        for j in range(len(self.qbc_models)):
            
            sub_model_copy = copy.deepcopy(self.qbc_models[j])
            sub_model_copy.seq_lstm.dropout = 0
            if len(dataset_features) == 1:
                sub_model_copy.eval()
            else:
                sub_model_copy.train()
            y[j] = sub_model_copy(dataset_features, pipeline_features)
            
        for index in range(len(candidate_ids)):
            for j in range(len(self.qbc_models)):
                eq_24[candidate_ids[index]] = 0
                eq_24[candidate_ids[index]] += np.linalg.norm((fx.detach()[index] - y[j].detach()[index]) * x[index])
                eq_24[candidate_ids[index]] /= (1.0 * len(self.qbc_models))

            res[candidate_ids[index]] = eq_24[candidate_ids[index]]
        
        res = sorted(res.items(), key=lambda x: x[1], reverse = True)
        return res