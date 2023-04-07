from .primitive import Primitive
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer, Normalizer, KBinsDiscretizer, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from copy import deepcopy
np.random.seed(1)

def catch_num(data):
    num_cols = [col for col in data.columns if str(data[col].dtypes) != 'object']
    cat_cols = [col for col in data.columns if col not in num_cols]
    cat_train_x = data[cat_cols]
    num_train_x = data[num_cols]
    return cat_train_x, num_train_x

class MinMaxScalerPrim(Primitive):
    def __init__(self, random_state=0):
        super(MinMaxScalerPrim, self).__init__(name='MinMaxScaler')
        self.id = 1
        self.gid = 9
        self.hyperparams = []
        self.type = 'FeaturePreprocessing'
        self.description = "Transforms features by scaling each feature to a given range. This estimator scales and translates each feature individually such that it is in the given range on the training set, e.g. between zero and one."
        self.scaler = MinMaxScaler()
        self.accept_type = 'c_t'
        self.need_y = False

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        cat_train_x, num_train_x = catch_num(train_x)
        cat_test_x, num_test_x = catch_num(test_x)
      
        self.scaler.fit(num_train_x)

        num_train_x = pd.DataFrame(self.scaler.transform(num_train_x), columns=list(num_train_x.columns)).reset_index(drop=True).infer_objects()
        train_data_x = pd.concat([cat_train_x.reset_index(drop=True), num_train_x.reset_index(drop=True)],axis=1)

        num_test_x = pd.DataFrame(self.scaler.transform(num_test_x), columns=list(num_test_x.columns)).reset_index(drop=True).infer_objects()
        test_data_x = pd.concat([cat_test_x.reset_index(drop=True), num_test_x.reset_index(drop=True)],axis=1)
        return train_data_x, test_data_x

class MaxAbsScalerPrim(Primitive):
    def __init__(self, random_state=0):
        super(MaxAbsScalerPrim, self).__init__(name='MaxAbsScaler')
        self.id = 2
        self.gid = 10
        self.hyperparams = []
        self.type = 'FeaturePreprocessing'
        self.description = "Scale each feature by its maximum absolute value. his estimator scales and translates each feature individually such that the maximal absolute value of each feature in the training set will be 1.0. It does not shift/center the data, and thus does not destroy any sparsity. This scaler can also be applied to sparse CSR or CSC matrices."
        self.scaler = MaxAbsScaler()
        self.accept_type = 'c_t'
        self.need_y = False

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        cat_train_x, num_train_x = catch_num(train_x)
        cat_test_x, num_test_x = catch_num(test_x)
      
        self.scaler.fit(num_train_x)

        num_train_x = pd.DataFrame(self.scaler.transform(num_train_x), columns=list(num_train_x.columns)).reset_index(drop=True).infer_objects()
        train_data_x = pd.concat([cat_train_x.reset_index(drop=True), num_train_x.reset_index(drop=True)],axis=1)

        num_test_x = pd.DataFrame(self.scaler.transform(num_test_x), columns=list(num_test_x.columns)).reset_index(drop=True).infer_objects()
        test_data_x = pd.concat([cat_test_x.reset_index(drop=True), num_test_x.reset_index(drop=True)],axis=1)
        return train_data_x, test_data_x


class RobustScalerPrim(Primitive):
    def __init__(self, random_state=0):
        super(RobustScalerPrim, self).__init__(name='RobustScaler')
        self.id = 3
        self.gid = 11
        self.hyperparams = []
        self.type = 'FeaturePreprocessing'
        self.description = "Scale features using statistics that are robust to outliers. This Scaler removes the median and scales the data according to the quantile range (defaults to IQR: Interquartile Range). The IQR is the range between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile). Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set. Median and interquartile range are then stored to be used on later data using the transform method. Standardization of a dataset is a common requirement for many machine learning estimators. Typically this is done by removing the mean and scaling to unit variance. However, outliers can often influence the sample mean / variance in a negative way. In such cases, the median and the interquartile range often give better results."
        self.scaler = RobustScaler()
        self.accept_type = 'c_t'
        self.need_y = False

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        cat_train_x, num_train_x = catch_num(train_x)
        cat_test_x, num_test_x = catch_num(test_x)
      
        self.scaler.fit(num_train_x)

        num_train_x = pd.DataFrame(self.scaler.transform(num_train_x), columns=list(num_train_x.columns)).reset_index(drop=True).infer_objects()
        train_data_x = pd.concat([cat_train_x.reset_index(drop=True), num_train_x.reset_index(drop=True)],axis=1)

        num_test_x = pd.DataFrame(self.scaler.transform(num_test_x), columns=list(num_test_x.columns)).reset_index(drop=True).infer_objects()
        test_data_x = pd.concat([cat_test_x.reset_index(drop=True), num_test_x.reset_index(drop=True)],axis=1)
        return train_data_x, test_data_x



class StandardScalerPrim(Primitive):
    def __init__(self, random_state=0):
        super(StandardScalerPrim, self).__init__(name='StandardScaler')
        self.id = 4
        self.gid = 12
        self.hyperparams = []
        self.type = 'FeaturePreprocessing'
        self.description = "Standardize features by removing the mean and scaling to unit variance"
        self.scaler = StandardScaler()
        self.accept_type = 'c_t'
        self.need_y = False

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        cat_train_x, num_train_x = catch_num(train_x)
        cat_test_x, num_test_x = catch_num(test_x)
      
        self.scaler.fit(num_train_x)

        num_train_x = pd.DataFrame(self.scaler.transform(num_train_x), columns=list(num_train_x.columns)).reset_index(drop=True).infer_objects()
        train_data_x = pd.concat([cat_train_x.reset_index(drop=True), num_train_x.reset_index(drop=True)],axis=1)

        num_test_x = pd.DataFrame(self.scaler.transform(num_test_x), columns=list(num_test_x.columns)).reset_index(drop=True).infer_objects()
        test_data_x = pd.concat([cat_test_x.reset_index(drop=True), num_test_x.reset_index(drop=True)],axis=1)
        return train_data_x, test_data_x



class QuantileTransformerPrim(Primitive):
    def __init__(self, random_state=0):
        super(QuantileTransformerPrim, self).__init__(name='QuantileTransformer')
        self.id = 5
        self.gid = 13
        self.hyperparams = []
        self.type = 'FeaturePreprocessing'
        self.description = "Transform features using quantiles information. This method transforms the features to follow a uniform or a normal distribution. Therefore, for a given feature, this transformation tends to spread out the most frequent values. It also reduces the impact of (marginal) outliers: this is therefore a robust preprocessing scheme. The transformation is applied on each feature independently. The cumulative distribution function of a feature is used to project the original values. Features values of new/unseen data that fall below or above the fitted range will be mapped to the bounds of the output distribution. Note that this transform is non-linear. It may distort linear correlations between variables measured at the same scale but renders variables measured at different scales more directly comparable."
        self.scaler = QuantileTransformer()
        self.accept_type = 'c_t'
        self.need_y = False

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        cat_train_x, num_train_x = catch_num(train_x)
        cat_test_x, num_test_x = catch_num(test_x)
      
        self.scaler.fit(num_train_x)

        num_train_x = pd.DataFrame(self.scaler.transform(num_train_x), columns=list(num_train_x.columns)).reset_index(drop=True).infer_objects()
        train_data_x = pd.concat([cat_train_x.reset_index(drop=True), num_train_x.reset_index(drop=True)],axis=1)

        num_test_x = pd.DataFrame(self.scaler.transform(num_test_x), columns=list(num_test_x.columns)).reset_index(drop=True).infer_objects()
        test_data_x = pd.concat([cat_test_x.reset_index(drop=True), num_test_x.reset_index(drop=True)],axis=1)
        return train_data_x, test_data_x



class PowerTransformerPrim(Primitive):
    def __init__(self, random_state=0):
        super(PowerTransformerPrim, self).__init__(name='PowerTransformer')
        self.id = 6
        self.gid = 14
        self.hyperparams = []
        self.type = 'FeaturePreprocessing'
        self.description = "Apply a power transform featurewise to make data more Gaussian-like. Power transforms are a family of parametric, monotonic transformations that are applied to make data more Gaussian-like. This is useful for modeling issues related to heteroscedasticity (non-constant variance), or other situations where normality is desired. Currently, PowerTransformer supports the Box-Cox transform and the Yeo-Johnson transform. The optimal parameter for stabilizing variance and minimizing skewness is estimated through maximum likelihood. Box-Cox requires input data to be strictly positive, while Yeo-Johnson supports both positive or negative data. By default, zero-mean, unit-variance normalization is applied to the transformed data."
        self.scaler = PowerTransformer()
        self.accept_type = 'c_t'
        self.need_y = False

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        cat_train_x, num_train_x = catch_num(train_x)
        cat_test_x, num_test_x = catch_num(test_x)
      
        self.scaler.fit(num_train_x)

        num_train_x = pd.DataFrame(self.scaler.transform(num_train_x), columns=list(num_train_x.columns)).reset_index(drop=True).infer_objects()
        train_data_x = pd.concat([cat_train_x.reset_index(drop=True), num_train_x.reset_index(drop=True)],axis=1)

        num_test_x = pd.DataFrame(self.scaler.transform(num_test_x), columns=list(num_test_x.columns)).reset_index(drop=True).infer_objects()
        test_data_x = pd.concat([cat_test_x.reset_index(drop=True), num_test_x.reset_index(drop=True)],axis=1)
        return train_data_x, test_data_x


class NormalizerPrim(Primitive):
    def __init__(self, random_state=0):
        super(NormalizerPrim, self).__init__(name='Normalizer')
        self.id = 7
        self.gid = 15
        self.hyperparams = []
        self.type = 'FeaturePreprocessing'
        self.description = "Normalize samples individually to unit norm. Each sample (i.e. each row of the data matrix) with at least one non zero component is rescaled independently of other samples so that its norm (l1 or l2) equals one. This transformer is able to work both with dense numpy arrays and scipy.sparse matrix (use CSR format if you want to avoid the burden of a copy / conversion). Scaling inputs to unit norms is a common operation for text classification or clustering for instance. For instance the dot product of two l2-normalized TF-IDF vectors is the cosine similarity of the vectors and is the base similarity metric for the Vector Space Model commonly used by the Information Retrieval community."
        self.scaler = Normalizer()
        self.accept_type = 'c_t'
        self.need_y = False

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        cat_train_x, num_train_x = catch_num(train_x)
        cat_test_x, num_test_x = catch_num(test_x)
      
        self.scaler.fit(num_train_x)

        num_train_x = pd.DataFrame(self.scaler.transform(num_train_x), columns=list(num_train_x.columns)).reset_index(drop=True).infer_objects()
        train_data_x = pd.concat([cat_train_x.reset_index(drop=True), num_train_x.reset_index(drop=True)],axis=1)

        num_test_x = pd.DataFrame(self.scaler.transform(num_test_x), columns=list(num_test_x.columns)).reset_index(drop=True).infer_objects()
        test_data_x = pd.concat([cat_test_x.reset_index(drop=True), num_test_x.reset_index(drop=True)],axis=1)
        return train_data_x, test_data_x



class KBinsDiscretizerOrdinalPrim(Primitive):
    def __init__(self, random_state=0):
        super(KBinsDiscretizerOrdinalPrim, self).__init__(name='KBinsDiscretizerOrdinal')
        self.id = 8
        self.gid = 16
        self.hyperparams = []
        self.type = 'FeaturePreprocessing'
        self.description = "Bin continuous data into intervals. Ordinal."
        self.hyperparams_run = {'default': True}
        self.preprocess = None
        self.accept_type = 'c_t_kbins'
        self.need_y = False

    def can_accept(self, data):
        if not self.can_accept_c(data):
            return False
        return True

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        cat_train_x, num_train_x = catch_num(train_x)
        cat_test_x, num_test_x = catch_num(test_x)
        self.scaler = ColumnTransformer([("discrit", KBinsDiscretizer(encode='ordinal'), list(num_train_x.columns))])
        self.scaler.fit(num_train_x)

        num_train_x = pd.DataFrame(self.scaler.transform(num_train_x), columns=list(num_train_x.columns)).reset_index(drop=True).infer_objects()
        train_data_x = pd.concat([cat_train_x.reset_index(drop=True), num_train_x.reset_index(drop=True)],axis=1)

        num_test_x = pd.DataFrame(self.scaler.transform(num_test_x), columns=list(num_test_x.columns)).reset_index(drop=True).infer_objects()
        test_data_x = pd.concat([cat_test_x.reset_index(drop=True), num_test_x.reset_index(drop=True)],axis=1)
        return train_data_x, test_data_x