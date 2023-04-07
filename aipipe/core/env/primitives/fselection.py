from .primitive import Primitive
from sklearn.feature_selection import VarianceThreshold, GenericUnivariateSelect, chi2, SelectKBest, f_classif,\
    mutual_info_classif, f_regression, mutual_info_regression, SelectPercentile, SelectFpr, SelectFdr, SelectFwe, RFE
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
import pandas as pd
from copy import deepcopy
from itertools import compress
import numpy as np
np.random.seed(1)

class VarianceThresholdPrim(Primitive):
    def __init__(self, random_state=0):
        super(VarianceThresholdPrim, self).__init__(name='VarianceThreshold')
        self.id = 1
        self.gid = 24
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Feature selector that removes all low-variance features."
        self.selector = VarianceThreshold()
        self.accept_type = 'c_t'
        self.need_y = True
    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        self.selector.fit(train_x)

        cols = list(train_x.columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        train_data_x = pd.DataFrame(self.selector.transform(train_x), columns=final_cols)

        cols = list(test_x.columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        test_data_x = pd.DataFrame(self.selector.transform(test_x), columns=final_cols)
        return train_data_x, test_data_x
