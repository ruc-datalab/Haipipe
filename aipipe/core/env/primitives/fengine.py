from .primitive import Primitive
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA, SparsePCA, TruncatedSVD, FastICA, NMF, LatentDirichletAllocation
from sklearn.ensemble import RandomTreesEmbedding
import pandas as pd
import numpy as np
from copy import deepcopy
import warnings
np.random.seed(1)

class PolynomialFeaturesPrim(Primitive):
    def __init__(self, random_state=0):
        super(PolynomialFeaturesPrim, self).__init__(name='PolynomialFeatures')
        self.id = 1
        self.gid = 17
        self.hyperparams = []
        self.type = 'FeatureEngine'
        self.description = "Generate polynomial and interaction features. Generate a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. For example, if an input sample is two dimensional and of the form [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2]."
        self.scaler = PolynomialFeatures(include_bias=False)
        self.accept_type = 'c_t'
        self.need_y = False

    def can_accept(self, data):
        if data.shape[1] > 100:
            return False
        else:
            return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        self.scaler.fit(train_x)

        train_data_x = self.scaler.transform(train_x)
        train_data_x = pd.DataFrame(train_data_x)
        train_data_x = train_data_x.loc[:, ~train_data_x.columns.duplicated()]

        test_data_x = self.scaler.transform(test_x)
        test_data_x = pd.DataFrame(test_data_x)
        test_data_x = test_data_x.loc[:, ~test_data_x.columns.duplicated()]
        return train_data_x, test_data_x


class InteractionFeaturesPrim(Primitive):
    def __init__(self, random_state=0):
        super(InteractionFeaturesPrim, self).__init__(name='InteractionFeatures')
        self.id = 2
        self.gid = 18
        self.hyperparams = []
        self.type = 'FeatureEngine'
        self.description = "Generate interaction features."
        self.scaler = PolynomialFeatures(interaction_only=True, include_bias=False)
        self.accept_type = 'c_t'
        self.need_y = False

    def can_accept(self, data):
        if data.shape[1] > 100:
            return False
        else:
            return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        self.scaler.fit(train_x)

        train_data_x = self.scaler.transform(train_x)
        train_data_x = pd.DataFrame(train_data_x)
        train_data_x = train_data_x.loc[:, ~train_data_x.columns.duplicated()]


        test_data_x = self.scaler.transform(test_x)
        test_data_x = pd.DataFrame(test_data_x)
        test_data_x = test_data_x.loc[:, ~test_data_x.columns.duplicated()]
        return train_data_x, test_data_x

class PCA_AUTO_Prim(Primitive):
    def __init__(self, random_state=0):
        super(PCA_AUTO_Prim, self).__init__(name='PCA_AUTO')
        self.id = 3
        self.gid = 19
        self.PCA_AUTO_Prim = []
        self.type = 'FeatureEngine'
        self.description = "LAPACK principal component analysis (PCA). Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space. It uses the LAPACK implementation of the full SVD or a randomized truncated SVD by the method of Halko et al. 2009, depending on the shape of the input data and the number of components to extract."
        self.pca = PCA(svd_solver='auto')  # n_components=0.9
        self.accept_type = 'c_t'
        self.need_y = False

    def can_accept(self, data):
        can_num = len(data.columns) > 4
        return self.can_accept_c(data) and can_num

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        
        cols = list(train_x.columns)
        self.pca.fit(train_x)

        train_data_x = self.pca.transform(train_x)
        train_data_x = pd.DataFrame(train_data_x, columns=cols[:train_data_x.shape[1]])

        test_data_x = self.pca.transform(test_x)
        test_data_x = pd.DataFrame(test_data_x, columns=cols[:test_data_x.shape[1]])
        return train_data_x ,test_data_x





class IncrementalPCA_Prim(Primitive):
    def __init__(self, random_state=0):
        super(IncrementalPCA_Prim, self).__init__(name='IncrementalPCA')
        self.id = 5
        self.gid = 20
        self.PCA_LAPACK_Prim = []
        self.type = 'FeatureEngine'
        self.description = "Incremental principal components analysis (IPCA). Linear dimensionality reduction using Singular Value Decomposition of centered data, keeping only the most significant singular vectors to project the data to a lower dimensional space. Depending on the size of the input data, this algorithm can be much more memory efficient than a PCA. This algorithm has constant memory complexity."
        self.hyperparams_run = {'default': True}
        self.pca = IncrementalPCA()
        self.accept_type = 'c_t'
        self.need_y = False

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        cols = list(train_x.columns)
        self.pca.fit(train_x)

        train_data_x = self.pca.transform(train_x)
        train_data_x = pd.DataFrame(train_data_x, columns=cols[:train_data_x.shape[1]])

        test_data_x = self.pca.transform(test_x)
        test_data_x = pd.DataFrame(test_data_x, columns=cols[:test_data_x.shape[1]])
        return train_data_x, test_data_x


class KernelPCA_Prim(Primitive):
    def __init__(self, random_state=0):
        super(KernelPCA_Prim, self).__init__(name='KernelPCA')
        self.id = 6
        self.gid = 21
        self.PCA_LAPACK_Prim = []
        self.type = 'FeatureEngine'
        self.description = "Kernel Principal component analysis (KPCA). Non-linear dimensionality reduction through the use of kernels"
        self.pca = KernelPCA(n_components=2)  # n_components=5
        self.accept_type = 'c_t_krnl'
        self.random_state = random_state
        self.need_y = False

    def can_accept(self, data):
        if data.shape[1] <= 2:
            return False
        else:
            return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        cols = list(train_x.columns)
        self.pca.fit(train_x)

        train_data_x = self.pca.transform(train_x)
        train_data_x = pd.DataFrame(train_data_x, columns=cols[:train_data_x.shape[1]])

        test_data_x = self.pca.transform(test_x)
        test_data_x = pd.DataFrame(test_data_x, columns=cols[:test_data_x.shape[1]])
        return train_data_x, test_data_x

class TruncatedSVD_Prim(Primitive):
    def __init__(self, random_state=0):
        super(TruncatedSVD_Prim, self).__init__(name='TruncatedSVD')
        self.id = 7
        self.gid = 22
        self.PCA_LAPACK_Prim = []
        self.type = 'FeatureEngine'
        self.description = "Dimensionality reduction using truncated SVD (aka LSA). This transformer performs linear dimensionality reduction by means of truncated singular value decomposition (SVD). Contrary to PCA, this estimator does not center the data before computing the singular value decomposition. This means it can work with scipy.sparse matrices efficiently. In particular, truncated SVD works on term count/tf-idf matrices as returned by the vectorizers in sklearn.feature_extraction.text. In that context, it is known as latent semantic analysis (LSA). This estimator supports two algorithms: a fast randomized SVD solver, and a “naive” algorithm that uses ARPACK as an eigensolver on (X * X.T) or (X.T * X), whichever is more efficient."
        self.hyperparams_run = {'default': True}
        self.pca = TruncatedSVD(n_components=2)
        self.accept_type = 'c_t_krnl'
        self.need_y = False

    def can_accept(self, data):
        if data.shape[1] <= 2:
            return False
        else:
            return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        cols = list(train_x.columns)
        self.pca.fit(train_x)

        train_data_x = self.pca.transform(train_x)
        train_data_x = pd.DataFrame(train_data_x, columns=cols[:train_data_x.shape[1]])

        test_data_x = self.pca.transform(test_x)
        test_data_x = pd.DataFrame(test_data_x, columns=cols[:test_data_x.shape[1]])
        return train_data_x, test_data_x


class RandomTreesEmbeddingPrim(Primitive):
    def __init__(self, random_state=0):
        super(RandomTreesEmbeddingPrim, self).__init__(name='RandomTreesEmbedding')
        self.id = 8
        self.gid = 23
        self.PCA_LAPACK_Prim = []
        self.type = 'FeatureEngine'
        self.description = "FastICA: a fast algorithm for Independent Component Analysis."
        self.hyperparams_run = {'default': True}
        self.pca = RandomTreesEmbedding(random_state=random_state)
        self.accept_type = 'c_t'
        self.need_y = False

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        self.pca.fit(train_x)
        cols = list(train_x.columns)
    
        train_data_x = self.pca.transform(train_x).toarray()
        new_cols = list(map(str, list(range(train_data_x.shape[1]))))
        train_data_x = pd.DataFrame(train_data_x, columns=new_cols)

        test_data_x = self.pca.transform(test_x).toarray()
        new_cols = list(map(str, list(range(test_data_x.shape[1]))))
        test_data_x = pd.DataFrame(test_data_x, columns=new_cols)
        return train_data_x, test_data_x

class PCA_ARPACK_Prim(Primitive):
    def __init__(self, random_state=0):
        super(PCA_ARPACK_Prim, self).__init__(name='PCA_ARPACK')
        self.id = 4
        self.gid = 24
        self.PCA_LAPACK_Prim = []
        self.type = 'FeatureEngine'
        self.description = "ARPACK principal component analysis (PCA). Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space. It uses the LAPACK implementation of the full SVD or a randomized truncated SVD by the method of Halko et al. 2009, depending on the shape of the input data and the number of components to extract."
        self.pca = PCA(svd_solver='arpack',n_components=2)
        self.accept_type = 'c_t_arpck'
        self.need_y = False

    def can_accept(self, data):
        can_num = len(data.columns) > 4
        return self.can_accept_c(data) and can_num

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        cols = list(train_x.columns)
        self.pca.fit(train_x)

        train_data_x = self.pca.transform(train_x)
        train_data_x = pd.DataFrame(train_data_x, columns=cols[:train_data_x.shape[1]])

        test_data_x = self.pca.transform(test_x)
        test_data_x = pd.DataFrame(test_data_x, columns=cols[:test_data_x.shape[1]])
        return train_data_x, test_data_x


class PCA_LAPACK_Prim(Primitive):
    def __init__(self, random_state=0):
        super(PCA_LAPACK_Prim, self).__init__(name='PCA_LAPACK')
        self.id = 3
        self.gid = 25
        self.PCA_LAPACK_Prim = []
        self.type = 'FeatureEngine'
        self.description = "LAPACK principal component analysis (PCA). Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space. It uses the LAPACK implementation of the full SVD or a randomized truncated SVD by the method of Halko et al. 2009, depending on the shape of the input data and the number of components to extract."
        self.pca = PCA(svd_solver='full')  # n_components=0.9
        self.accept_type = 'c_t'
        self.need_y = False

    def can_accept(self, data):
        can_num = len(data.columns) > 4
        return self.can_accept_c(data) and can_num

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        
        cols = list(train_x.columns)
        self.pca.fit(train_x)

        train_data_x = self.pca.transform(train_x)
        train_data_x = pd.DataFrame(train_data_x, columns=cols[:train_data_x.shape[1]])

        test_data_x = self.pca.transform(test_x)
        test_data_x = pd.DataFrame(test_data_x, columns=cols[:test_data_x.shape[1]])
        return train_data_x ,test_data_x



