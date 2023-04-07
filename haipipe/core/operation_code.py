OperationType = {
    'Scaler':[
        'MinMaxScaler',
        'MaxAbsScaler',
        'RobustScaler',
        'StandardScaler',
        'QuantileTransformer',
        'PowerTransformer',
        'Normalizer',
        'KBinsDiscretizer'
    ],
    'FeatureEngine':[
        'PolynomialFeatures',
        'InteractionFeatures',
        'PCA_LAPACK',
        'PCA_ARPACK',
        'PCA_Randomized',
        'IncrementalPCA',
        'KernelPCA',
        'TruncatedSVD',
        'RandomTreesEmbedding'
    ],
    'FeatureEngine_simple':[
        'PolynomialFeatures',
        'InteractionFeatures',
        'PCA_AUTO',
        'IncrementalPCA',
        'KernelPCA',
        'TruncatedSVD',
        'RandomTreesEmbedding'
    ],
    'FeatureSelection':[
        'VarianceThreshold',
        # 'UnivariateSelectChiKbest',
        # 'f_classifKbest',
        # 'mutual_info_classifKbest',
        # 'f_classifPercentile',
        # 'mutual_info_classifPercentile',

    ]
}

EdgeOperationType = {
    'FitTransform':[
        'fit_transform',
        'transform',
        'fit'
    ],
    'Cleaning':[
        'confusion_matrix',
        # 'sum',
        'remove',
        'append',
        'drop',
        'unstack',
        'reshape',
        'replace',
        'drop_duplicates',
        'groupby',
        'merge',
        'reset_index',
        'join',
        'sort_values',
        'concat'
    ],
    'Need':[
        'get_dummies',
        'fillna',
        'dropna'
    ]
}


OperationCode = {
    'MinMaxScaler':{
        "pre_code":
        """
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
def catch_num(data):
    num_cols = [col for col in data.columns if str(data[col].dtypes) != 'object']
    cat_cols = [col for col in data.columns if col not in num_cols]
    cat_train_x = data[cat_cols]
    num_train_x = data[num_cols]
    return cat_train_x, num_train_x
        """,

        "code":
        """
add_scaler = MinMaxScaler()
-[PLACEHOLDER]- = pd.DataFrame(-[PLACEHOLDER]-).reset_index(drop=True).infer_objects()
cat_train_x, num_train_x = catch_num(-[PLACEHOLDER]-)
add_scaler.fit(num_train_x)
num_train_x = pd.DataFrame(add_scaler.transform(num_train_x), columns=list(num_train_x.columns)).reset_index(drop=True).infer_objects()
-[PLACEHOLDER]- = pd.concat([cat_train_x.reset_index(drop=True), num_train_x.reset_index(drop=True)],axis=1)
        """
    }, 



    'MaxAbsScaler':{
        "pre_code":
        """
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd
import numpy as np
def catch_num(data):
    num_cols = [col for col in data.columns if str(data[col].dtypes) != 'object']
    cat_cols = [col for col in data.columns if col not in num_cols]
    cat_train_x = data[cat_cols]
    num_train_x = data[num_cols]
    return cat_train_x, num_train_x
        """,

        "code":
        """
add_scaler = MaxAbsScaler()
-[PLACEHOLDER]- = pd.DataFrame(-[PLACEHOLDER]-).reset_index(drop=True).infer_objects()
cat_train_x, num_train_x = catch_num(-[PLACEHOLDER]-)
add_scaler.fit(num_train_x)
num_train_x = pd.DataFrame(add_scaler.transform(num_train_x), columns=list(num_train_x.columns)).reset_index(drop=True).infer_objects()
-[PLACEHOLDER]- = pd.concat([cat_train_x.reset_index(drop=True), num_train_x.reset_index(drop=True)],axis=1)
        """
    }, 


    'RobustScaler':{
        "pre_code":
        """
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
def catch_num(data):
    num_cols = [col for col in data.columns if str(data[col].dtypes) != 'object']
    cat_cols = [col for col in data.columns if col not in num_cols]
    cat_train_x = data[cat_cols]
    num_train_x = data[num_cols]
    return cat_train_x, num_train_x
        """,

        "code":
        """
add_scaler = RobustScaler()
-[PLACEHOLDER]- = pd.DataFrame(-[PLACEHOLDER]-).reset_index(drop=True).infer_objects()
cat_train_x, num_train_x = catch_num(-[PLACEHOLDER]-)
add_scaler.fit(num_train_x)
num_train_x = pd.DataFrame(add_scaler.transform(num_train_x), columns=list(num_train_x.columns)).reset_index(drop=True).infer_objects()
-[PLACEHOLDER]- = pd.concat([cat_train_x.reset_index(drop=True), num_train_x.reset_index(drop=True)],axis=1)
        """
    }, 


    'StandardScaler':{
        "pre_code":
        """
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
def catch_num(data):
    num_cols = [col for col in data.columns if str(data[col].dtypes) != 'object']
    cat_cols = [col for col in data.columns if col not in num_cols]
    cat_train_x = data[cat_cols]
    num_train_x = data[num_cols]
    return cat_train_x, num_train_x
        """,

        "code":
        """
add_scaler = StandardScaler()
-[PLACEHOLDER]- = pd.DataFrame(-[PLACEHOLDER]-).reset_index(drop=True).infer_objects()
cat_train_x, num_train_x = catch_num(-[PLACEHOLDER]-)
add_scaler.fit(num_train_x)
num_train_x = pd.DataFrame(add_scaler.transform(num_train_x), columns=list(num_train_x.columns)).reset_index(drop=True).infer_objects()
-[PLACEHOLDER]- = pd.concat([cat_train_x.reset_index(drop=True), num_train_x.reset_index(drop=True)],axis=1)
        """
    },


    'QuantileTransformer':{
        "pre_code":
        """
from sklearn.preprocessing import QuantileTransformer
import pandas as pd
import numpy as np
def catch_num(data):
    num_cols = [col for col in data.columns if str(data[col].dtypes) != 'object']
    cat_cols = [col for col in data.columns if col not in num_cols]
    cat_train_x = data[cat_cols]
    num_train_x = data[num_cols]
    return cat_train_x, num_train_x
        """,

        "code":
        """
add_scaler = QuantileTransformer()
-[PLACEHOLDER]- = pd.DataFrame(-[PLACEHOLDER]-).reset_index(drop=True).infer_objects()
cat_train_x, num_train_x = catch_num(-[PLACEHOLDER]-)
add_scaler.fit(num_train_x)
num_train_x = pd.DataFrame(add_scaler.transform(num_train_x), columns=list(num_train_x.columns)).reset_index(drop=True).infer_objects()
-[PLACEHOLDER]- = pd.concat([cat_train_x.reset_index(drop=True), num_train_x.reset_index(drop=True)],axis=1)
        """
    }, 



    'PowerTransformer':{
        "pre_code":
        """
from sklearn.preprocessing import PowerTransformer
import pandas as pd
import numpy as np
def catch_num(data):
    num_cols = [col for col in data.columns if str(data[col].dtypes) != 'object']
    cat_cols = [col for col in data.columns if col not in num_cols]
    cat_train_x = data[cat_cols]
    num_train_x = data[num_cols]
    return cat_train_x, num_train_x
        """,

        "code":
        """
add_scaler = PowerTransformer()
-[PLACEHOLDER]- = pd.DataFrame(-[PLACEHOLDER]-).reset_index(drop=True).infer_objects()
cat_train_x, num_train_x = catch_num(-[PLACEHOLDER]-)
add_scaler.fit(num_train_x)
num_train_x = pd.DataFrame(add_scaler.transform(num_train_x), columns=list(num_train_x.columns)).reset_index(drop=True).infer_objects()
-[PLACEHOLDER]- = pd.concat([cat_train_x.reset_index(drop=True), num_train_x.reset_index(drop=True)],axis=1)
        """
    }, 


    'Normalizer':{
        "pre_code":
        """
from sklearn.preprocessing import Normalizer
import pandas as pd
import numpy as np
def catch_num(data):
    num_cols = [col for col in data.columns if str(data[col].dtypes) != 'object']
    cat_cols = [col for col in data.columns if col not in num_cols]
    cat_train_x = data[cat_cols]
    num_train_x = data[num_cols]
    return cat_train_x, num_train_x
        """,

        "code":
        """
add_scaler = Normalizer()
-[PLACEHOLDER]- = pd.DataFrame(-[PLACEHOLDER]-).reset_index(drop=True).infer_objects()
cat_train_x, num_train_x = catch_num(-[PLACEHOLDER]-)
add_scaler.fit(num_train_x)
num_train_x = pd.DataFrame(add_scaler.transform(num_train_x), columns=list(num_train_x.columns)).reset_index(drop=True).infer_objects()
-[PLACEHOLDER]- = pd.concat([cat_train_x.reset_index(drop=True), num_train_x.reset_index(drop=True)],axis=1)
        """
    },

    'KBinsDiscretizer':{
        "pre_code":
        """
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
def catch_num(data):
    num_cols = [col for col in data.columns if str(data[col].dtypes) != 'object']
    cat_cols = [col for col in data.columns if col not in num_cols]
    cat_train_x = data[cat_cols]
    num_train_x = data[num_cols]
    return cat_train_x, num_train_x
        """,

        "code":
        """
-[PLACEHOLDER]- = pd.DataFrame(-[PLACEHOLDER]-).reset_index(drop=True).infer_objects()
add_scaler = ColumnTransformer([("discrit", KBinsDiscretizer(encode='ordinal'), list(-[PLACEHOLDER]-.columns))])
cat_train_x, num_train_x = catch_num(-[PLACEHOLDER]-)
add_scaler.fit(num_train_x)
num_train_x = pd.DataFrame(add_scaler.transform(num_train_x), columns=list(num_train_x.columns)).reset_index(drop=True).infer_objects()
-[PLACEHOLDER]- = pd.concat([cat_train_x.reset_index(drop=True), num_train_x.reset_index(drop=True)],axis=1)
        """
    },

###################################################

    'PolynomialFeatures':{
        "pre_code":
        """
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
        """,

        "code":
        """
-[PLACEHOLDER]- = pd.DataFrame(-[PLACEHOLDER]-).reset_index(drop=True).infer_objects()
add_engine = PolynomialFeatures(include_bias=False)
add_engine.fit(-[PLACEHOLDER]-)
train_data_x = add_engine.transform(-[PLACEHOLDER]-)
train_data_x = pd.DataFrame(train_data_x)
-[PLACEHOLDER]- = train_data_x.loc[:, ~train_data_x.columns.duplicated()]
        """
    },



    'InteractionFeatures':{
        "pre_code":
        """
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
        """,

        "code":
        """
-[PLACEHOLDER]- = pd.DataFrame(-[PLACEHOLDER]-).reset_index(drop=True).infer_objects()
add_engine = PolynomialFeatures(interaction_only=True, include_bias=False)
add_engine.fit(-[PLACEHOLDER]-)
train_data_x = add_engine.transform(-[PLACEHOLDER]-)
train_data_x = pd.DataFrame(train_data_x)
-[PLACEHOLDER]- = train_data_x.loc[:, ~train_data_x.columns.duplicated()]
        """
    },
'PCA_AUTO':{
        "pre_code":
        """
-[PLACEHOLDER]- = pd.DataFrame(-[PLACEHOLDER]-).reset_index(drop=True).infer_objects()
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
        """,

        "code":
        """
-[PLACEHOLDER]- = pd.DataFrame(-[PLACEHOLDER]-).reset_index(drop=True).infer_objects()
add_engine = PCA(svd_solver='auto',n_components=2)
cols = list(-[PLACEHOLDER]-.columns)
add_engine.fit(-[PLACEHOLDER]-)

train_data_x = add_engine.transform(-[PLACEHOLDER]-)
-[PLACEHOLDER]- = pd.DataFrame(train_data_x, columns=cols[:train_data_x.shape[1]])
        """
    },

    'PCA_LAPACK':{
        "pre_code":
        """
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
        """,

        "code":
        """
-[PLACEHOLDER]- = pd.DataFrame(-[PLACEHOLDER]-).reset_index(drop=True).infer_objects()
add_engine = PCA(svd_solver='full')
cols = list(-[PLACEHOLDER]-.columns)
add_engine.fit(-[PLACEHOLDER]-)

train_data_x = add_engine.transform(-[PLACEHOLDER]-)
-[PLACEHOLDER]- = pd.DataFrame(train_data_x, columns=cols[:train_data_x.shape[1]])
        """
    },


    'PCA_ARPACK':{
        "pre_code":
        """
-[PLACEHOLDER]- = pd.DataFrame(-[PLACEHOLDER]-).reset_index(drop=True).infer_objects()
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
        """,

        "code":
        """
-[PLACEHOLDER]- = pd.DataFrame(-[PLACEHOLDER]-).reset_index(drop=True).infer_objects()
add_engine = PCA(svd_solver='arpack',n_components=2)
cols = list(-[PLACEHOLDER]-.columns)
add_engine.fit(-[PLACEHOLDER]-)

train_data_x = add_engine.transform(-[PLACEHOLDER]-)
-[PLACEHOLDER]- = pd.DataFrame(train_data_x, columns=cols[:train_data_x.shape[1]])
        """
    },

    'PCA_Randomized':{
        "pre_code":
        """
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
        """,

        "code":
        """
-[PLACEHOLDER]- = pd.DataFrame(-[PLACEHOLDER]-).reset_index(drop=True).infer_objects()
add_engine = PCA(svd_solver='randomized')
cols = list(-[PLACEHOLDER]-.columns)
add_engine.fit(-[PLACEHOLDER]-)
train_data_x = add_engine.transform(-[PLACEHOLDER]-)
-[PLACEHOLDER]- = pd.DataFrame(train_data_x, columns=cols[:train_data_x.shape[1]])
        """
    },
    


    'IncrementalPCA':{
        "pre_code":
        """
from sklearn.decomposition import IncrementalPCA
import pandas as pd
import numpy as np
        """,

        "code":
        """
-[PLACEHOLDER]- = pd.DataFrame(-[PLACEHOLDER]-).reset_index(drop=True).infer_objects()
add_engine = IncrementalPCA()
cols = list(-[PLACEHOLDER]-.columns)
add_engine.fit(-[PLACEHOLDER]-)

train_data_x = add_engine.transform(-[PLACEHOLDER]-)
-[PLACEHOLDER]- = pd.DataFrame(train_data_x, columns=cols[:train_data_x.shape[1]])
        """
    },




    'KernelPCA':{
        "pre_code":
        """
from sklearn.decomposition import KernelPCA
import pandas as pd
import numpy as np
        """,

        "code":
        """
-[PLACEHOLDER]- = pd.DataFrame(-[PLACEHOLDER]-).reset_index(drop=True).infer_objects()
add_engine = KernelPCA(n_components=2)
cols = list(-[PLACEHOLDER]-.columns)
add_engine.fit(-[PLACEHOLDER]-)

train_data_x = add_engine.transform(-[PLACEHOLDER]-)
-[PLACEHOLDER]- = pd.DataFrame(train_data_x, columns=cols[:train_data_x.shape[1]])
        """
    },



    'TruncatedSVD':{
        "pre_code":
        """
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
        """,

        "code":
        """
-[PLACEHOLDER]- = pd.DataFrame(-[PLACEHOLDER]-).reset_index(drop=True).infer_objects()
add_engine = TruncatedSVD(n_components=2)
cols = list(-[PLACEHOLDER]-.columns)
add_engine.fit(-[PLACEHOLDER]-)

train_data_x = add_engine.transform(-[PLACEHOLDER]-)
-[PLACEHOLDER]- = pd.DataFrame(train_data_x, columns=cols[:train_data_x.shape[1]])
        """
    },



    'RandomTreesEmbedding':{
        "pre_code":
        """
from sklearn.ensemble import RandomTreesEmbedding
import pandas as pd
import numpy as np
        """,

        "code":
        """
-[PLACEHOLDER]- = pd.DataFrame(-[PLACEHOLDER]-).reset_index(drop=True).infer_objects()
add_engine = RandomTreesEmbedding(random_state=0)
cols = list(-[PLACEHOLDER]-.columns)
add_engine.fit(-[PLACEHOLDER]-)

train_data_x = add_engine.transform(-[PLACEHOLDER]-).toarray()
new_cols = list(map(str, list(range(train_data_x.shape[1]))))
-[PLACEHOLDER]- = pd.DataFrame(train_data_x, columns=new_cols)
        """
    },
######################################
    'VarianceThreshold':{
        "pre_code":
        """
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np
from itertools import compress
        """,

        "code":
        """
-[PLACEHOLDER]- = pd.DataFrame(-[PLACEHOLDER]-).reset_index(drop=True).infer_objects()
add_seletcion = VarianceThreshold()
add_seletcion.fit(-[PLACEHOLDER]-)

cols = list(-[PLACEHOLDER]-.columns)
mask = add_seletcion.get_support(indices=False)
final_cols = list(compress(cols, mask))
-[PLACEHOLDER]- = pd.DataFrame(add_seletcion.transform(-[PLACEHOLDER]-), columns=final_cols)
        """
    },


    'UnivariateSelectChiKbest':{
        "pre_code":
        """
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd
import numpy as np
        """,

        "code":
        """
k = 10
-[PLACEHOLDER]- = pd.DataFrame(-[PLACEHOLDER]-).reset_index(drop=True).infer_objects()
if -[PLACEHOLDER]-.shape[1] < k: k = 'all'
add_seletcion = SelectKBest(chi2, k=k)
add_seletcion.fit(-[PLACEHOLDER]-, train_y)

cols = list(-[PLACEHOLDER]-.columns)
mask = add_seletcion.get_support(indices=False)
final_cols = list(compress(cols, mask))
-[PLACEHOLDER]- = pd.DataFrame(add_seletcion.transform(-[PLACEHOLDER]-), columns=final_cols)
        """
    },


    'f_classifKbest':{
        "pre_code":
        """
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import numpy as np
        """,

        "code":
        """
k = 10
-[PLACEHOLDER]- = pd.DataFrame(-[PLACEHOLDER]-).reset_index(drop=True).infer_objects()
if -[PLACEHOLDER]-.shape[1] < k: k = 'all'
add_seletcion = SelectKBest(f_classif, k=k)
add_seletcion.fit(-[PLACEHOLDER]-, train_y)

cols = list(-[PLACEHOLDER]-.columns)
mask = add_seletcion.get_support(indices=False)
final_cols = list(compress(cols, mask))
-[PLACEHOLDER]- = pd.DataFrame(add_seletcion.transform(-[PLACEHOLDER]-), columns=final_cols)
        """
    },


    'mutual_info_classifKbest':{
        "pre_code":
        """
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import pandas as pd
import numpy as np
        """,

        "code":
        """
k = 10
-[PLACEHOLDER]- = pd.DataFrame(-[PLACEHOLDER]-).reset_index(drop=True).infer_objects()
if -[PLACEHOLDER]-.shape[1] < k: k = 'all'
add_seletcion = SelectKBest(mutual_info_classif, k=k)
add_seletcion.fit(-[PLACEHOLDER]-, train_y)

cols = list(-[PLACEHOLDER]-.columns)
mask = add_seletcion.get_support(indices=False)
final_cols = list(compress(cols, mask))
-[PLACEHOLDER]- = pd.DataFrame(add_seletcion.transform(-[PLACEHOLDER]-), columns=final_cols)
        """
    },


    'f_classifPercentile':{
        "pre_code":
        """
from sklearn.feature_selection import SelectPercentile, f_classif
import pandas as pd
import numpy as np
        """,

        "code":
        """
k = 10
-[PLACEHOLDER]- = pd.DataFrame(-[PLACEHOLDER]-).reset_index(drop=True).infer_objects()
if -[PLACEHOLDER]-.shape[1] < k: k = 'all'
add_seletcion = SelectPercentile(f_classif)
add_seletcion.fit(-[PLACEHOLDER]-, train_y)

cols = list(-[PLACEHOLDER]-.columns)
mask = add_seletcion.get_support(indices=False)
final_cols = list(compress(cols, mask))
-[PLACEHOLDER]- = pd.DataFrame(add_seletcion.transform(-[PLACEHOLDER]-), columns=final_cols)
        """
    },



    'mutual_info_classifPercentile':{
        "pre_code":
        """
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
import pandas as pd
import numpy as np
        """,

        "code":
        """
k = 10
-[PLACEHOLDER]- = pd.DataFrame(-[PLACEHOLDER]-).reset_index(drop=True).infer_objects()
if -[PLACEHOLDER]-.shape[1] < k: k = 'all'
add_seletcion = SelectPercentile(mutual_info_classif)
add_seletcion.fit(-[PLACEHOLDER]-, train_y)

cols = list(-[PLACEHOLDER]-.columns)
mask = add_seletcion.get_support(indices=False)
final_cols = list(compress(cols, mask))
-[PLACEHOLDER]- = pd.DataFrame(add_seletcion.transform(-[PLACEHOLDER]-), columns=final_cols)
        """
    },
    
    'UnivariateSelectChiFPRPrim':{
        "pre_code":
        """
from sklearn.feature_selection import SelectFpr, chi2
import pandas as pd
import numpy as np
        """,

        "code":
        """
k = 10
-[PLACEHOLDER]- = pd.DataFrame(-[PLACEHOLDER]-).reset_index(drop=True).infer_objects()
if -[PLACEHOLDER]-.shape[1] < k: k = 'all'
add_seletcion = SelectFpr(chi2, alpha=0.05)
add_seletcion.fit(-[PLACEHOLDER]-, train_y)

cols = list(-[PLACEHOLDER]-.columns)
mask = add_seletcion.get_support(indices=False)
final_cols = list(compress(cols, mask))
-[PLACEHOLDER]- = pd.DataFrame(add_seletcion.transform(-[PLACEHOLDER]-), columns=final_cols)
        """
    },

}
