
from aipipe.core.env.primitives.encoder import *
from aipipe.core.env.primitives.fengine import *
from aipipe.core.env.primitives.fpreprocessing import *
from aipipe.core.env.primitives.fselection import *
from aipipe.core.env.primitives.imputernum import *
from aipipe.core.env.primitives.predictor import *
from aipipe.core.env.metric import *
import json
import math
import os
class Config:
    version = 0
    
    ### hyperparameters for DQN
    env: str = None
    gamma: float = 0
    learning_rate: float = 1*1e-5
    frames: int = 100000
    test_frames: int = 6
    episodes: int = 1
    max_buff: int = 2000
    batch_size: int = 100
    logic_batch_size: int = 20
    column_num: int = 20
    column_num: int = 100
    column_feature_dim: int = 19
    epsilon: float = 1
    eps_decay: float = 500
    epsilon_min: float = 0.4
    # state dim
    data_dim: int = column_num*column_feature_dim
    prim_state_dim: int = data_dim + 6 + 1 + 1
    lpip_state_dim: int = data_dim + 1
    # RNN param
    seq_embedding_dim: int = 30
    seq_hidden_size: int = 18
    seq_num_layers: int = 2
    predictor_embedding_dim: int = 10
    lpipeline_embedding_dim: int = 5
    log_interval: int = 200
    print_interval: int = 200
    test_interval: int = 500
    train_interval: int= 100
    update_tar_interval: int = 100
    win_reward: float = 198
    win_break: bool = True
    taskdim: int = 56
    classification_metric_id: int = 1
    regression_metric_id: int = 4
    k_fold = 3


    dataset_path : str = 'data/dataset/'
    output = 'out'
    use_cuda: bool = False
    checkpoint: bool = True
    checkpoint_interval: int = None
    record: bool = False

    logic_pipeline_1 = ['ImputerNum', 'ImputerCat', 'Encoder', 'FeaturePreprocessing', 'FeatureEngine', 'FeatureSelection']
    logic_pipeline_2 = ['ImputerNum', 'ImputerCat', 'Encoder', 'FeaturePreprocessing', 'FeatureSelection', 'FeatureEngine']
    logic_pipeline_3 = ['ImputerNum', 'ImputerCat', 'Encoder', 'FeatureEngine', 'FeatureSelection', 'FeaturePreprocessing']
    logic_pipeline_4 = ['ImputerNum', 'ImputerCat', 'Encoder', 'FeatureEngine', 'FeaturePreprocessing', 'FeatureSelection']
    logic_pipeline_5 = ['ImputerNum', 'ImputerCat', 'Encoder', 'FeatureSelection', 'FeatureEngine', 'FeaturePreprocessing']
    logic_pipeline_6 = ['ImputerNum', 'ImputerCat', 'Encoder', 'FeatureSelection', 'FeaturePreprocessing', 'FeatureEngine']

    
    with open('aipipe/core/jsons/classification_task_dic.json', 'r') as f:
        classification_task_dic = json.load(f)
    
    ### load information files
    train_classification_task_dic = {}
    fold_length = math.ceil(len(classification_task_dic)/k_fold)
    with open('aipipe/core/jsons/test_index.json','r') as f:
        test_index = json.load(f)
    train_index = list(set(classification_task_dic.keys())-set(test_index))

    classifier_predictor_list = [
        RandomForestClassifierPrim(),
        AdaBoostClassifierPrim(),
        BaggingClassifierPrim(),
        BernoulliNBClassifierPrim(),
        # ComplementNBClassifierPrim(),
        DecisionTreeClassifierPrim(),
        ExtraTreesClassifierPrim(),
        GaussianNBClassifierPrim(),
        GaussianProcessClassifierPrim(),
        GradientBoostingClassifierPrim(),
        KNeighborsClassifierPrim(),
        LinearDiscriminantAnalysisPrim(),
        LinearSVCPrim(),
        LogisticRegressionPrim(),
        # LogisticRegressionCVPrim(),
        # MultinomialNBPrim(),
        NearestCentroidPrim(),
        PassiveAggressiveClassifierPrim(),
        # QuadraticDiscriminantAnalysisPrim(),
        RidgeClassifierPrim(),
        RidgeClassifierCVPrim(),
        SGDClassifierPrim(),
        SVCPrim(),
    ]

    metric_list = [
        AccuracyMetric(),
        F1Metric(),
        AucMetric(),
        MseMetric(),
    ]

    dtype_dic = {
        'interval[float64]':4,
        'uint8':1,
        'uint16': 1,
        'int64': 1,
        'int': 1,
        'int32': 1,
        'int16': 1,
        'np.int32': 1,
        'np.int64': 1,
        'np.int': 1,
        'np.int16': 1,
        'float64': 2,
        'float': 2,
        'float32': 2,
        'float16': 2,
        'np.float32': 2,
        'np.float64': 2,
        'np.float': 2,
        'np.float16': 2,
        'str':3,
        'Category':4,
        'object':4,
    }

    ### operations
    imputernums = [ImputerMean(), ImputerMedian(), ImputerNumPrim()]
    encoders = [NumericDataPrim(), LabelEncoderPrim(), OneHotEncoderPrim()] #
    fpreprocessings = [MinMaxScalerPrim(), MaxAbsScalerPrim(), RobustScalerPrim(), StandardScalerPrim(), QuantileTransformerPrim(), PowerTransformerPrim(), NormalizerPrim(), KBinsDiscretizerOrdinalPrim(), Primitive()]
    fengines = [PolynomialFeaturesPrim(), InteractionFeaturesPrim(), PCA_AUTO_Prim(), IncrementalPCA_Prim(), KernelPCA_Prim(), TruncatedSVD_Prim(), RandomTreesEmbeddingPrim(), Primitive()]
    fselections = [VarianceThresholdPrim(), Primitive()]
    lpipelines = [logic_pipeline_1, logic_pipeline_2, logic_pipeline_3, logic_pipeline_4, logic_pipeline_5, logic_pipeline_6]
 
    imputernum_action_dim: int = len(imputernums)
    encoder_action_dim: int = len(encoders)
    fpreprocessing_action_dim: int = len(fpreprocessings)
    fegine_action_dim: int = len(fengines)
    fselection_action_dim: int = len(fselections)
    lpipeline_action_dim: int = len(lpipelines)
    single_action_dim: int = max([imputernum_action_dim, encoder_action_dim, fpreprocessing_action_dim, fegine_action_dim, fselection_action_dim])

    ### result save path
    if not os.path.exists('aipipe/core/logs/'+str(version)+'_more_model'):
        os.mkdir('aipipe/core/logs/'+str(version)+'_more_model')
    if not os.path.exists('aipipe/core/models/'+str(version)+'_more_model'):
        os.mkdir('aipipe/core/models/'+str(version)+'_more_model')
    result_log_file_name: str = 'aipipe/core/logs/'+str(version)+'_more_model'+'/result_log.npy'
    loss_log_file_name: str = 'aipipe/core/logs/'+str(version)+'_more_model'+'/loss_log.npy'
    lp_loss_log_file_name: str = 'aipipe/core/logs/'+str(version)+'_more_model'+'/lp_loss_log.npy'
    test_reward_dic_file_name: str = 'aipipe/core/logs/'+str(version)+'_more_model'+'/test_reward_dict.npy'
    model_dir: str = 'aipipe/core/models/'+str(version)