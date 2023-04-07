import pandas as pd

class Primitive:
    def __init__(self, name='blank'):
        self.id = 0
        self.gid = 25 
        self.name = name
        self.description = str(name)
        self.hyperparams = []
        self.type = "blank"

    def fit(self, data):
        pass

    def transform(self,train_x, test_x, train_y):
        return train_x, test_x

    def can_accept(self, data):
        return True

    def can_accept_a(self, data): 
        if data.empty:
            return False
        elif data.shape[1] == 0:
            return False
        num_cols = data._get_numeric_data().columns
        if not len(num_cols) == 0:
            return True
        return False

    def can_accept_b(self, data):
        if data.empty:
            return False
        elif data.shape[1] == 0:
            return False
        return True

    def can_accept_c(self, data, task=None, larpack=False):
        if data.empty:
            return False
        elif data.shape[1] == 0:
            return False
        cols = data
        num_cols = data._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))

        with pd.option_context('mode.use_inf_as_null', True):
            if data.isnull().any().any():
                return False
        if not len(cat_cols) == 0:
            return False
        return True

    def can_accept_c1(self, data, task=None, larpack=False):
        if data.empty:
            return False
        elif data.shape[1] == 0:
            return False
        cols = data
        num_cols = data._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))

        if not len(cat_cols) == 0:
            return False
        return True

    def can_accept_c2(self, data, task=None, larpack=False):
        if data.empty:
            return False
        elif data.shape[1] == 0:
            return False
        cols = data
        num_cols = data._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))

        if not len(num_cols) == 0:
            return False
        return True

    def can_accept_d(self, data, task): 
        if data.empty:
            return False
        elif data.shape[1] == 0:
            return False
        cols = data
        num_cols = data._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))
        if not len(cat_cols) == 0:
            return False

        with pd.option_context('mode.use_inf_as_null', True):
            if data.isnull().any().any():
                return False
            return True

    def is_needed(self, data):
        return True
