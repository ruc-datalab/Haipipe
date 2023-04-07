import json
import numpy as np
import os
import shutil
from aipipe.core.tester import Tester
from aipipe.core.config import Config
from aipipe.core.agent.dqn import DQNAgent
from aipipe.core.env.enviroment import Environment
class AIPipe:
    def __init__(self, data_path, label_index):
        """
        data_path: the path of the dataset
        label_index: the index of the label column

        model is saved in information files.
        """
        self.config = Config()
        self.agent = DQNAgent(self.config.version, self.config)
        self.env = Environment(self.config,train=False)
        self.tester = Tester(self.agent, self.env, 0, self.config)
        self.data_path = data_path
        self.label_index =label_index

    def inference(self):
        return self.tester.inference(self.data_path)