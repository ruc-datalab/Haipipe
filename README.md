# HAIPipe

Given a human-generated pipeline (HI-pipeline) for an ML task, HAIPipe introduces a reinforcement learning based approach to search an optimized ML-generated pipeline (ML-pipeline) and adopts an enumeration-sampling strategy to carefully select the best performing combined pipeline(HAI-pipeline).

## Requirements

This code is written in Python. Requirements include

- python = 3.8.12
- NumPy = 1.19.5 
- pandas = 1.1.3
- torch = 1.9.0
- Scikit-learn = 0.23.2

You can install multiple packages:

```
pip install -r requirements.txt
```



## Quick Start

### run example to generate hybrid-pipeline

```
python example.py
```

The file `example.py` is an example. Modify it according to your configuration.

- The **input** contains 1 notebook, 1 CSV file, some information about the ML task (including model and label_index)
- The **output** is the best HAI-program, the accuracy of HI-pipeline and the accuracy of HAI-pipeline.

#### Customized input

You can put new data in the folder `$hybridpipe/data/`and organize them as:

```python
hybridpipe
├── data
│   ├── dataset 
│   │   └── new_folder   #input dataset(csv file)
│   └── notebook #input notebook
├── MLPipeGen
└── HybridPipeGen
```

Then modify the input in example.py.

## Dataset

The whole dataset and notebooks are put in the follow link:
https://www.dropbox.com/s/reqenlqzggpx8bk/haipipe_datasets.zip?dl=0



