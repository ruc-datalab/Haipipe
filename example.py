from haipipe.HAIPipe import *

support_model = ['RandomForestClassifier', 'KNeighborsClassifier', 'LogisticRegression', 'SVC']
def quick_start(notebook_path, dataset_path, label_index, model, hai_program_save_path='hai_program.py'):
    """
    This function is a quick start for the HAIPipe.

    Parameters:
    -----------
    notebook_path: str
        The path to the HI-program notebook file.
    dataset_path: str
        The path to the dataset file.
    label_index: int
        The index of the label column in the dataset file.
    model: str
        The model name and now it only supports those in "support_model".
    hai_program_save_path: str
        The path to save the generated HAI-program.
    """
    
    hai_pipe = HAIPipe(notebook_path, dataset_path, label_index, model)
    hai_pipe.evaluate_hi()
    hai_pipe.generate_aipipe()
    hai_pipe.combine()
    hai_pipe.select_best_hai_by_al()
    hai_pipe.output(hai_program_save_path,save_fig=True)

if __name__ == "__main__":
    notebook_path = 'data/notebook/datascientist25_gender-recognition-by-voice-using-machine-learning.ipynb'
    dataset_path = 'data/dataset/primaryobjects_voicegender/voice.csv'
    label_index = 20
    quick_start(notebook_path, dataset_path, label_index, support_model[2])