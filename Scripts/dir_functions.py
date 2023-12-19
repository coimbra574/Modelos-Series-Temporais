import os
import glob
import shutil
from posixpath import join


def create_validation_folder(df_path_name):
    # Rodar sempre antes da validação, que vem antes do treino
    if not os.path.exists(df_path_name):
        os.mkdir(df_path_name)
        os.mkdir(df_path_name + '/validation')
    else:
        if not os.path.exists(df_path_name + '/validation'):
            os.mkdir(df_path_name + '/validation')
        else:
            shutil.rmtree(df_path_name + '/validation')
            os.mkdir(df_path_name + '/validation')


def erase_train_files(df_path_name):
    for f in glob.glob(df_path_name + '/model_*'):
       os.remove(f)


def create_results_folder(path_name):
    if not os.path.exists(path_name):
        os.mkdir(path_name)
    else:
        shutil.rmtree(path_name.replace('/', '\\'))
        os.mkdir(path_name)

            
def check_model_path(model_path):
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    else:
        for f in glob.glob(model_path + '/*'):
            #os.remove(f)
            shutil.rmtree(f)
            