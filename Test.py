import argparse
import json
import os
import glob

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from Dataset import *
from Layers import *
from Models import *
from Losses import *
from Metrics import *
from Utils import *
from PyFire import Trainer

with open("Bird\\config.json") as f:
    data = f.read()
config = json.loads(data)

global classifier_dataset_config
classifier_dataset_config = config['classifier_dataset_params']
global classifier_learning_params
classifier_learning_params = config['classifier_learning_params']
global classifier_trainer_params
classifier_trainer_params = config['classifier_trainer_params']
global classifier_model_config
classifier_model_config = config['classifier_model_params']

global separator_dataset_config
separator_dataset_config = config['separator_dataset_params']
global separator_learning_params
separator_learning_params = config['separator_learning_params']
global separator_trainer_params
separator_trainer_params = config['separator_trainer_params']
global separator_model_config
separator_model_config = config['separator_model_params']

global separator_preprocessing
separator_preprocessing = config['data_preprocessing']['separator']
n_src = config['data_preprocessing']['general']['n_src']

root = 'Bird'
if root[-1] != r'/':
    root += r'/'

X_test = torch.load(root+f'Data/SeparatorClosed2Speakers/X_test.pt')
Y_test = torch.load(root+f'Data/SeparatorClosed2Speakers/Y_test.pt')
Y_test_id = torch.load(root+f'Data/SeparatorClosed2Speakers/Y_test_id.pt')

separator_dataset_test = PipelineDataset(X_test, 
										Y_test, 
										Y_test_id, 
										**separator_dataset_config)

separator_dataloader_test = torch.utils.data.DataLoader(separator_dataset_test,
                                                        batch_size=separator_learning_params['batch_size'],
                                                        shuffle=False)
 
model = RepUNet(**separator_model_config)
model.load_state_dict(torch.load('Bird\\Separator\\Models\\saver_epoch100.pt'))

optimizer = optim.AdamW(model.parameters(), lr=separator_learning_params['learning_rate'])

trainer = Trainer(model, optimizer, 
                    loss_func=separator_trainer_params['loss_func'],
                    metric_func=separator_trainer_params['metric_func'],
                    verbose=separator_trainer_params['verbose'],
                    device=separator_trainer_params['device'],
                    dest=root+'Test',
                    **separator_trainer_params['params'])

 
separator_data_test, separator_predictions_test = trainer.evaluate(separator_dataloader_test, 
																	   'test', 
																	   to_device='cuda',
																	   return_data=True)
print(separator_data_test, separator_predictions_test)