import csv 
import os
import torch
import random
import numpy as np
from typing import Union, Dict, List

                            # {'video_name': [acc, avg, [frame_num, real_prob, fake_prob, pred]]} 
def csv_writer(result_dict: Dict[str, List[Union[float, Union[int, float]]]], 
               model_name: str, 
               csv_save_path: str
              ):
                                 
    csv_save_path = '/media/data1/inho/df_detection/inference_csv' if csv_save_path is None else csv_save_path
    os.makedirs(csv_save_path, exist_ok=True)
    
    for k in result_dict.keys():
        acc = result_dict[k][0]
        avg = result_dict[k][1]
        data = result_dict[k][2]
    
        save_path = os.path.join(csv_save_path, k)
        os.makedirs(save_path, exist_ok=True)

        OF = open(f'{save_path}/{acc}_{avg}_{model_name}.csv','w')
        for row in data:
            OF.write(','.join(map(str, row))+'\n') # frame_num, real_prob, fake_prob, pred
        OF.close()

def torch_seed(random_seed: int = 42):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    

############################################################### Discarded
    
    # def csv_writer(result: Union[int, float], # frame_num, real_prob, fake_prob, pred 
#                frame_path: str, 
#                model_name: str, 
#                avg_prob: float, 
#                csv_save_path: str = None
#               ):
#     csv_save_path = '/media/data1/inho/df_detection/inference_csv' if csv_save_path is None else csv_save_path
#     os.makedirs(csv_save_path, exist_ok=True)
#     video_name = frame_path.split('/')[-2]
#     save_path = os.path.join(csv_save_path, video_name)
#     os.makedirs(save_path, exist_ok=True)
    
#     OF = open(f'{save_path}/{avg_prob}_{model_name}.csv','w')
#     for row in result:
#         OF.write(','.join(map(str, row))+'\n') # frame_num, real_prob, fake_prob, pred
#     OF.close()