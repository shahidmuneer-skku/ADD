"""
####################################################################################################
Sungkyunkwan University Property
Authors of the original Deepfake Detection Model: Simon S. Woo, Binh Lee
Integration of the AI model into the AI Summit: Muhammad Shahid Muneer

This project includes an AI detector named ADD, which was accepted and published at:
https://cdn.aaai.org/ojs/19886/19886-13-23899-1-2-20220628.pdf.

License:
This AI model and associated resources are provided to you by Sungkyunkwan University under the 
following license. By obtaining, using, and/or modifying this AI model, you agree to the terms 
and conditions outlined below:

Permission:
Permission to use, copy, modify, and distribute this AI model and its documentation for any purpose 
and without fee or royalty is hereby granted, provided that:
1. The following copyright notice and statements, including this disclaimer, appear in all copies 
   of the AI model and its documentation, including modifications made for internal use or distribution.
2. Attribution is provided to Sungkyunkwan University and the original authors of the AI model.

Disclaimer:
THIS AI MODEL IS PROVIDED "AS IS," AND SUNGKYUNKWAN UNIVERSITY MAKES NO REPRESENTATIONS OR WARRANTIES, 
EXPRESS OR IMPLIED. THIS INCLUDES, BUT IS NOT LIMITED TO, WARRANTIES OF MERCHANTABILITY OR FITNESS 
FOR A PARTICULAR PURPOSE. USE OF THIS AI MODEL IS AT YOUR OWN RISK.

Restrictions:
- The name "Sungkyunkwan University" or its affiliates may not be used in advertising or publicity 
  pertaining to the distribution of the AI model and/or its documentation without explicit permission.
- The AI model may not be used in a manner that infringes any third-party intellectual property rights.

Ownership:
Title to copyright in this AI model, associated resources, and documentation shall remain with 
Sungkyunkwan University. Users agree to preserve this attribution and comply with the terms outlined herein.

For further inquiries, please contact Sungkyunkwan University or authors at swoo@g.skku.edu
####################################################################################################
"""
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