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

import os
import argparse

import timm
import torch
#import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision

def transfer_weights(modelchoice: str, weight_path: str, num_out_classes=2):
    modelchoice = modelchoice
    if modelchoice == 'xception':
        model = timm.create_model('xception', pretrained=False)
    # finetuning for bianry classification
        in_features = model.fc.in_features # 1차원 2048 벡터
        model.fc = nn.Linear(in_features, num_out_classes) # 출력층 fine-tuning
        # del model.fc # last_linear가 추가된 상태라서, 원래 있던 fc layer제거

    elif modelchoice == 'resnet50':
        model = torchvision.models.resnet50(pretrained=False)
        # Replace fc
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_out_classes)            
    else:
        raise Exception('Choose valid model, e.g. ADD, QAD, cored, xception...')

        # fc layer 이름 바뀜을 방지
    # if modelchoice == 'xception' and model.last_linear is not None:
    #     model.fc = model.last_linear
    #     del model.last_linear
    
    if weight_path is not None:
        state_dict = torch.load(weight_path)
        if 'state_dict' in state_dict.keys():
            weights = state_dict['state_dict']
        else:
            weights = state_dict
        # model.load_state_dict(weights)
    
    if modelchoice == 'xception' and 'fc.weight' not in weights.keys():
        renamed_weight = {}
        for k,v in weights.items():
            new_k = k.split('module.')[-1]
            renamed_weight[new_k] = v
            
        weights = renamed_weight

        weights['fc.weight'] = weights['last_linear.weight']
        weights['fc.bias'] = weights['last_linear.bias']
        del weights['last_linear.weight']
        del weights['last_linear.bias']
        
    model.load_state_dict(weights)
        
    return model

def model_selection(model_name: str, weight_path: str, num_out_classes: int):
    """
    :param modelname:
    :return: model, image size, pretraining<yes/no>, input_list
    """
    if model_name == 'xception' or model_name == 'cored':
        return transfer_weights(modelchoice='xception',
                             weight_path = weight_path,
                             num_out_classes=num_out_classes)

    elif 'qad' in model_name.lower() or 'add' in model_name.lower():
        return transfer_weights(modelchoice='resnet50',
                             weight_path = weight_path,
                             num_out_classes=num_out_classes)

    else:
        raise NotImplementedError(model_name)