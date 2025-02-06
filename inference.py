import torch
from torchvision import transforms
from PIL import Image
import argparse
import yaml
import easydict
import os

import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm
# from transformers import ViTForImageClassification, AutoFeatureExtractor
from pathlib import Path


import glob
import load_model
import torch
import torch.nn as nn
# from generated.unused_code.utils import csv_writer, torch_seed
import warnings
from torchvision import transforms # 이미지 데이터 transform
from torch.utils.data import DataLoader # 이미지 데이터 로더
from tqdm import tqdm
import argparse
from omegaconf import OmegaConf
import logging
import warnings
import yaml
from tqdm import tqdm



# 모든 경고 메시지 무시
warnings.filterwarnings("ignore")
_logger = logging.getLogger('inference') # 이름이 ingerence인 로거 객체 생성

# Seed for reproducibility
# ()
# Argument parser for configuration file and image path
# parser = argparse.ArgumentParser(description='Single Image Inference for Deepfake Detection')
# parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
# parser.add_argument('--image_path', type=str, required=True, help='Path to the image file')

def load_model_add(config, device):
    """Load the pre-trained model from checkpoint."""
    # print(f"Loading model: {bcolors.OKBLUE} {config.model} {bcolors.ENDC}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model.model_selection(model_name="add", weight_path="../weights/ADD/resnet50_kd_valacc_img128_kd54_freq_swd_best.pth", num_out_classes=2)
    # Load weights
    model.to(device)
    model.eval()  # Set model to evaluation mode
    
    return model








def process_image(image_path, image_size):
    """Process a single image for inference."""
    image = Image.open(image_path).convert('RGB')
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    image = transform(image)
    return image



# ---------- DATASET ----------
class CustomImageDataset(Dataset):
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_files = [
            str(f) for f in Path(image_folder).resolve().rglob("*") if str(f).lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = process_image(img_path,299)
        return image, img_path


# ---------- EXECUTE ----------
def execute(model, dataset, batch_size=128, device="cpu"):
    model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    img_paths = []
    predictions = []

    with torch.no_grad():
        for batch in tqdm(loader):
            image = batch[0]
            img_path = batch[1]
            image = image.to(device)
            
            outputs = model(image)
                
            probs = torch.softmax(outputs, dim=1)
            real_prob = probs[0][0].item()
            fake_prob = probs[0][1].item()
            # print(pred)
            # predictions.append(fake_prob)
            # img_paths.append(img_path)
            # Iterate through the batch to collect predictions and image paths
            for i in range(len(probs)):
                fake_prob = probs[i][1].item()  # Fake probability for each image
                predictions.append(fake_prob)  # Append fake_prob for each image
                img_paths.append(img_path[i])  # Append corresponding image path


    return img_paths, predictions


# def run_inference(model, image_tensor, device):
#     """Run inference on a single image tensor."""
#     image_tensor = image_tensor.to(device)
    
#     with torch.no_grad():
#         output = model(image_tensor)
#         prediction = torch.argmax(output, dim=1).item()
    
#     return prediction

def main(config, data_path, batch_size):
    """Main function to set up the model and run inference on a single image."""
    # device, _ = setup_device(config.SYS.num_gpus)
    device= "cuda"

    # Load the pre-trained model
    model = load_model_add(config, device)
    # Process the image
    # image_tensor = process_image(image_path, config.TRAIN.image_size)

    # Run inference
    image_paths, y_pred = execute(model, data_path, batch_size,"cuda")
    
    # Create a DataFrame
    df = pd.DataFrame({'filepath': image_paths, 'prediction': y_pred})

    # Write to CSV
    os.makedirs(Path(opt.result_file_path).parent, exist_ok=True)
    df.to_csv(opt.result_file_path, index=False)
    # Print results
    # print(f"{bcolors.OKGREEN}Prediction: {prediction}{bcolors.ENDC}")

if __name__ == '__main__':
    # args = parser.parse_args()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str, default=None, help="Folder containing images for inference")
    parser.add_argument("--result_file_path", type=str, default="./results/CoDE.csv", help="")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--cuda", action='store_true')
    opt = parser.parse_args()


    config = "./configs/hkrawp_effnet.yaml"
    with open(config, 'r') as cf:
        config = yaml.safe_load(cf)
    config = easydict.EasyDict(config)
    
    
    # Create dataset and dataloader
    dataset = CustomImageDataset(opt.data_path)
    
    main(config, dataset,opt.batch_size)
