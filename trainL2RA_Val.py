import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.optim import lr_scheduler
from torchvision import transforms
from PIL import Image
import pandas as pd
import argparse
from transformers import BertTokenizer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from model_L2RA_Net import UnetPlusPlusOrigin, UnetPlusPlusDecoder, ClassifierCroSelfAttn3_1
import clip
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import random
import shutil
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with classifier only")
    parser.add_argument('--model_path_classifier', type=str, required=True, help="Path to full classifier weights")
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint_gap', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()

def initialize_model(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    net, preprocess = clip.load("RN50", device=device, jit=False)
    for param in net.parameters():
        param.requires_grad = False

    classification_loss_func = nn.CrossEntropyLoss()
    classifier = ClassifierCroSelfAttn3_1(input_dim=1024, output_dim=1024).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    return net, optimizer, scheduler, classification_loss_func, classifier, device, preprocess

def load_classifier_weights(classifier, checkpoint_path, device):
    print(f"Loading classifier weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state_dict[k] = v

    classifier.load_state_dict(new_state_dict)
    print("Classifier weights loaded.")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

class YourDataset(data.Dataset):
    def __init__(self, txt_file, root_dir, is_train, preprocess):
        self.root_dir = root_dir
        self.data = pd.read_csv(txt_file, sep='\t').dropna(axis=0, how='any')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.is_train = is_train
        self.label_groups = self._group_by_label()
        self.label_mapping = {1: 0, 3: 1, 5: 2, 7: 3, 9: 4}

    def __len__(self):
        return len(self.data)

    def _group_by_label(self):
        grouped_data = {}
        dataList = self.data.sort_values(by='label')
        for _, row in dataList.iterrows():
            label = row['label']
            if label not in grouped_data:
                grouped_data[label] = []
            grouped_data[label].append(row)
        return grouped_data

    def get_other_label_groups(self, label):
        other_labels = [lbl for lbl in self.label_groups if lbl != label]
        selected_texts = []
        for lbl in other_labels:
            selected_texts.extend([row['text'][:77] for row in random.sample(self.label_groups[lbl], 1)])
        return selected_texts

    def __getitem__(self, idx):
        d = self.data.iloc[idx]
        text = d['text'][:77]
        img_path = os.path.join(self.root_dir, d['file_path'])

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise

        if self.transform:
            image = self.transform(image)

        text_input = clip.tokenize(text).squeeze(0)
        original_label = d['label']
        mapped_label = self.label_mapping[original_label]

        if self.is_train:
            other_label_group = self.get_other_label_groups(d['label'])
            return image, torch.tensor(mapped_label), text_input, mapped_label, other_label_group
        else:
            return image, torch.tensor(mapped_label), text_input, mapped_label, img_path  # ← 加上 img_path！


def train_model(args, net, optimizer, scheduler, classification_loss_func, classifier, device, preprocess):
    # 构建验证集
    test_dataset = YourDataset(
        txt_file='./dataset/grade/val_data1.csv',
        root_dir="./dataset/grade/",
        is_train=False, preprocess=preprocess
    )
    test_dataloader = data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8)
    val_num = len(test_dataset)
    print(f"val_num: {val_num}")

    # 加载分类器参数
    load_classifier_weights(classifier, args.model_path_classifier, device)
    classifier.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, label_tokens, text_tokens, class_label, img_paths in test_dataloader:
            images, label_tokens, text_tokens, class_label = (
                images.to(device),
                label_tokens.to(device),
                text_tokens.to(device),
                class_label.to(device)
            )

            # 推理预测
            outputs = classifier(images, train=False)
            _, predicted = torch.max(outputs.data, 1)

            total += class_label.size(0)
            correct += (predicted == class_label).sum().item()

    # 计算平均准确率
    accuracy = 100 * correct / total
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Average accuracy: {accuracy:.2f}%")

    return accuracy

def main():
    args = parse_args()
    net, optimizer, scheduler, classification_loss_func, classifier, device, preprocess = initialize_model(args)
    train_model(args, net, optimizer, scheduler, classification_loss_func, classifier, device, preprocess)

if __name__ == "__main__":
    main()
