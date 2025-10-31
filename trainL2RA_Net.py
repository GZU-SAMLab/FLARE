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
from tqdm import tqdm  # 引入 tqdm 库
import numpy as np
from collections import defaultdict
import random
# Set environment variable for CUDA
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Command-line arguments for parameter passing
def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--model_path_lesion', type=str, required=True, help="Path to lesion model weights")
    parser.add_argument('--model_path_leaf', type=str, required=True, help="Path to leaf model weights")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--checkpoint_gap', type=int, default=1, help="Interval for saving checkpoints")
    parser.add_argument('--device', type=str, default='cuda:0',
                        help="Device to run the model on (e.g., 'cpu', 'cuda:0')")
    return parser.parse_args()

# Custom Dataset class
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
        self.label_groups = self._group_by_label()  # Group data by labels only once
        # 定义标签映射规则：{原始标签: 映射后标签}
        self.label_mapping = {1: 0, 3: 1, 5: 2, 7: 3, 9: 4}

    def __len__(self):
        return len(self.data)

    def _group_by_label(self):
        """
        Group the data by labels.
        This method is only called once during initialization.
        """
        grouped_data = {}
        dataList = self.data.sort_values(by='label')
        for _, row in dataList.iterrows():
            label = row['label']
            if label not in grouped_data:
                grouped_data[label] = []
            grouped_data[label].append(row)
        return grouped_data

    def get_other_label_groups(self, label):
        """
        Return texts from all labels except the given label.
        Randomly select data from those labels.
        """
        other_labels = [lbl for lbl in self.label_groups if lbl != label]
        selected_texts = []

        for lbl in other_labels:
            # Randomly pick one sample from each label and extract the 'text' field
            selected_texts.extend([row['text'][:77] for row in random.sample(self.label_groups[lbl], 1)])

        return selected_texts

    def __getitem__(self, idx):
        d = self.data.iloc[idx]
        text = d['text'][:77]  # Truncate to max length for CLIP model
        img_path = os.path.join(self.root_dir, d['file_path'])

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise

        if self.transform:
            image = self.transform(image)

        text_input = clip.tokenize(text).squeeze(0)

        # 获取原始标签并映射
        original_label = d['label']
        mapped_label = self.label_mapping[original_label]
        # print(f"mapped_label:{mapped_label}")

        if self.is_train:
            # Get data from other labels (this is now efficient)
            other_label_group = self.get_other_label_groups(d['label'])
            # print(f"idx:{idx}")
            # print(f"text:{text}")
            # print(f"other_label_group:{other_label_group}")

            return image, torch.tensor(mapped_label), text_input, mapped_label, other_label_group
        else:
            return image, torch.tensor(mapped_label), text_input, mapped_label


# Initialize model and optimizer
def initialize_model(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    net, preprocess = clip.load("RN50", device=device, jit=False)
    for param in net.parameters():
        param.requires_grad = False

    classification_loss_func = nn.CrossEntropyLoss()

    # Initialize Classifier
    classifier = ClassifierCroSelfAttn3_1(input_dim=1024, output_dim=1024).to(device)
    classifier.train()

    optimizer = optim.Adam(classifier.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    return net, optimizer, scheduler, classification_loss_func, classifier, device, preprocess


# Load pre-trained weights for UNet++
def load_pretrained_weights(model, checkpoint_path, model_name, device):
    print(f"Loading {model_name} weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove "module." prefix if trained with DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state_dict[k] = v

    # Filter out unmatched keys
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

    # Load matched weights
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters for {model_name}")

# Training loop
def train_model(args, net, optimizer, scheduler, classification_loss_func, classifier, device, preprocess):
    # Dataset and DataLoader
    train_dataset = YourDataset(
        txt_file='./dataset/grade/train_data1.1.csv',
        root_dir='./dataset/grade/',
        is_train=True, preprocess=preprocess
    )
    train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    train_num = len(train_dataset)

    test_dataset = YourDataset(
        txt_file='./dataset/grade/val_data1.1.csv',
        root_dir="./dataset/grade/",
        is_train=False, preprocess=preprocess
    )
    test_dataloader = data.DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=8)
    val_num = len(test_dataset)
    print(f"val_num:{val_num}, train_num:{train_num}")

    # Load pre-trained weights
    # 加载预训练权重
    load_pretrained_weights(classifier.UnetPlusPlusDecoder, args.model_path_lesion, "lesion model", device)
    load_pretrained_weights(classifier.UnetPlusPlusOrigin, args.model_path_leaf, "leaf model", device)

    # 冻结两个子模块的参数
    for param in classifier.UnetPlusPlusDecoder.parameters():
        param.requires_grad = False
    for param in classifier.UnetPlusPlusOrigin.parameters():
        param.requires_grad = False

    best_accuracy = 0

    # Training loop
    for epoch in range(args.epochs):
        total_loss, batch_num = 0, 0
        train_steps = len(train_dataloader)
        for images, label_tokens, text_tokens, class_label, label_group in tqdm(train_dataloader,
                                                                   desc=f"Epoch {epoch + 1}/{args.epochs}",
                                                                   unit="batch"):
            images, label_tokens, text_tokens, class_label = images.to(device), label_tokens.to(device), text_tokens.to(
                device), class_label.to(device)
            # print(f"class_label:{class_label}, label_group:{label_group}")
            # 对每个标签生成独立的文本特征
            label_group_text_features = []
            for label_data in label_group:
                label_text = label_data  # 获取该标签的文本
                # print(f"label_group:{label_text}")
                label_text_tokens = clip.tokenize(label_text).to(device)  # 对文本进行tokenize并转移到设备
                label_text_feature = net.encode_text(label_text_tokens).to(device)  # 对文本进行编码，获取特征
                label_group_text_features.append(label_text_feature)  # 将每个标签的特征添加到列表中

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                text_features = net.encode_text(text_tokens)
                label, cur_loss = classifier(images, text_features, label_group_text_features, class_label)

                # Compute loss
                classification_loss = classification_loss_func(label, class_label)
                # print(f"classification_loss:{classification_loss}, cur_loss:{cur_loss.mean()}")
                total_loss = 0.1*cur_loss.mean() + 1*classification_loss

                total_loss.backward()
                optimizer.step()

        epoch_loss = total_loss / train_steps
        classifier.eval()
        correct = 0
        total = 0
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        with torch.no_grad():
            for images, label_tokens, text_tokens, class_label in test_dataloader:
                images, label_tokens, text_tokens, class_label = images.to(device), label_tokens.to(
                    device), text_tokens.to(
                    device), class_label.to(device)

                label = classifier(images, train=False)

                _, predicted = torch.max(label.data, 1)

                total += class_label.size(0)
                correct += (predicted == class_label).sum().item()

                # 逐类别统计
                for i in range(class_label.size(0)):
                    true_label = class_label[i].item()
                    pred_label = predicted[i].item()
                    class_total[true_label] += 1
                    if pred_label == true_label:
                        class_correct[true_label] += 1

            # 总体准确率
            print(f"total:{val_num}, correct:{correct}")
            accuracy = 100 * correct / val_num
            print(
                f"Epoch [{epoch + 1}/{args.epochs}], total_loss:{total_loss}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

            # 各类别准确率输出
            print("Per-class accuracy:")
            num_classes = max(class_total.keys()) + 1  # 假设类别标签是从0开始的
            for class_id in range(num_classes):
                total_c = class_total[class_id]
                correct_c = class_correct[class_id]
                acc = 100 * correct_c / total_c if total_c > 0 else 0.0
                print(f"Class {class_id}: {acc:.2f}% ({correct_c}/{total_c})")

            # 保存最优模型
            if accuracy > best_accuracy:
                checkpoint_path = "./Weight/ERMA-Net/ERMA-Net.pth"
                save_dir = os.path.dirname(checkpoint_path)

                # 若文件夹不存在则新建
                os.makedirs(save_dir, exist_ok=True)

                # 保存模型
                torch.save(classifier.state_dict(), checkpoint_path)
                print(f"Checkpoint saved at epoch {epoch}.")

                best_accuracy = accuracy

# Main function
def main():
    args = parse_args()

    # Initialize model components
    net, optimizer, scheduler, classification_loss_func, classifier, device, preprocess = initialize_model(args)

    # Start training
    train_model(args, net, optimizer, scheduler, classification_loss_func, classifier, device, preprocess)


if __name__ == "__main__":
    main()
