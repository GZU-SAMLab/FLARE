from d2l import torch as d2l
from tqdm import tqdm
import pandas as pd
from dataloder import CamVidDataset
from model_DLAR_Net import UnetPlusPlusOrigin
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import os.path as osp
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import os


def init_confusion_matrix(num_classes):
    return np.zeros((num_classes, num_classes))

# 更新混淆矩阵
def update_confusion_matrix(conf_mat, pred, target):
    pred = pred.flatten().cpu().numpy()
    target = target.flatten().cpu().numpy()

    # 将标签中的 255 值映射为 1
    target = np.where(target > 0, 1, target)
    pred = np.where(pred > 0, 1, pred)

    # 更新混淆矩阵
    for p, t in zip(pred, target):
        conf_mat[p, t] += 1
    return conf_mat


# 从混淆矩阵计算指标
def compute_metrics(conf_mat, num_classes):
    TP = np.diag(conf_mat)
    FP = conf_mat.sum(axis=0) - TP
    FN = conf_mat.sum(axis=1) - TP
    TN = conf_mat.sum() - (TP + FP + FN)

    # 计算 Precision
    precision = TP / (TP + FP + 1e-5)
    # 计算 Recall
    recall = TP / (TP + FN + 1e-5)
    # 计算 F1 Score
    f1_score = 2 * precision * recall / (precision + recall + 1e-5)
    # 计算 mIoU
    iou = TP / (TP + FP + FN + 1e-5)
    miou = np.nanmean(iou)
    # 计算 Dice Coefficient
    dice = 2 * TP / (2 * TP + FP + FN + 1e-5)
    mdice = np.nanmean(dice)

    return precision, recall, f1_score, miou, dice, mdice

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 设置随机种子
torch.manual_seed(17)

# 设置数据集路径
x_train_dir = './dataset/segment/Train/JPEGImages'
z_train_dir = './dataset/segment/Train/LesionSegmentationClass'
y_train_dir = './dataset/segment/Train/LeafSegmentationClass'

x_valid_dir = './dataset/segment/Test/JPEGImages'
z_valid_dir = './dataset/segment/Test/LesionSegmentationClass'
y_valid_dir = './dataset/segment/Test/LeafSegmentationClass'

# 创建数据集对象
train_dataset = CamVidDataset(x_train_dir, y_train_dir, z_train_dir)
# print(train_dataset)
val_dataset = CamVidDataset(x_valid_dir, y_valid_dir, z_valid_dir)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,drop_last=True, num_workers=16)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True,drop_last=True, num_workers=16)

# 打印数据集的长度
print(f'Training dataset size: {len(train_dataset)}')
print(f'Validation dataset size: {len(val_dataset)}')

deep_supervision = False
Attention = False
model = UnetPlusPlusOrigin(num_classes=2, deep_supervision=False).cuda()

# 损失函数选用多分类交叉熵损失函数
lossf = nn.CrossEntropyLoss(ignore_index=255)

# 选用SGD优化器来训练
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, last_epoch=-1)

# 训练50轮
epochs_num = 200

def train_batch_ch13(net, X, y, leaf, loss, trainer, devices, deep_supervision):
    """Train for a minibatch with multiple GPUs (defined in Chapter 13).

    Defined in :numref:`sec_image_augmentation`"""
    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    leaf = leaf.to(devices[0])
    net.train()
    trainer.zero_grad()
    if not deep_supervision:
        pred = net(X)
        l = loss(pred, y)
        l.sum().backward()
        trainer.step()
        acc_sum = d2l.accuracy(pred, y)
        return l.sum(), acc_sum
    else:
        preds = net(X)
        # Initialize loss and accuracy sums
        l_sum = torch.zeros(1).to(devices[0])
        acc_sum = torch.zeros(1).to(devices[0])

        # Calculate loss and accuracy for each segmentation branch with deep supervision
        for i, pred in enumerate(preds):
            l = loss(pred, y)
            l.sum().backward(retain_graph=True)  # Backward with retain_graph for multiple losses
            l_sum += l.sum()
            acc = d2l.accuracy(pred, y)
            acc_sum += acc
            # print(f"loss{i}:{l.sum()};accuracy{i}:{acc}")

        # Step the optimizer
        trainer.step()

        return l.sum(), acc

def evaluate_accuracy_gpu(net, data_iter, deep_supervision, device=None):
    """Compute the accuracy for a model on a dataset using a GPU.

    Defined in :numref:`sec_utils`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)

    with torch.no_grad():
        for X, y, z in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            if not deep_supervision:
                metric.add(d2l.accuracy(net(X), y), d2l.size(y))
            else:
                _, _, _, output4 = net(X)
                metric.add(d2l.accuracy(output4, y), d2l.size(y))
    return metric[0] / metric[1]

def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, scheduler,
               devices=d2l.try_all_gpus()):
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])

    loss_list = []
    train_acc_list = []
    test_acc_list = []
    epochs_list = []
    time_list = []
    best_miou = 0  # 用于保存最高mIoU

    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (features, labels, leaf) in enumerate(train_iter):
            timer.start()
            # 将标签转换为二进制（0: 背景, 1: 前景）
            labels[labels == 255] = 1  # 将 255 转换为 1
            labels = labels.long()  # 确保标签为长整型

            l, acc = train_batch_ch13(
                net, features, labels, leaf, loss, trainer, devices, deep_supervision)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))

        test_acc = evaluate_accuracy_gpu(net, test_iter, deep_supervision)
        animator.add(epoch + 1, (None, None, test_acc))
        scheduler.step()

        print(f"epoch {epoch+1} --- loss {metric[0] / metric[2]:.6f} --- train acc {metric[1] / metric[3]:.6f} --- test acc {test_acc:.6f} --- cost time {timer.sum()}")

        # ---------保存训练数据---------------
        df = pd.DataFrame()
        loss_list.append(metric[0] / metric[2])
        train_acc_list.append(metric[1] / metric[3])
        test_acc_list.append(test_acc)
        epochs_list.append(epoch)
        time_list.append(timer.sum())

        # 设置模型为评估模式
        net.eval()

        # 初始化混淆矩阵
        num_classes = 2
        confusion_matrix = init_confusion_matrix(num_classes)

        # 开始评估
        with torch.no_grad():
            for images, labels, leaf in tqdm(val_loader):
                images = images.cuda()
                labels = labels.cuda()

                # 前向传播
                if deep_supervision:
                    _, _, _, outputs = model(images)
                else:
                    outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                # 更新混淆矩阵
                confusion_matrix = update_confusion_matrix(confusion_matrix, predicted, labels)

        # 计算指标
        precision, recall, f1_score, miou, dice, mdice = compute_metrics(confusion_matrix, num_classes)

        # 输出结果
        print(f"Precision per class: {precision}")
        print(f"Recall per class: {recall}")
        print(f"F1 Score per class: {f1_score}")
        print(f"mIoU: {miou:.4f}")
        print(f"Dice Coefficient: {mdice:.4f}")


        #----------------保存模型-------------------
        if miou > best_miou:
            save_path = './Weight/FLLA-Net_Leaf.pth'
            save_dir = os.path.dirname(save_path)

            # 若文件夹不存在则新建
            os.makedirs(save_dir, exist_ok=True)

            # 保存模型
            torch.save(model.state_dict(), save_path)
            best_miou = miou
            print(f"New best IoU: {best_miou:.4f}, model saved to {save_path}")


train_ch13(model, train_loader, val_loader, lossf, optimizer, epochs_num, scheduler)
