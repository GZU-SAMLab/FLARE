from d2l import torch as d2l
import pandas as pd
from dataloder import CamVidDataset
from model_DLAR_Net import UnetPlusPlusDecoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import numpy as np
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
    # print(f"pred shape:{pred.shape}, pred:{pred}")

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
DATA_DIR = './data3168/segment'
x_train_dir = os.path.join(DATA_DIR, '/home/xiaridehehe/ownProgram/ReMM/data_set/data3401/segment/Train/JPEGImages')
y_train_dir = os.path.join(DATA_DIR, '/home/xiaridehehe/ownProgram/ReMM/data_set/data3401/segment/Train/LesionSegmentationClass')
z_train_dir = os.path.join(DATA_DIR, '/home/xiaridehehe/ownProgram/ReMM/data_set/data3401/segment/Train/LeafSegmentationClass')
x_valid_dir = os.path.join(DATA_DIR, '/home/xiaridehehe/ownProgram/ReMM/data_set/data3401/segment/Test/JPEGImages')
y_valid_dir = os.path.join(DATA_DIR, '/home/xiaridehehe/ownProgram/ReMM/data_set/data3401/segment/Test/LesionSegmentationClass')
z_valid_dir = os.path.join(DATA_DIR, '/home/xiaridehehe/ownProgram/ReMM/data_set/data3401/segment/Test/LeafSegmentationClass')

# 创建数据集对象
train_dataset = CamVidDataset(x_train_dir, y_train_dir, z_train_dir)
# print(train_dataset)
val_dataset = CamVidDataset(x_valid_dir, y_valid_dir, z_valid_dir)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True, num_workers=16)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, drop_last=True, num_workers=8)
print(f"train dataset:{len(train_dataset)},test dataset:{len(val_dataset)}")

deep_supervision = True
au_branch = True
model = UnetPlusPlusDecoder(num_classes=2, deep_supervision=deep_supervision, Attention=True, au_branch=au_branch).cuda()

# 损失函数选用多分类交叉熵损失函数
lossf = nn.CrossEntropyLoss(ignore_index=255)

# 选用SGD优化器来训练
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, last_epoch=-1)

# 训练50轮
epochs_num = 200

def train_batch_ch13(net, X, y, leaf, loss, trainer, devices, deep_supervision, au_branch):
    """Train for a minibatch with multiple GPUs (defined in Chapter 13).

    Defined in :numref:`sec_image_augmentation`"""
    # if isinstance(X, list):
    #     # Required for BERT fine-tuning (to be covered later)
    #     X = [x.to(devices) for x in X]
    # else:
    #     X = X.to(devices)
    # y = y.to(devices)
    # leaf = leaf.to(devices)
    net.train()
    trainer.zero_grad()
    if not deep_supervision:
        pred = net(X)
        l = loss(pred[0], y) + loss(pred[1], leaf)
        print(f"pred[0] shape:{pred[0].shape}, pred[0]:{pred[0]}")
        l.sum().backward()
        trainer.step()
        acc_sum = (d2l.accuracy(pred[0], y) + d2l.accuracy(pred[1], leaf))/2
        return l.sum(), acc_sum
    else:
        preds = net(X)
        # Initialize loss and accuracy sums
        l_sum = torch.zeros(1).cuda()
        acc_sum = torch.zeros(1).cuda()
        # print(f"preds shape:{preds[3].shape}, preds:{preds[3]}")

        # Calculate loss and accuracy for each segmentation branch with deep supervision
        for i, pred in enumerate(preds):
            if i <= 3:
                if au_branch:
                    l = loss(pred, y) + loss(preds[i + 4], leaf)
                else:
                    l = loss(pred, y)
                l.sum().backward(retain_graph=True)  # Backward with retain_graph for multiple losses
                l_sum += l.sum()
                if au_branch:
                    acc = (d2l.accuracy(pred, y) + d2l.accuracy(preds[i + 4], leaf))/2
                else:
                    acc = d2l.accuracy(pred, y)
                acc_sum += acc
            # print(f"loss{i}:{l.sum()};accuracy{i}:{acc}")

        # Step the optimizer
        trainer.step()

        return l.sum(), acc

def evaluate_accuracy_gpu(net, data_iter, deep_supervision, device=None, au_branch = True):
    """Compute the accuracy for a model on a dataset using a GPU.

    Defined in :numref:`sec_utils`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric_lesion = d2l.Accumulator(2)
    metric_leaf = d2l.Accumulator(2)

    with torch.no_grad():
        for X, y, z in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            z = z.to(device)
            if not deep_supervision:
                lesion, leaf = net(X)
                metric_lesion.add(d2l.accuracy(lesion, y), d2l.size(y))
                metric_leaf.add(d2l.accuracy(leaf, z), d2l.size(z))
            else:
                if au_branch:
                    _, _, _, output4_lesion, _, _, _, output4_leaf = net(X)
                    metric_lesion.add(d2l.accuracy(output4_lesion, y), d2l.size(y))
                    metric_leaf.add(d2l.accuracy(output4_leaf, z), d2l.size(z))
                else:
                    _, _, _, output4_lesion = net(X)
                    metric_lesion.add(d2l.accuracy(output4_lesion, y), d2l.size(y))
                    metric_leaf.add(d2l.accuracy(output4_lesion, y), d2l.size(y))
    return metric_lesion[0] / metric_lesion[1], metric_leaf[0] / metric_leaf[1]

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
    best_miou_leaf = 0  # 用于保存最高mIoU

    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (features, labels, leaf) in enumerate(train_iter):
            timer.start()
            # 将标签转换为二进制（0: 背景, 1: 前景）
            labels[labels == 255] = 1  # 将 255 转换为 1
            leaf[leaf == 255] = 1  # 将 255 转换为 1
            labels = labels.long().cuda()  # 确保标签为长整型
            leaf = leaf.long().cuda()

            l, acc = train_batch_ch13(
                net, features, labels, leaf, loss, trainer, devices, deep_supervision, au_branch)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))

        test_acc, train_acc = evaluate_accuracy_gpu(net, test_iter, deep_supervision, au_branch=au_branch)
        animator.add(epoch + 1, (None, None, test_acc))
        scheduler.step()

        print(f"epoch {epoch+1} --- loss {metric[0] / metric[2]:.6f} --- train acc {metric[1] / metric[3]:.6f} --- test acc {(test_acc+train_acc)/2:.6f} --- cost time {timer.sum()}")

        # ---------保存训练数据---------------
        df = pd.DataFrame()
        loss_list.append(metric[0] / metric[2])
        train_acc_list.append(metric[1] / metric[3])
        test_acc_list.append(test_acc)
        epochs_list.append(epoch)
        time_list.append(timer.sum())

        df['epoch'] = epochs_list
        df['loss'] = loss_list
        df['train_acc'] = train_acc_list
        df['test_acc'] = test_acc_list
        df['time'] = time_list
        df.to_excel("savefile/Unet++_camvid1.xlsx")

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

                if au_branch:
                    # 前向传播
                    if deep_supervision:
                        _, _, _, outputs, _, _, _, _ = model(images)
                    else:
                        outputs, _ = model(images)
                else:
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
        print("lesion:")
        print(f"Precision per class: {precision}")
        print(f"Recall per class: {recall}")
        print(f"F1 Score per class: {f1_score}")
        print(f"mIoU: {miou:.4f}")
        print(f"Dice Coefficient: {mdice:.4f}")


        #----------------保存模型-------------------
        if miou > best_miou:
            torch.save(model.state_dict(), f'/home/xiaridehehe/ownProgram/ReMM/Weight/DLAR_Ablation/DLAR_4301_AuAttnFalse/Unet++_1_{epoch+1}.pth')
            best_miou = miou
            print(f"lesion best IoU:{best_miou}")

        if au_branch:
            # 设置模型为评估模式
            net.eval()

            # 初始化混淆矩阵
            num_classes = 2
            confusion_matrix = init_confusion_matrix(num_classes)

            # 开始评估
            with torch.no_grad():
                for images, labels, leaf in tqdm(val_loader):
                    images = images.cuda()
                    leaf = leaf.cuda()

                    # 前向传播
                    if deep_supervision:
                        _, _, _, _, _, _, _, outputs = model(images)
                    else:
                        _, outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)

                    # 更新混淆矩阵
                    confusion_matrix = update_confusion_matrix(confusion_matrix, predicted, leaf)

            # 计算指标
            precision, recall, f1_score, miou, dice, mdice = compute_metrics(confusion_matrix, num_classes)

            # 输出结果
            print("leaf:")
            print(f"Precision per class: {precision}")
            print(f"Recall per class: {recall}")
            print(f"F1 Score per class: {f1_score}")
            print(f"mIoU: {miou:.4f}")
            print(f"Dice Coefficient: {mdice:.4f}")
            if miou > best_miou_leaf:
                best_miou_leaf = miou
                print(f"leaf best IoU:{best_miou_leaf}")

train_ch13(model, train_loader, val_loader, lossf, optimizer, epochs_num, scheduler)
