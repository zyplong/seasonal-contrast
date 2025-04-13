from pathlib import Path
from copy import deepcopy
from argparse import ArgumentParser

import torch
from torch import nn, optim
from torchvision.models import resnet
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.loggers import TensorBoardLogger

from datasets.eurosat_datamodule import EurosatDataModule
from models.moco2_module import MocoV2


class Classifier(LightningModule):
#接受一个 backbone（就是预训练的 ResNet18）； 定义下游任务分类模型：encoder + linear 分类头
    def __init__(self, backbone, in_features, num_classes):
        super().__init__()
        self.encoder = backbone # 特征提取网络（比如 ResNet18）
        self.classifier = nn.Linear(in_features, num_classes) # 线性分类器
        self.criterion = nn.CrossEntropyLoss() # 分类损失
        self.accuracy = Accuracy()  # 准确率评估指标

#前向传播
    def forward(self, x): # ❗ 默认冻结 encoder，不更新其参数
        with torch.no_grad():
            feats = self.encoder(x)# 提取图像特征
        logits = self.classifier(feats) # 将特征输入到线性层得到分类结果
        return logits

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        # 记录训练损失与准确率到 TensorBoard 和进度条
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        # 同样记录验证指标
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/acc', acc, prog_bar=True)
        return loss

    def shared_step(self, batch):
        x, y = batch# 输入图像与标签
        logits = self(x) # 前向传播
        loss = self.criterion(logits, y)  # 计算交叉熵损失
        acc = self.accuracy(torch.argmax(logits, dim=1), y) # 计算准确率
        return loss, acc

    def configure_optimizers(self): # 只训练分类器的参数，encoder 是 frozen 的
        optimizer = optim.Adam(self.classifier.parameters())
        # 设置分阶段学习率下降策略（在 60%、80% 的 epoch 降低 LR）
        max_epochs = self.trainer.max_epochs
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.6*max_epochs), int(0.8*max_epochs)])
        return [optimizer], [scheduler]

# ✅ 主函数入口
if __name__ == '__main__':
    pl.seed_everything(42)  # 设置随机种子，保证可复现性

    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--data_dir', type=str)# EuroSAT 数据目录
    parser.add_argument('--backbone_type', type=str, default='imagenet') # backbone 选择：random / imagenet / pretrain
    parser.add_argument('--ckpt_path', type=str, default=None)  # SeCo 预训练 ckpt 路径
    args = parser.parse_args()

    args.gpus = 1
    args.data_dir = "F:\zyp\Thesis source code\seasonal-contrast\EuroSAT\2750"
    args.backbone_type = "pretrain"
    args.ckpt_path ="F:\zyp\Thesis source code\seasonal-contrast\seco_resnet18_100k.ckpt"

    # 构造数据模块，会自动下载 EuroSAT 并分为 train/val
    datamodule = EurosatDataModule(args.data_dir)

    # ✅ 根据指定的 backbone 类型加载 encoder
    if args.backbone_type == 'random':  # 不加载任何预训练，直接随机初始化 ResNet18
        backbone = resnet.resnet18(pretrained=False)
        backbone = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
    elif args.backbone_type == 'imagenet':  # 加载 torchvision 自带的 ImageNet 预训练 ResNet18
        backbone = resnet.resnet18(pretrained=True)
        backbone = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
    elif args.backbone_type == 'pretrain': # 从 SeCo 预训练的 MoCo ckpt 中加载 encoder_q
        model = MocoV2.load_from_checkpoint(args.ckpt_path)
        backbone = deepcopy(model.encoder_q)
    else:
        raise ValueError()

    # ✅ 构建完整模型：encoder + 线性分类头
    model = Classifier(backbone, in_features=512, num_classes=datamodule.num_classes)
    # 提供一个输入样例，方便 Lightning 打印模型结构图（可选）
    model.example_input_array = torch.zeros((1, 3, 64, 64))
    # ✅ 设置 TensorBoard 日志目录
    experiment_name = args.backbone_type
    logger = TensorBoardLogger(save_dir=str(Path.cwd() / 'logs' / 'eurosat'), name=experiment_name)
    # ✅ 配置训练器 Trainer
    trainer = Trainer(gpus=args.gpus, logger=logger, checkpoint_callback=False, max_epochs=100, weights_summary='full')  # checkpoint_callback=False：不自动保存 checkpoint # 显示网络结构摘要
    trainer.fit(model, datamodule=datamodule)
