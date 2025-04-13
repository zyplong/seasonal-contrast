import os
from argparse import ArgumentParser
import warnings
warnings.simplefilter('ignore', UserWarning) # 禁用 UserWarning 类型的警告，使进程输出更简洁

# ======= PyTorch Lightning 相关组件 =======
from pytorch_lightning import Trainer  # 培训模型的主控组件
from pytorch_lightning.loggers import TensorBoardLogger  # 用于日志记录
from pytorch_lightning.callbacks import ModelCheckpoint  # 模型保存回调
from pl_bolts.models.self_supervised.moco.callbacks import MocoLRScheduler  # MoCo自相关学习的LR调度器

# ======= 数据集加载与模型组件 =======
from datasets.seco_datamodule import SeasonalContrastBasicDataModule, SeasonalContrastTemporalDataModule, SeasonalContrastMultiAugDataModule
from models.moco2_module import MocoV2  # 核心 MoCo V2 结构，用于学习推导特征
#from models.ssl_online import SSLOnlineEvaluator  # 在学习过程中进行下游分类验证（被关闭）



def get_experiment_name(hparams):#拼接实验名称，用于 tensorboard 日志目录命名。
    data_name = os.path.basename(hparams.data_dir)#提取出数据集路径的最后一级目录名称
    name = f'{hparams.base_encoder}-{data_name}-{hparams.data_mode}-epochs={hparams.max_epochs}'
    return name


if __name__ == '__main__':
    # ========== 分析命令行参数 ==========
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)  # 添加培训相关参数
    parser = MocoV2.add_model_specific_args(parser)  # 添加 MoCoV2 模型特定参数
    parser = ArgumentParser(parents=[parser], conflict_handler='resolve', add_help=False)  # 处理反复参数
    #测试能不能跑
    # ========== 通用设置 ==========
    '''parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--data_dir', type=str,default="F:\\zyp\\Thesis source code\\seasonal-contrast\\seco_100k\\seasonal_contrast_100k")#直接修改地址
    #parser.add_argument('--data_dir', type=str)#原来代码
    parser.add_argument('--data_mode', type=str, choices=['moco', 'moco_tp', 'seco'], default='seco')
    #parser.add_argument('--max_epochs', type=int, default=200) 重复
    parser.add_argument('--schedule', type=int, nargs='*', default=[120, 160]) # LR 调度节点
    parser.add_argument('--online_data_dir', type=str)# 下游分类数据集地址
    parser.add_argument('--online_max_epochs', type=int, default=25)
    parser.add_argument('--online_val_every_n_epoch', type=int, default=25)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    # print("DEBUG: data_dir =", args.data_dir)
    # print("DEBUG: data_mode =", args.data_mode)
    # print("DEBUG: batch_size =", args.batch_size)
    # print("DEBUG: num_workers =", args.num_workers)'''

    # ============最小训练测试============================
    parser.add_argument('--gpus', type=int, default=1)  # 如果用CPU测试可设为0，用GPU的话保持为1
    parser.add_argument('--data_dir', type=str,
                        default="F:\\zyp\\Thesis source code\\seasonal-contrast\\seco_100k\\seasonal_contrast_100k")

    parser.add_argument('--data_mode', type=str, choices=['moco', 'moco_tp', 'seco'], default='seco')

    parser.add_argument('--max_epochs', type=int, default=1)  # 最小1个epoch
    parser.add_argument('--batch_size', type=int, default=8)  # 最小批次大小
    parser.add_argument('--num_workers', type=int, default=0)  # 避免Windows多线程问题

    parser.add_argument('--schedule', type=int, nargs='*', default=[1])  # 无调度影响

    parser.add_argument('--online_data_dir', type=str, default=None)  # 不用下游分类，设为None
    parser.add_argument('--online_max_epochs', type=int, default=0)  # 禁用在线评估
    parser.add_argument('--online_val_every_n_epoch', type=int, default=0)

    parser.add_argument('--debug', action='store_true')  # 运行时加上 --debug 表示开启调试模式
    args = parser.parse_args()
    # ========== 根据数据类型，创建 datamodule ==========
    if args.data_mode == 'moco':
        datamodule = SeasonalContrastBasicDataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    elif args.data_mode == 'moco_tp':
        datamodule = SeasonalContrastTemporalDataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    elif args.data_mode == 'seco': # 最重要！这里启用 SeCo 封装的数据增强
        datamodule = SeasonalContrastMultiAugDataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    else:
        raise ValueError()

        # ========== 创建 MoCo 模型对象 ==========
    model = MocoV2(**vars(args), emb_spaces=datamodule.num_keys)  # 传入所有参数

    # ========== 日志 + 回调 ==========
    if args.debug:
        logger = False
        checkpoint_callback = False
    else:
        logger = TensorBoardLogger(
            save_dir=os.path.join(os.getcwd(), 'logs', 'pretrain'),  # 日志保存路径
            name=get_experiment_name(args)
        )
        checkpoint_callback = ModelCheckpoint(filename='{epoch}')

# ========== LR 调度器 + 自评价 ==========
scheduler = MocoLRScheduler(initial_lr=args.learning_rate, schedule=args.schedule, max_epochs=args.max_epochs)

    # online_evaluator = SSLOnlineEvaluator(
    #     data_dir=args.online_data_dir,
    #     z_dim=model.mlp_dim,
    #     max_epochs=args.online_max_epochs,
    #     check_val_every_n_epoch=args.online_val_every_n_epoch
    # )
    #关闭在线评估模块
    
# ========== 培训器 ==========
trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        #callbacks=[scheduler, online_evaluator],原本
        callbacks=[scheduler],
        max_epochs=args.max_epochs,
        weights_summary='full'
    )
trainer.fit(model, datamodule=datamodule)



