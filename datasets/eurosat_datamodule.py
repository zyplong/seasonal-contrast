# from torch.utils.data import DataLoader
# from torchvision import transforms as T
# from pytorch_lightning import LightningDataModule
#
# from datasets.eurosat_dataset import EurosatDataset
#
#
# class EurosatDataModule(LightningDataModule):
#
#     def __init__(self, data_dir):
#         super().__init__()
#         self.data_dir = data_dir
#
#     @property
#     def num_classes(self):
#         return 10
#
#     def setup(self, stage=None):
#         self.train_dataset = EurosatDataset(self.data_dir, split='train', transform=T.ToTensor())
#         self.val_dataset = EurosatDataset(self.data_dir, split='val', transform=T.ToTensor())
#
#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset,
#             batch_size=32,
#             shuffle=True,
#             num_workers=8,
#             drop_last=True,
#             pin_memory=True
#         )
#
#     def val_dataloader(self):
#         return DataLoader(
#             self.val_dataset,
#             batch_size=32,
#             shuffle=False,
#             num_workers=8,
#             drop_last=True,
#             pin_memory=True
#         )


from pytorch_lightning import LightningDataModule
from torchvision.datasets import EuroSAT
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

class EurosatDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size=128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # 保持与模型一致
            transforms.ToTensor()
        ])

    def setup(self, stage=None):
        full_dataset = EuroSAT(root=self.data_dir, download=True, transform=self.transform)
        n_train = int(len(full_dataset) * 0.8)
        n_val = len(full_dataset) - n_train
        self.train_dataset, self.val_dataset = random_split(full_dataset, [n_train, n_val])
        self.num_classes = len(full_dataset.classes)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
