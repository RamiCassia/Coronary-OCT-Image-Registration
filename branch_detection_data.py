import os
import torch
import numpy as np
from PIL import Image
from typing import Tuple, List
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

# For reproducability
torch.manual_seed(12)
np.random.seed(12)


# Class to load OCT imaging data as PyTorch dataset
class OCTDataset(Dataset):
    def __init__(
        self,
        img_path: str,
        csv_path: str,
        device: torch.device,
        test: bool,
        transform=None,
    ) -> None:
        super().__init__()

        self.test = test
        self.device = device
        self.transform = transform
        self.labels = self.load_labels(csv_path=csv_path)
        self.img_path = img_path

        # Split into train, validation and test set idxs
        _, _, file_idxs = next(os.walk(img_path))
        #idxs = np.random.permutation(file_idxs)
        idxs = np.array(file_idxs)
        val_test_idxs = idxs[idxs.shape[0] // 6 * 5 :]
        self.train_idxs = idxs[: idxs.shape[0] // 6 * 5]
        self.val_idxs = val_test_idxs[: val_test_idxs.shape[0] // 2]
        self.test_idxs = val_test_idxs[val_test_idxs.shape[0] // 2 :]

    def __len__(self) -> int:
        return len(self.labels.keys())

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.load_file(img_path=self.img_path, index=index)
        img = img.to(self.device)
        if not self.test and self.transform is not None:
            if index[-7:-4].isdigit():
                idx = int(index[-7:-4]) - 1
                labels = np.array(
                    self.labels[index[:-8] + " sidebranch" + str(idx) + ".nii"]
                )
                labels = torch.from_numpy(labels.astype(np.longlong))
                return (
                    self.transform(img),
                    labels,
                )
            elif index[-6:-4].isdigit():
                idx = int(index[-6:-4]) - 1
                labels = np.array(
                    self.labels[index[:-7] + " sidebranch" + str(idx) + ".nii"]
                )
                labels = torch.from_numpy(labels.astype(np.longlong))
                return (
                    self.transform(img),
                    labels,
                )
            else:
                idx = int(index[-5:-4]) - 1
                labels = np.array(
                    self.labels[index[:-6] + " sidebranch" + str(idx) + ".nii"]
                )
                labels = torch.from_numpy(labels.astype(np.longlong))
                return (
                    self.transform(img),
                    labels,
                )
        else:
            if index[-7:-4].isdigit():
                idx = int(index[-7:-4]) - 1
                labels = np.array(
                    self.labels[index[:-8] + " sidebranch" + str(idx) + ".nii"]
                )
                labels = torch.from_numpy(labels.astype(np.longlong))
                return (
                    img,
                    labels,
                )
            elif index[-6:-4].isdigit():
                idx = int(index[-6:-4]) - 1
                labels = np.array(
                    self.labels[index[:-7] + " sidebranch" + str(idx) + ".nii"]
                )
                labels = torch.from_numpy(labels.astype(np.longlong))
                return (
                    img,
                    labels,
                )
            else:
                idx = int(index[-5:-4]) - 1
                labels = np.array(
                    self.labels[index[:-6] + " sidebranch" + str(idx) + ".nii"]
                )
                labels = torch.from_numpy(labels.astype(np.longlong))
                return (
                    img,
                    labels,
                )
    # Load single image file
    def load_file(self, img_path: str, index: str) -> torch.Tensor:
        img = os.path.join(img_path, index)
        if not os.path.isfile(img):
            raise IOError("Please enter a valid path.")
        img_data = np.array(Image.open(img), dtype=np.float32)
        return torch.from_numpy(img_data).permute(2, 0, 1)

    # Load all labels into dictionary
    def load_labels(self, csv_path: str) -> torch.Tensor:
        if not os.path.isfile(csv_path):
            raise IOError("Please enter a valid path.")
        else:
            dic = dict()

            with open(csv_path) as csv_file:
                for i, line in enumerate(csv_file.readlines()):
                    if i != 0:
                        arr = line.split(",")
                        key = arr[0][:-4] + arr[1] + arr[0][-4:]
                        value = arr[2].replace("\n", "")
                        assert value != ""
                        dic[key] = value

        return dic

    # Load all files (currently not used)
    def load_all_files(self, img_path: str, csv_path: str) -> torch.Tensor:
        if not os.path.isdir(img_path) or not os.path.isfile(csv_path):
            raise IOError("Please enter a valid path.")
        else:
            _, _, files = next(os.walk(img_path))
            file_count = len(files)
            np_img_data = np.ndarray((file_count, 1024, 1024, 3))
            labels = np.ndarray((file_count,))
            dic = dict()

            with open(csv_path) as csv_file:
                for i, line in enumerate(csv_file.readlines()):
                    if i != 0:
                        arr = line.split(",")
                        key = arr[0][:-4] + arr[1] + arr[0][-4:]
                        value = arr[2].replace("\n", "")
                        dic[key] = value

            for i, f in enumerate(os.listdir(path=img_path)):
                file = os.path.join(img_path, f)
                np_img_data[i] = np.array(Image.open(file))
                labels[i] = dic[f]

            return (
                torch.from_numpy(np_img_data).permute(0, 3, 1, 2),
                torch.from_numpy(labels),
            )

    # Get list of indices
    def get_idxs(self, path: str) -> List[str]:
        idx_list = []
        if not os.path.isdir(path):
            raise IOError("Please specify valid path.")
        else:
            for f in os.listdir(path=path):
                if not os.path.isdir(f):
                    idx_list.append(f)

        return idx_list


# Class handling data loading of OCTDataset
class OCTDataloader:
    def __init__(
        self,
        split: str,
        batch_size: int,
        img_path: str,
        csv_path: str,
        device: torch.device,
        transform=None,
    ) -> None:
        assert split == "train" or split == "val" or split == "test"
        if split == "train":
            self.dataset = OCTDataset(
                img_path=img_path,
                csv_path=csv_path,
                device=device,
                transform=transform,
                test=False,
            )
            train_sampler = SubsetRandomSampler(indices=self.dataset.train_idxs)
            self.data_loader = DataLoader(
                dataset=self.dataset, batch_size=batch_size, sampler=train_sampler
            )
        elif split == "val":
            self.dataset = OCTDataset(
                img_path=img_path,
                csv_path=csv_path,
                device=device,
                transform=transform,
                test=True,
            )
            val_sampler = SubsetRandomSampler(indices=self.dataset.val_idxs)
            self.data_loader = DataLoader(
                dataset=self.dataset, batch_size=batch_size, sampler=val_sampler
            )
        else:
            self.dataset = OCTDataset(
                img_path=img_path,
                csv_path=csv_path,
                device=device,
                transform=transform,
                test=True,
            )
            test_sampler = SubsetRandomSampler(indices=self.dataset.test_idxs)
            self.data_loader = DataLoader(
                dataset=self.dataset, batch_size=batch_size, sampler=test_sampler
            )

