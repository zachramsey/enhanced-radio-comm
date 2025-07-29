import os
import pickle
import PIL.Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split

import torch
import numpy as np

class ImageDataLoader:
    train_data: DataLoader
    val_data: DataLoader
    test_data: DataLoader
    example_data: DataLoader

    def __init__(
            self,
            dataset_dir: str,
            dataset: str = 'sun397',
            batch_size: int = 32,
            train_pct: float = 0, 
            val_pct: float = 0, 
            test_pct: float = 0, 
            num_examples: int = 0,
            device: str = ''
        ):

        self.dataset_dir = dataset_dir
        self.batch_size = batch_size

        self.train_pct = train_pct
        self.val_pct = val_pct
        self.test_pct = test_pct
        self.num_examples = num_examples

        self.device = device

        self.common_transform = transforms.Compose([
            transforms.RandomCrop((480, 640), pad_if_needed=True),
            transforms.ToTensor()
        ])

        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            
        if dataset == 'places365':
            self._get_places365()
        elif dataset == 'sun397':
            self._get_sun397()
        else:
            raise ValueError(f"Dataset {dataset} not supported")

    def _get_places365(self):
        print(">>> Loading Places365 dataset")
        train_data = datasets.Places365(root=self.dataset_dir, split='train-standard', download=True, transform=self.common_transform)
        # val_test_dataset = datasets.Places365(root=self.dataset_dir, split='val', download=True, transform=self.common_transform)
        new_image_files, new_labels = self._filter_size(train_data, "places365", train_data.imgs, train_data.targets, min_size=(640, 480))
        train_data.imgs = new_image_files
        train_data.targets = new_labels

        print(">>> Splitting dataset")
        self._split_data(train_data)

    def _get_sun397(self):
        print(">>> Loading SUN397 dataset")
        dataset = datasets.SUN397(root=self.dataset_dir, download=True, transform=self.common_transform)
        new_image_files, new_labels = self._filter_size(dataset, "sun397", dataset._image_files, dataset._labels, min_size=(640, 480))
        dataset._image_files = new_image_files
        dataset._labels = new_labels

        print(">>> Splitting dataset")
        self._split_data(dataset)

    def _filter_size(self, dataset: Dataset, name: str, image_files: list, labels: list, min_size: tuple = (640, 480)):
        if os.path.exists(f"{self.dataset_dir}/{name}_image_files.pkl") and os.path.exists(f"{self.dataset_dir}/{name}_labels.pkl"):
            print(">>> Loading existing image files and labels")
            with open(f"{self.dataset_dir}/{name}_image_files.pkl", "rb") as f:
                new_image_files = pickle.load(f)
            with open(f"{self.dataset_dir}/{name}_labels.pkl", "rb") as f:
                new_labels = pickle.load(f)
        else:
            new_image_files = []
            new_labels = []
            len_dataset = len(dataset)
            img_res = f"{min_size[0]}x{min_size[1]}"
            for i in range(len_dataset):
                if i % 250 == 0: print(f"Removing images smaller than {img_res}: {i}/{len_dataset}", end="\r")
                image_file = image_files[i]
                image = PIL.Image.open(image_file[0] if isinstance(image_file, tuple) else image_file).convert("RGB")
                if image.size[0] >= min_size[0] and image.size[1] >= min_size[1]:
                    new_image_files.append(image_file)
                    new_labels.append(labels[i])
            print(f"\nRemoved {len_dataset - len(new_image_files)} images smaller than {img_res}")

            print(">>> Saving image files and labels")
            with open(f"{self.dataset_dir}/{name}_image_files.pkl", "wb") as f:
                pickle.dump(new_image_files, f)
            with open(f"{self.dataset_dir}/{name}_labels.pkl", "wb") as f:
                pickle.dump(new_labels, f)

        return new_image_files, new_labels

    def _split_data(self, dataset: Dataset):
        n_rem = len(dataset)
        if self.train_pct > 0:
            n_train = int(n_rem * self.train_pct)
            n_rem -= n_train
            train_data, dataset = random_split(dataset, [n_train, n_rem])
            self.train_data = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)

        if self.num_examples > 0:
            n_rem -= self.num_examples
            example_data, dataset = random_split(dataset, [self.num_examples, n_rem])
            self.example_data = DataLoader(example_data, batch_size=1, shuffle=False, num_workers=0)

        if self.val_pct == 0:
            self.test_data = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        elif self.test_pct == 0:
            self.val_data = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        else:
            n_test = int(n_rem * self.test_pct / (self.val_pct + self.test_pct)) - self.num_examples
            val_data, test_data = random_split(dataset, [n_rem - n_test, n_test])
            self.val_data = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)
            self.test_data = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    @property
    def train_dl(self):
        assert self.train_data is not None, "Where's the training data? This should not happen!"
        return self.train_data
    
    @property
    def val_dl(self):
        assert self.val_data is not None, "Validation set not available"
        return self.val_data
    
    @property
    def test_dl(self):
        assert self.test_data is not None, "Test set not available"
        return self.test_data
    
    @property
    def example_dl(self):
        assert self.example_data is not None, "Example set not available"
        return self.example_data
    