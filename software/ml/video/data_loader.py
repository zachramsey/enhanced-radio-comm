
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset

class ImageDataLoader:
    train_data: Dataset
    val_data: Dataset = None
    test_data: Dataset = None
    example_data: Dataset = None

    def __init__(
            self,
            dataset_dir: str,
            dataset: str = 'sun397',
            batch_size: int = 32,
            train_pct: float = 0, 
            val_pct: float = 0, 
            test_pct: float = 0, 
            num_examples: int = 0
        ):

        self.dataset_path = dataset_dir
        self.batch_size = batch_size

        self.train_pct = train_pct
        self.val_pct = val_pct
        self.test_pct = test_pct
        self.num_examples = num_examples

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
        self.train_data = datasets.Places365(root=self.dataset_path, split='train-standard', download=True, transform=self.common_transform)
        val_test_dataset = datasets.Places365(root=self.dataset_path, split='val', download=True, transform=self.common_transform)
        self._split_data(val_test_dataset)

    def _get_sun397(self):
        dataset = datasets.SUN397(root=self.dataset_path, download=True, transform=self.common_transform)
        self._split_data(dataset)

    def _split_data(self, dataset: Dataset):
        n_rem = len(dataset)
        if self.train_pct > 0:
            n_train = int(n_rem * self.train_pct)
            n_rem -= n_train
            self.train_data, dataset = random_split(dataset, [n_train, n_rem])

        if self.num_examples > 0:
            n_rem -= self.num_examples
            self.example_data, dataset = random_split(dataset, [self.num_examples, n_rem])

        if self.val_pct == 0:
            self.test_data = dataset
        elif self.test_pct == 0:
            self.val_data = dataset
        else:
            n_test = int(n_rem * self.test_pct / (self.val_pct + self.test_pct)) - self.num_examples
            self.val_data, self.test_data = random_split(dataset, [n_rem - n_test, n_test])

    @property
    def train_dl(self):
        assert self.train_data is None, "Where's the training data? This should not happen!"
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    @property
    def val_dl(self):
        assert self.val_data is None, "Validation set not available"
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=True)
    
    @property
    def test_dl(self):
        assert self.test_data is None, "Test set not available"
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=True)
    
    @property
    def example_dl(self):
        assert self.example_data is not None, "Example set not available"
        return DataLoader(self.example_data, batch_size=1, shuffle=False, num_workers=4)
    