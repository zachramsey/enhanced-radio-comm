
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset

class ImageDataLoader:
    train_dataset: Dataset
    val_dataset: Dataset = None
    test_dataset: Dataset = None
    example_dataset: Dataset = None

    def __init__(
            self,
            dataset_dir: str,
            dataset: str = 'places365',
            batch_size: int = 32,
            train_pct: float = None, 
            val_pct: float = None, 
            test_pct: float = None, 
            num_examples: int = None
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

        os.makedirs(dataset_dir, exist_ok=True)
        if dataset == 'places365':
            self._get_places365()
        else:
            raise ValueError(f"Dataset {dataset} not supported")

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        if self.val_dataset is not None:
            self.val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
        if self.test_dataset is not None:
            self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
        if self.example_dataset is not None:
            self.example_dataloader = DataLoader(self.example_dataset, batch_size=1, shuffle=False, num_workers=4)

    def _get_places365(self):
        self.train_dataset = datasets.Places365(root=self.dataset_path, split='train-standard', download=True, transform=self.common_transform)
        val_test_dataset = datasets.Places365(root=self.dataset_path, split='val', download=True, transform=self.common_transform)
        
        if self.num_examples is not None:
            self.example_dataset, val_test_dataset = random_split(val_test_dataset, [self.num_examples, len(val_test_dataset) - self.num_examples])
        
        if self.val_pct is not None:
            self.val_dataset, self.test_dataset = random_split(val_test_dataset, (self.val_pct, 1-self.val_pct))
            self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=True)
            self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=True)
        elif self.test_pct is not None:
            self.val_dataset, self.test_dataset = random_split(val_test_dataset, (1-self.test_pct, self.test_pct))
            self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=True)
            self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=True)
        else:
            self.val_dataset = val_test_dataset
            self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=True)
            self.test_dataloader = None

    @property
    def train(self):
        assert self.train_dataloader is not None, "Where's the training data? This should not happen!"
        return self.train_dataloader
    
    @property
    def val(self):
        assert self.val_dataloader is not None, "Validation set not available"
        return self.val_dataloader
    
    @property
    def test(self):
        assert self.test_dataloader is not None, "Test set not available"
        return self.test_dataloader
    
    @property
    def example(self):
        assert self.example_dataloader is not None, "Example set not available"
        return self.example_dataloader