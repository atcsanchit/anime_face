import sys
import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException


@dataclass
class BatchCreationConfig:
    data_dir = os.path.join("artifacts","data_ingestion","anime_data")
    image_size = 64
    batch_size = 128
    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    shuffle = True
    num_workers = 3
    pin_memory = True

class BatchCreation:
    def __init__(self):
        self.batch_creation = BatchCreationConfig()

    def initiate_batch_creation(self):
        try:
            train_ds = ImageFolder(self.batch_creation.data_dir, transform=T.Compose([
                T.Resize(self.batch_creation.image_size),
                T.CenterCrop(self.batch_creation.image_size),
                T.ToTensor(),
                T.Normalize(*self.batch_creation.stats)
            ]))

            train_dl = DataLoader(
                train_ds, 
                self.batch_creation.batch_size, 
                shuffle=self.batch_creation.shuffle, 
                num_workers=self.batch_creation.num_workers,
                pin_memory=self.batch_creation.pin_memory
            )
            logging.info("Successfully initiated batch_creation")
            return train_dl

        except Exception as e:
            logging.info("Error in initiate_batch_creation")
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    batch_creation_obj = BatchCreation()
    batch_creation_obj.initiate_batch_creation()