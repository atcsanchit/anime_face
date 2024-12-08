import sys
import os
import torchvision.transforms as T
import torch
import torch.nn as nn
from torchvision.utils import save_image
import torch.nn.functional as F
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.components.batch_creation import BatchCreation

@dataclass
class GanTrainerConfig:
    learning_rate = 0.002
    epochs = 50
    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

class GanTrainer:
    def __init__(self):
        self.gan_trainer = GanTrainerConfig()
        self.batch_creation_obj = BatchCreation()
        self.train_dl = self.batch_creation_obj.initiate_batch_creation()

    def denormalise(self, img_tensors):
        try:
            return img_tensors * self.gan_trainer.stats[1][0] + self.gan_trainer.stats[0][0]

        except Exception as e:
            logging.info("Error in denormalise")
            raise CustomException(e,sys)
    
    def get_default_device(self):
        try:
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")

        except Exception as e:
            logging.info("Error in get_default_device")
            raise CustomException(e,sys)
        
