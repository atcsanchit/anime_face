import sys
import os
import zipfile
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    data_path = os.path.join("artifacts","data_ingestion","archive.zip")
    unzip_path = os.path.join("artifacts","data_ingestion","anime_data")

class DataIngestion:
    def __init__(self):
        self.data_ingestion = DataIngestionConfig()

    def unzip_folder(self):
        try:
            if not os.path.exists(self.data_ingestion.data_path):
                raise FileNotFoundError(f"The file '{self.data_ingestion.data_path}' does not exist.")
            
            if not zipfile.is_zipfile(self.data_ingestion.data_path):
                raise ValueError(f"The file '{self.data_ingestion.data_path}' is not a valid zip file.")
            
            os.makedirs(self.data_ingestion.unzip_path, exist_ok=True)
            
            with zipfile.ZipFile(self.data_ingestion.data_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_ingestion.unzip_path)
                print(f"Extracted '{self.data_ingestion.data_path}' to '{self.data_ingestion.unzip_path}'")

        except Exception as e:
            logging.info("Error in unzip_folder")
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    data_ingestion_obj = DataIngestion()
    data_ingestion_obj.unzip_folder()