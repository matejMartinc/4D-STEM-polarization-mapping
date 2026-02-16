import lightning as pl
from typing import Optional
import torch
import tifffile
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import pandas as pd
from collections import Counter
import cv2


def ensure_float32(image, **kwargs):
    return image.astype(np.float32)


def smart_blur(image, **kwargs):
    """Blur that preserves center-of-mass shifts"""
    image = image.astype(np.float32)
    sigma = np.random.uniform(2, 5)  # Much milder!
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)

    return blurred.astype(np.float32)

global_coms = {
    'data/KNN_New_Simulations/KNN_Right_Mid_tiff': (127.98361635368127, 128.32980878432326),
    'data/KNN_New_Simulations/KNN_Right_Up_tiff': (128.21378217578123, 128.1064649411597),
    'data/KNN_New_Simulations/KNN_Mid_Up_tiff': (128.32916216927336, 127.98298797281409),
    'data/KNN_New_Simulations/KNN_Left_Up_tiff': (128.2051107565511, 127.77572077732808),
    'data/KNN_New_Simulations/KNN_Left_Mid_tiff': (127.99353984684956, 127.56101923160612),
    'data/KNN_New_Simulations/KNN_Left_Down_tiff': (127.81147857774144, 127.76667104557288),
    'data/KNN_New_Simulations/KNN_Mid_Down_tiff': (127.5606598384147, 127.99382638449558),
    'data/KNN_New_Simulations/KNN_Right_Down_tiff': (127.81103193407083, 128.08999197025423),
    'data/KNN_Experimental/KNNsc_40MX_minus20V_a': (117.54096079850935, 124.90014511407156),
    'data/KNN_Experimental/KNNsc_40MX_0V_a': (120.83684409295618, 127.36708076884995),
    'data/KNN_Experimental/KNNsc_40MX_20V_a': (119.17878349115057, 126.8409871931911),
    'data/KNN_Experimental/KNNsc_40MX_0V_c': (120.61415874144447, 127.52322138691976),
    'data/KNN_Experimental/KNN_Scr_12b_x100016383': (127.08426548356918, 121.81153175000344),
    'data/KNN_Experimental/KNN_Janina_a': (126.98689420542041, 123.2817309777044),
    'data/KNN_Experimental/KNNsc_tripod_tiff': (124.89649400447742, 123.22913202951223),
    'data/KNN_Experimental/KNNsc_40MX_0V_b': (120.83684409295552, 127.36708076885009),
    'data/KNN_Experimental/Simulated_anomaly': (127.86537540588256, 128.08908346465432),
    'data/LU_4DSTEM/LU_4DSTEM_20nm_NoTDS': (128.21647437916096, 127.84473797666828),
    'data/LU_4DSTEM/LU_4DSTEM_50nm_NoTD': (127.77318551811736, 127.77334435480056),
    'data/KNN_Right_Mid_20nm_24mrad16383': (127.98361635368224, 128.32980878432403),
    'data/RD_KNN_Simpl_50nm_24mrad_NoTDS16383': (127.93535140232265, 128.10867304914018),
    'data/RU_KNN_Simpl_20nm_24mrad_NoTDS16383': (127.89713423085495, 127.89677841284399),
    'data/LD_KNN_Simpl_20nm_24mrad_NoTDS': (127.85917481704536, 127.88918756786924),
    'data/KNN_Simpl_Right_Down_20nm': (127.81103193407107, 128.0899919702541),
    'data/with_anomalies': (128.05006912971808, 127.88601020577771),
    'data/without_anomalies': (127.99944841130528, 128.01623108636488),
    'data/KNN_Janina_a_45_degROT': (124.16774347883354, 124.88362588112079),
    'data/KNN_Janina_a_90_degROT': (123.28173097770474, 128.01310579457981),
    'data/KNN_Janina_a_180_degROT': (128.01310579458004, 131.71826902229552)
}


def crop_around_corrected_com(
    image: np.ndarray,
    crop_size: int,
    average_com: tuple[float, float],
) -> np.ndarray:
    h, w = image.shape
    avg_com_y, avg_com_x = average_com


    local_com_y, local_com_x = h / 2, w / 2
    image_center_y = h / 2
    image_center_x = w / 2

    shift_y = avg_com_y - image_center_y
    shift_x = avg_com_x - image_center_x

    # The corrected CoM reflects the structure's position without the global shift.
    corrected_com_y = local_com_y - shift_y
    corrected_com_x = local_com_x - shift_x

    # 4. Determine the top-left corner of the crop box
    start_h = int(round(corrected_com_y - crop_size / 2))
    start_w = int(round(corrected_com_x - crop_size / 2))

    # 5. Handle edge cases to ensure the crop box stays within image bounds.
    # Clamp the starting coordinates to be at least 0.
    start_h = max(0, start_h)
    start_w = max(0, start_w)

    # If the crop box extends beyond the image, shift it back.
    if start_h + crop_size > h:
        start_h = h - crop_size
    if start_w + crop_size > w:
        start_w = w - crop_size

    # Re-clamp in case the image is smaller than the crop size.
    start_h = max(0, start_h)
    start_w = max(0, start_w)

    # 6. Perform the crop and return the result
    return image[start_h:start_h + crop_size, start_w:start_w + crop_size]


class KNNDataset(torch.utils.data.Dataset):
    def __init__ (self, df, transforms=None, backbone=False):
        super(KNNDataset, self).__init__()
        self.df = df
        self.transforms = transforms
        self.backbone = backbone

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path, label = self.df.iloc[index].image_path,  self.df.iloc[index].label
        image = tifffile.imread(image_path).astype(np.float32)
        if image.ndim > 2:
            image = image[0]

        label = np.array(label)
        crop_size = 64
        idx = image_path.rfind('/')

        f_path = image_path[:idx]
        image = crop_around_corrected_com(image, crop_size=crop_size, average_com=global_coms[f_path])
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

        if self.transforms:
            image = self.transforms(image = image)['image'] #/ 255.0
            label = torch.from_numpy(label)#.float()

        return image_path, image, label



class DataLoaderGenerator:
    def __init__(self, dataframes, image_size, batch_size, backbone):
        super().__init__()
        self.dataframes = dataframes
        self.image_size = image_size
        self.batch_size = batch_size
        self.backbone = backbone

        self.info_message("\n")
        [self.info_message("The length of the {} dataframe is: {}", split, len(df)) for split, df in self.dataframes.items()]
        self.info_message("\n")

        self.augmentations = self.transform()

        self.transforms = {
            'train': self.augmentations['train'],
            'val': self.augmentations['val'],
            'test': self.augmentations['test'],
        }

        self.shuffle = {
            'train': True,
            'val': False,
            'test': False
        }
        self.drop_last = {
            'train': True,
            'val': False,
            'test': False
        }


    def transform(self):
        data_transforms = {
            "val": A.Compose([
                ToTensorV2()], p=1.),
            #Comment out "train" transforms if you do not want to augment training images
            "train": A.Compose([
                A.Lambda(image=ensure_float32, p=1.0),
                A.Lambda(image=smart_blur, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
                A.GaussNoise(per_channel=False, p=0.2),
                ToTensorV2()
            ], p=1.0),

            "test": A.Compose([
                ToTensorV2()], p=1.),
        }

        return data_transforms

    def get_data_loader(self, data_type):
        if data_type not in self.dataframes:
            raise ValueError(f"Invalid data type: {data_type}. Supported types are {', '.join(self.dataframes.keys())}")

        transform = self.transforms[data_type]

        dataset = KNNDataset(
            df=self.dataframes[data_type],
            transforms=transform,
            backbone=self.backbone
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle[data_type],
            drop_last=self.drop_last[data_type],
            num_workers=24,
            pin_memory=True,
        )
        self.info_message("Number of batches in {} dataloader: {}", data_type, len(dataloader))
        return dataloader

    def get_data_loaders(self):
        dataloaders = {
            data_type: self.get_data_loader(data_type) for data_type in self.dataframes.keys()
        }
        return dataloaders

    @staticmethod
    def info_message(message, *args, end="\n"):
        print(message.format(*args), end=end)


class KNNDataModule(pl.LightningDataModule):
    def __init__(self, dataframes, image_size, batch_size, backbone):
        super().__init__()
        self.dataframes = dataframes
        self.image_size = image_size
        self.batch_size = batch_size
        self.backbone = backbone

    def setup(self, stage: Optional[str] = None):
        self.dataloaders = DataLoaderGenerator(self.dataframes, self.image_size, self.batch_size, self.backbone).get_data_loaders()

    def train_dataloader(self):
        return self.dataloaders['train']

    def val_dataloader(self):
        return self.dataloaders['val']

    # Include predict_dataloader if your DataLoaderGenerator also provides for a prediction set
    def predict_dataloader(self):
        return self.dataloaders['test']

    # Include predict_dataloader if your DataLoaderGenerator also provides for a prediction set
    def test_dataloader(self):
        return self.dataloaders['test']


def create_dataframe(path, training_files=None):

    available_classes = {
        'data/KNN_New_Simulations/KNN_Right_Mid_tiff': 0,
        'data/KNN_New_Simulations/KNN_Right_Up_tiff': 1,
        'data/KNN_New_Simulations/KNN_Mid_Up_tiff': 2,
        'data/KNN_New_Simulations/KNN_Left_Up_tiff': 3,
        'data/KNN_New_Simulations/KNN_Left_Mid_tiff': 4,
        'data/KNN_New_Simulations/KNN_Left_Down_tiff': 5,
        'data/KNN_New_Simulations/KNN_Mid_Down_tiff': 6,
        'data/KNN_New_Simulations/KNN_Right_Down_tiff': 7,
        'data/KNN_Experimental/KNNsc_40MX_0V_a': 6,
        'data/KNN_Experimental/KNNsc_40MX_0V_b': 6,
        'data/KNN_Experimental/KNNsc_40MX_0V_c': 6,
        'data/KNN_Experimental/KNN_Janina_a': 0,
        'data/KNN_Experimental/KNNsc_tripod_tiff': 2,
        'data/KNN_Experimental/Simulated_anomaly': 6,
        'data/LU_4DSTEM/LU_4DSTEM_20nm_NoTDS': 3,
        'data/LU_4DSTEM/LU_4DSTEM_50nm_NoTD': 3,
        'data/KNN_Right_Mid_20nm_24mrad16383': 0,
        'data/RD_KNN_Simpl_50nm_24mrad_NoTDS16383': 7,
        'data/RU_KNN_Simpl_20nm_24mrad_NoTDS16383': 1,
        'data/LD_KNN_Simpl_20nm_24mrad_NoTDS': 5,
        'data/KNN_Simpl_Right_Down_20nm': 7,
    }

    data = []
    all_labels = []
    folder_paths = []
    if isinstance(path, str):
        dir_names = os.listdir(path)
        for folder in dir_names:
            folder_paths.append(os.path.join(path, folder))
    else:
        folder_paths = path
    # Iterate over each folder

    for folder_path in folder_paths:
        if training_files is not None:
            training_f = training_files[folder_path]
        for filename in os.listdir(folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):  # Check file extension
                file_path = os.path.join(folder_path, filename)
                if training_files is not None:
                    if file_path in training_f:
                        if isinstance(training_f, dict):
                            label = training_f[file_path]
                        else:
                            label = available_classes[folder_path] # Get class id from folder name
                        if label is not None:  # Only add data if folder name is known
                            data.append((file_path, label))
                            all_labels.append(label)
                else:
                    labels = available_classes[folder_path]  # Get class id from folder name
                    if labels is not None:  # Only add data if folder name is known
                        data.append((file_path, labels))
                        all_labels.append(labels)


    # Convert the list of tuples into a DataFrame
    df = pd.DataFrame(data, columns=['image_path', 'label'])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print('Label distrib', Counter(all_labels))
    return df, len(available_classes), available_classes
