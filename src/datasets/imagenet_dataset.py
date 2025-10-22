from typing import Optional

import torchvision.transforms.functional as F
from datasets import load_dataset

from src.datasets.base_dataset import BaseDataset


DATASET_URLS = {
    "train-21k": "timm/imagenet-w21-webp-wds",

    "train-1k": "timm/imagenet-1k-wds",
    "val-1k": "timm/imagenet-1k-wds",

    "train-200": "zh-plus/tiny-imagenet",
    "val-200": "zh-plus/tiny-imagenet",
}


class ImageNetDataset(BaseDataset):
    def __init__(self,
                 part: str, 
                 limit: Optional[int] = None,
                 instance_transforms=None):
        assert part in DATASET_URLS.keys(), (
            f"Part '{part}' is not recognized. "
            f"Available parts: {list(DATASET_URLS.keys())}"
        )

        self.instance_transforms = instance_transforms

        self.limit = limit or {
            "train-21k": 13151276,
            "train-1k": 1281167,
            "train-200": 100000,
            "val-1k": 50000,
            "val-200": 10000,
        }[part]
        self.num_classes = {
            "train-21k": 19167,
            "train-1k": 1000,
            "train-200": 200,
            "val-1k": 1000,
            "val-200": 200,
        }[part]

        self.index = load_dataset(
            DATASET_URLS[part],
            split="train" if "train" in part else "validation",
            streaming=True, # to avoid downloading the entire dataset of 3TB \(-_-)/
        )


        self.current_index = -1
        self.iterator = iter(self.index)

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        if ind == 0:
            self.current_index = -1
            self.iterator = iter(self.index)

        self.current_index += 1

        data_dict = next(self.iterator)
        image_path = data_dict.get("__url__", None)
        
        if "image" in data_dict:
            image = data_dict["image"]
        elif "webp" in data_dict:
            image = data_dict["webp"]
        elif "jpg" in data_dict:
            image = data_dict["jpg"]
        else:
            raise ValueError("No image found in the dataset item.")
        
        if "label" in data_dict:
            target = data_dict["label"]
        elif "cls" in data_dict:
            target = data_dict["cls"]
        else:
            raise ValueError("No label found in the dataset item.")

        instance_data = {
            "image": F.pil_to_tensor(image.convert("RGB")),
            "target": target,

            "image_path": image_path,
        }

        instance_data = self.preprocess_data(instance_data)

        return instance_data

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return self.limit

    def preprocess_data(self, instance_data):
        """
        Preprocess data with instance transforms.

        Each tensor in a dict undergoes its own transform defined by the key.

        Args:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element) (possibly transformed via
                instance transform).
        """
        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
                instance_data[transform_name] = self.instance_transforms[
                    transform_name
                ](instance_data[transform_name])
        return instance_data
