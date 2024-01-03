import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, Callable
import torch
from utils import expanded_join
import torch.nn.functional as F
import PIL

class CropSegmentationDataset(Dataset):
    ROOT_PATH: str = "/content/gdrive/MyDrive/AI/project-dataset"
    id2cls: dict = {0: "background",
                    1: "crop",
                    2: "weed",
                    3: "partial-crop",
                    4: "partial-weed"}
    cls2id: dict = {"background": 0,
                    "crop": 1,
                    "weed": 2,
                    "partial-crop": 3,
                    "partial-weed": 4}

    def __init__(self, set_type: str = "train", transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 merge_small_items: bool = True,
                 remove_small_items: bool = False):
        """Class to load datasets for the Project.

        Remark: `target_transform` is applied before merging items (this eases data augmentation).

        :param set_type: Define if you load training, validation or testing sets. Should be either "train", "val" or "test".
        :param transform: Callable to be applied on inputs.
        :param target_transform: Callable to be applied on labels.
        :param merge_small_items: Boolean to either merge classes of small or occluded objects.
        :param remove_small_items: Boolean to consider as background class small or occluded objects. If `merge_small_items` is set to `True`, then this parameter is ignored.
        """
        super(CropSegmentationDataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.merge_small_items = merge_small_items
        self.remove_small_items = remove_small_items

        if set_type not in ["train", "val", "test"]:
            raise ValueError("'set_type has an unknown value. "
                             f"Got '{set_type}' but expected something in ['train', 'val', 'test'].")

        self.set_type = set_type
        images = glob(expanded_join(self.ROOT_PATH, set_type, "images/*"))
        images.sort()
        self.images = np.array(images)

        labels = glob(expanded_join(self.ROOT_PATH, set_type, "labels/*"))
        labels.sort()
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        try:
            input_img = Image.open(self.images[index], "r")
            target = Image.open(self.labels[index], "r")

            if self.transform is not None:
                input_img = self.transform(input_img)

            if self.target_transform is not None:
                target = self.target_transform(target)
            # Convert PIL Images to NumPy arrays for modification
            target_np = np.array(target, dtype=np.uint8)  # Ensure the data type is uint8
            if self.merge_small_items:
                target_np[target_np == self.cls2id["partial-crop"]] = self.cls2id["crop"]
                target_np[target_np == self.cls2id["partial-weed"]] = self.cls2id["weed"]
            elif self.remove_small_items:
                target_np[target_np == self.cls2id["partial-crop"]] = self.cls2id["background"]
                target_np[target_np == self.cls2id["partial-weed"]] = self.cls2id["background"]
            # Convert NumPy array back to PIL Image
            torch_tensor = torch.from_numpy(target_np).squeeze(0)
            # Get unique elements and their counts
            target_indices = torch.tensor([cls for cls in torch_tensor.flatten().numpy()], dtype=torch.long)
            target_indices = target_indices.view(torch_tensor.shape)
            target_one_hot = F.one_hot(target_indices, num_classes=3).permute(2, 0, 1).float()


            return input_img, target_one_hot
        except PIL.UnidentifiedImageError as e:
            print(f"Error opening image file at index {index}: {e}")
            # You can choose to return a default image or skip this sample
            return self.__getitem__((index + 1) % len(self))

        except IndexError:
            return self.__getitem__((index + 1) % len(self))


    def get_class_number(self):
        if self.merge_small_items or self.remove_small_items:
            return 3
        else:
            return 5
