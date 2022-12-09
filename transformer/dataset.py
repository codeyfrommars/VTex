# Thanks to https://github.com/jungomi/math-formula-recognition for extracting Crohme Dataset!

import csv
import os
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

START = "<SOS>"
END = "<EOS>"
PAD = "<PAD>"
START_IDX = 0
END_IDX = 1
PAD_IDX = 2

TRG_VOCAB_SIZE = 113
MAX_IMG_SIZE = 500
MAX_TRG_LEN = 50

# File paths
gt_test = "./transformer/data2/groundtruth_2019.txt"
tokensfile = "./transformer/tokens.txt"
root = "./transformer/data2/2019/"
checkpoint_path = "./checkpoints_bttr_data500"
gt_train = "./transformer/data2/gt_split/train.txt"
gt_validation = "./transformer/data2/gt_split/validation.txt" # Train on 2014 dataset
root_train = "./transformer/data2/train/"
train_checkpoint_path = "./checkpoints"

transformers = transforms.Compose(
    [
        transforms.ToTensor(), # normalize to [0,1]
    ]
)


def load_vocab(tokens_file):
    with open(tokens_file, "r") as fd:
        tokens = [START, END, PAD]
        for line in fd.readlines():
            w = line.strip()
            tokens.append(w)
        token_to_id = {tok: i for i, tok in enumerate(tokens)}
        id_to_token = {i: tok for i, tok in enumerate(tokens)}
        return token_to_id, id_to_token


def collate_batch(data):
    batch_size = len(data)
    max_len = max([len(d["truth"]["encoded"]) for d in data])
    max_h = max([d["image"].size(1)for d in data])
    max_w = max([d["image"].size(2)for d in data])
    padded_encoded = [
        d["truth"]["encoded"] + (max_len - len(d["truth"]["encoded"])) * [PAD_IDX]
        for d in data
    ]
    # Resize images to max size
    images = torch.zeros(batch_size, 1, max_h, max_w)
    for idx, d in enumerate(data):
        images[idx, :, : d["image"].size(1), : d["image"].size(2)] = d["image"]


    return {
        "path": [d["path"] for d in data],
        "image": images,
        "truth": {
            "text": [d["truth"]["text"] for d in data],
            "encoded": torch.tensor(padded_encoded),
        },
    }


class CrohmeDataset(Dataset):
    """Dataset CROHME's handwritten mathematical formulas"""

    def __init__(
        self,
        groundtruth,
        tokens_file,
        root=None,
        ext=".bmp",
        crop=False,
        transform=None
    ):
        """
        Args:
            groundtruth (string): Path to ground truth TSV file
            tokens_file (string): Path to tokens text file
            root (string): Path of the root directory of the dataset
            ext (string): Extension of the input files
            crop (bool, optional): Crop images to their bounding boxes [Default: False]
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(CrohmeDataset, self).__init__()
        if root is None:
            root = os.path.dirname(groundtruth)
        self.crop = crop
        self.transform = transform
        self.token_to_id, self.id_to_token = load_vocab(tokens_file)
        self.data = []
        with open(groundtruth, "r") as fd:
            for line in fd.readlines():
                tmp = line.strip().split()
                img_name = tmp[0]
                formula = tmp[1:]
                path = os.path.join(root, img_name + ext)
                image = Image.open(path)
                if max(image.size) > MAX_IMG_SIZE:
                    continue
                if len(formula) > MAX_TRG_LEN:
                    continue
                self.data.append(
                    {
                        "path": path,
                        "truth": {
                            "text": ' '.join(formula),
                            "encoded": [
                                self.token_to_id[START],
                                *[self.token_to_id[x] for x in formula],
                                self.token_to_id[END],
                            ],
                        },
                    }
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        image = Image.open(item["path"]) # Image is a bitmap

        if self.crop:
            # Image needs to be inverted because the bounding box cuts off black pixels,
            # not white ones.
            bounding_box = ImageOps.invert(image).getbbox()
            image = image.crop(bounding_box)

        if self.transform:
            image = self.transform(image)

        return {"path": item["path"], "truth": item["truth"], "image": image}