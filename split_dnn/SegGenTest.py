import sys 
import math

import torch
import torch.utils.data
from copy import deepcopy
# from torch import multiprocessing
from torchvision.transforms import v2
# import torchvision.transforms as trans

from torchmetrics.detection import MeanAveragePrecision

from wrapper_MaskRCNN import *
from utils import *
from coco_utils import *

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamPredictor(sam)

transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

# transforms = trans.Compose(
#     [
#         trans.ToTensor()
#     ]
# )

# path to your own data and coco file
train_data_dir = '../dataset/coco/'#train2017'
train_coco = '../dataset/coco/annotations/instances_train2017.json'
train_coco_captions = '../../dataset/coco/annotations/captions_train2017.json'

val_data_dir = '../dataset/coco/'#val2017'
val_coco = '../dataset/coco/annotations/instances_val2017.json'

use_nu = False
use_rsud = True

if use_nu:
    train_data_dir = "../dataset/nu_data/data/"
    path_type = 'rsud'
elif use_rsud:
    train_data_dir = "../dataset/rsud/rsud20k/images/"

# coco_dataset = get_coco(root=train_data_dir,
#                           image_set='train',
#                           transforms = transforms
#                           )

val_coco_dataset = get_coco(root=val_data_dir,
                          image_set='val',
                          transforms = transforms
                          )

#If we are checking jpeg qualities test this to the quality else set to None
quality = None
# coco_dataset.quality = None
val_coco_dataset.quality = quality

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

# Batch size
train_batch_size = 1
eval_batch_size = 1
accum = 4
clip_value = 5

# indices = torch.randperm(len(coco_dataset))[:17000]
# subset_coco = torch.utils.data.Subset(coco_dataset, indices)

# own DataLoader
# data_loader = torch.utils.data.DataLoader(coco_dataset,#subset_coco,#
#                                           batch_size=train_batch_size,
#                                           collate_fn = collate_fn,
#                                           shuffle=True,
#                                           num_workers=2)

val_loader = torch.utils.data.DataLoader(val_coco_dataset,
                                          batch_size=eval_batch_size,
                                          collate_fn = collate_fn,
                                          shuffle=False,
                                          num_workers=2)

for data in val_loader: #tqdm(train_dataloader): #
    for d in data[1]:
        if 'image_id' in d:
            del d['image_id']

    images = list(image.to(device) for image in data[0])#list(image for image in data[0])#
    # print(data[1])
    targets = [{k: v.to(device) for k, v in t.items()} if t else {k: v.to(device) for k, v in {"boxes": torch.zeros(0,4).to(device), "labels": torch.zeros(1).type(torch.int64).to(device)}.items()} for t in data[1]]
    #[{k: v for k, v in t.items()} if t else {k: v for k, v in {"boxes": torch.zeros(0,4), "labels": torch.zeros(1).type(torch.int64)}.items()} for t in data[1]]

    print(targets[0])

    image = cv2.imread('../dataset/rsud/rsud20k/images/val2017/val999.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image.dtype)
    # masks = mask_generator.generate(image)
    mask_generator.set_image(image)

    transformed_boxes = mask_generator.transform.apply_boxes_torch(targets[0]["boxes"], image.shape[:2])
    masks, _, _ = mask_generator.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
)

    print(masks[0])
    print(len(masks))

    print(targets[0]["boxes"].shape)

    break