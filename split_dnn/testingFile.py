import sys 
import math

import torch
import torch.utils.data
from copy import deepcopy
# from torch import multiprocessing
from torchvision.transforms import v2
# import torchvision.transforms as trans

from torchmetrics.detection import MeanAveragePrecision

from wrapper_FRCNN import FRCNN_wrapper, FRCNN_wrapper_HEAD, FRCNN_wrapper_TAIL
from wrapper_MaskRCNN import *
from utils import *
from coco_utils import *
from torchvision.ops import box_iou

device = "cpu"

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

teacher = FRCNN_wrapper().to(device).eval()

detector_head = FRCNN_wrapper_HEAD(bottleneck_channel=12, is_training=False, entropy_split=False, new_standard=False).to(device)#torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="COCO_V1").to(device)#
detector_tail = FRCNN_wrapper_TAIL(bottleneck_channel=12, entropy_split=False).to(device)

detector_head.load_state_dict(torch.load('models/standard/FRCNN_head-S1-12CH.pth'), strict=False)
detector_tail.load_state_dict(torch.load('models/standard/FRCNN_tail-S1-12CH.pth'))

detector_head.eval()
detector_tail.eval()

bbox_loss_fn = nn.SmoothL1Loss()  # Loss for bounding box regression
classification_loss_fn = nn.KLDivLoss(reduction='batchmean')  # KL Divergence for class logits

def match_boxes(teacher_boxes, student_boxes, iou_threshold=0.5):
    # Calculate the IoU matrix between teacher and student boxes
    iou_matrix = box_iou(teacher_boxes, student_boxes)
    
    # Get the best matches using IoU threshold
    matched_pairs = []
    teacher_indices, student_indices = torch.where(iou_matrix > iou_threshold)
    
    # Iterate through matches to add to the list
    for t_idx, s_idx in zip(teacher_indices, student_indices):
        matched_pairs.append((t_idx.item(), s_idx.item()))

    return matched_pairs

def compute_distillation_loss(teacher_outputs, student_outputs, epsilon=1e-6):
    distillation_loss = 0.0

    for teacher_output, student_output in zip(teacher_outputs, student_outputs):
        teacher_boxes = teacher_output['boxes']
        student_boxes = student_output['boxes']

        print(teacher_boxes)
        print(student_boxes)

        if len(teacher_boxes) == 0 or len(student_boxes) == 0:
            continue  # Skip if no boxes detected by either model

        # Match teacher and student boxes using IoU
        matched_pairs = match_boxes(teacher_boxes, student_boxes, iou_threshold=0.5)
        
        print(matched_pairs)

        for t_idx, s_idx in matched_pairs:
            # Bounding Box Loss: Match student bbox outputs to teacher
            teacher_box = teacher_boxes[t_idx]
            student_box = student_boxes[s_idx]
            distillation_loss += bbox_loss_fn(student_box, teacher_box.detach())
            print(bbox_loss_fn(student_box, teacher_box.detach()))

            # # Classification Loss: Minimize divergence between teacher and student class logits
            # teacher_scores = teacher_output['scores'][t_idx]
            # student_scores = student_output['scores'][s_idx]

            # #  # Add epsilon to prevent log(0) issues
            # # teacher_scores = teacher_scores + epsilon
            # # student_scores = student_scores + epsilon

            # # # Normalize scores to ensure they are proper probabilities
            # # teacher_probs = teacher_scores / teacher_scores.sum()
            # # student_probs = student_scores / student_scores.sum()

            # # # Convert to log probabilities for KLDivLoss
            # # teacher_log_probs = torch.log(teacher_probs)
            # # student_log_probs = torch.log(student_probs)

            # # # Compute KL divergence loss
            # # distillation_loss += classification_loss_fn(student_log_probs, teacher_log_probs.detach())
            # print(classification_loss_fn(student_log_probs, teacher_log_probs.detach()))
            # print('------------------')

    return distillation_loss

for data in val_loader: #tqdm(train_dataloader): #
    for d in data[1]:
        if 'image_id' in d:
            del d['image_id']

    images = list(image.to(device) for image in data[0])#list(image for image in data[0])#
    # print(data[1])
    targets = [{k: v.to(device) for k, v in t.items()} if t else {k: v.to(device) for k, v in {"boxes": torch.zeros(0,4).to(device), "labels": torch.zeros(1).type(torch.int64).to(device)}.items()} for t in data[1]]
    #[{k: v for k, v in t.items()} if t else {k: v for k, v in {"boxes": torch.zeros(0,4), "labels": torch.zeros(1).type(torch.int64)}.items()} for t in data[1]]

    # print(targets[0])

    # print('=================================================================================')

    t_out = teacher(images)
    # print(t_out)

    # print('=================================================================================')

    features_from_head, likelihoods_from_head, images_sizes, images_size, original_image_sizes, _ = detector_head(images)         
    d_out = detector_tail(features_from_head,  images_sizes, images_size, original_image_sizes, size=likelihoods_from_head)

    # print(d_out)

    distillation_loss = compute_distillation_loss(t_out, d_out)
    print(distillation_loss)

    # bb_loss = nn.SmoothL1Loss()
    # c_loss = nn.KLDivLoss(reduction='batchmean')

    # l1 = bb_loss(d_out[0]['boxes'], t_out[0]['boxes'].detach())
    # print(l1)

    # teacher_logits = torch.log(t_out[0]['scores'] + 1e-10)
    # student_logits = torch.log(d_out[0]['scores'] + 1e-10)
    # l2 = c_loss(student_logits, teacher_logits.detach())
    # print(l2)

    break