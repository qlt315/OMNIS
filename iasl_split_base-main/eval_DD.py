import sys 

import torch
import torch.utils.data
from copy import deepcopy
# from torch import multiprocessing
from torchvision.transforms import v2
# import torchvision.transforms as trans

from torchmetrics.detection import MeanAveragePrecision

from wrapper_FRCNN import *
from utils import *
from coco_utils import *

transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

# Batch size
train_batch_size = 2
eval_batch_size = 1
accum = 4 
clip_value = 5

val_data_dir = '../dataset/coco/'#val2017'
val_coco = '../dataset/coco/annotations/instances_val2017.json'

val_coco_dataset = get_coco(root=val_data_dir,
                          image_set='val',
                          transforms = transforms,
                          )

val_loader = torch.utils.data.DataLoader(val_coco_dataset,
                                          batch_size=eval_batch_size,
                                          collate_fn = collate_fn,
                                          shuffle=False,
                                          num_workers=2)

channels = 12
training_stage = '1' # 1, 2
entropy_model = True
testing = True
train_teacher = False
test_teacher = True

device = torch.device('cuda:0') 

detector_head = FRCNN_wrapper_HEAD(bottleneck_channel=channels, is_training=False, entropy_split=entropy_model).to(device)#torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="COCO_V1").to(device)#
detector_tail = FRCNN_wrapper_TAIL(bottleneck_channel=channels, entropy_split=entropy_model).to(device)

detector_head.load_state_dict(torch.load('models/entrop/FRCNN_head-S1-'+str(channels)+'CH-L0500.pth'), strict=False)
detector_tail.load_state_dict(torch.load('models/entrop/FRCNN_tail-S1-'+str(channels)+'CH-L0500.pth'), strict=False)
detector_tail.backbone_tail.decoder.entropy_bb_for_decode = deepcopy(detector_head.backbone_head.encoder.entropy_bb)

detector_head.backbone_head.encoder.entropy_bb.update()
detector_tail.backbone_tail.decoder.entropy_bb_for_decode.update()

detector_head.eval()
detector_tail.eval()

detector_head.backbone_head.encoder.is_testing = True
detector_tail.backbone_tail.decoder.is_testing = True

metric = MeanAveragePrecision(iou_type="bbox")

with torch.no_grad():
    avg_size = 0
    i = 0

    for data in val_loader:
        for d in data[1]:
            if 'image_id' in d:
                del d['image_id']

        images = list(image.to(device) for image in data[0])#list(image.to for image in data[0])#
        targets = [{k: v.to(device) for k, v in t.items()} if t else {k: v.to(device) for k, v in {"boxes": torch.zeros(0,4).to(device), "labels": torch.zeros(1).type(torch.int64).to(device)}.items()} for t in data[1]]

        features_from_head, likelihoods_from_head, images_sizes, images_size, original_image_sizes, _ = detector_head(images)
        avg_size += sys.getsizeof(features_from_head[0])
        detections = detector_tail(features_from_head,  images_sizes, images_size, original_image_sizes, size=likelihoods_from_head)

        metric.update(detections, targets)

        i+=1
        if i % 100 == 0:
            print(metric.compute()['map'])
            print(avg_size/i)

    print('Total mAP:', metric.compute()['map'], '---',  metric.compute()['map_50'], '\n\n') 
    print(avg_size/len(val_loader))