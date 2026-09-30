import sys 
import math

import torch
import torch.utils.data
from copy import deepcopy
# from torch import multiprocessing
from torchvision.transforms import v2
# import torchvision.transforms as trans

from torchmetrics.detection import MeanAveragePrecision

#need this for updated teacher
from rpn_updated import RegionProposalNetwork

from wrapper_FRCNN import *
from utils import *
from coco_utils import *

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
path_type = 'coco'

use_nu = False
use_rsud = False

if use_nu:
    train_data_dir = "../dataset/nu_data/data/"
    path_type = 'rsud'
elif use_rsud:
    train_data_dir = "../dataset/rsud/rsud20k/images/"
    path_type = 'nuscenes'

# coco_dataset = get_coco(root=train_data_dir,
#                           image_set='train',
#                           transforms = transforms,
#                           path_type=path_type
#                           )

val_coco_dataset = get_coco(root=train_data_dir,
                          image_set='val',
                          transforms = transforms,
                          path_type=path_type
                          )

#If we are checking jpeg qualities test this to the quality else set to None
quality = None
# coco_dataset.quality = None
val_coco_dataset.quality = quality

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

# Batch size
train_batch_size = 2
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

channels = 6
training_stage = '2' # 1, 2
entropy_model = True
testing = True

device = torch.device('cuda:0') 

detector_head = FRCNN_wrapper_HEAD(bottleneck_channel=channels, is_training=(training_stage=='1'), entropy_split=entropy_model).to(device)#torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="COCO_V1").to(device)#
detector_tail = FRCNN_wrapper_TAIL(bottleneck_channel=channels, entropy_split=entropy_model).to(device)

if training_stage == '1' and (use_nu or use_rsud):
    base_model = FRCNN_wrapper().to('cpu')
    if use_nu:
        base_model.load_state_dict(torch.load('models/teachers/nu_teacher.pth'),strict=False)
    elif use_rsud:
        base_model.load_state_dict(torch.load('models/teachers/rsud_teacher.pth'),strict=False)

    detector_head.backbone_head.conv1 = base_model.backbone_body.conv1

    detector_tail.backbone_tail.layer2 = base_model.backbone_body.layer2
    detector_tail.backbone_tail.layer3 = base_model.backbone_body.layer3
    detector_tail.backbone_tail.layer4 = base_model.backbone_body.layer4

    detector_tail.fpn = base_model.backbone_fpn

    detector_tail.rpn = RegionProposalNetwork(base_model.rpn) 
    detector_tail.roi_heads = base_model.roi_heads
    
    del base_model

    detector_head.to(device)
    detector_tail.to(device)

if training_stage == '2' and entropy_model:
    if use_nu:
        detector_head.load_state_dict(torch.load('models/nu_models/FRCNN_head-S2-'+str(channels)+'CH-L0500.pth'))
        detector_tail.load_state_dict(torch.load('models/nu_models/FRCNN_tail-S2-'+str(channels)+'CH-L0500.pth'), strict=False)
    elif use_rsud:
        detector_head.load_state_dict(torch.load('models/rsud_models/FRCNN_head-S2-'+str(channels)+'CH-L0500.pth'))
        detector_tail.load_state_dict(torch.load('models/rsud_models/FRCNN_tail-S2-'+str(channels)+'CH-L0500.pth'), strict=False)
    else:
        detector_head.backbone_head.encoder.entropy_bb._offset = torch.zeros([6], dtype=torch.int32)
        detector_head.backbone_head.encoder.entropy_bb._quantized_cdf = torch.zeros([6,134], dtype=torch.int32)
        detector_head.backbone_head.encoder.entropy_bb._cdf_length = torch.zeros(6, dtype=torch.int32)

        detector_tail.backbone_tail.decoder.entropy_bb_for_decode._offset = torch.zeros([6], dtype=torch.int32)
        detector_tail.backbone_tail.decoder.entropy_bb_for_decode._quantized_cdf = torch.zeros([6,134], dtype=torch.int32)
        detector_tail.backbone_tail.decoder.entropy_bb_for_decode._cdf_length = torch.zeros([6], dtype=torch.int32)

        detector_head.load_state_dict(torch.load('models/entrop/FRCNN_head-S2-'+str(channels)+'CH-L0500.pth'))
        detector_tail.load_state_dict(torch.load('models/entrop/FRCNN_tail-S2-'+str(channels)+'CH-L0500.pth'), strict=False)
    detector_tail.backbone_tail.decoder.entropy_bb_for_decode = deepcopy(detector_head.backbone_head.encoder.entropy_bb)
    detector_tail.backbone_tail.decoder.entropy_bb_for_decode.requires_grad = False
elif training_stage == '2':
    detector_head.load_state_dict(torch.load('models/standard/FRCNN_head-S2-'+str(channels)+'CH.pth'))
    detector_tail.load_state_dict(torch.load('models/standard/FRCNN_tail-S2-'+str(channels)+'CH.pth'))

def training_loop(model_head, model_tail, learning_rate, train_dataloader, n_epochs, val_loader, batch_size):

    #BEGIN EVAL SECTION
    model_head.eval()
    model_tail.eval()
    if entropy_model and training_stage == '2':
        model_head.backbone_head.encoder.is_testing = True
        model_tail.backbone_tail.decoder.is_testing = True  
        model_head.backbone_head.encoder.entropy_bb.update()
        model_tail.backbone_tail.decoder.entropy_bb_for_decode.update()

    metric = MeanAveragePrecision(iou_type="bbox")
    # iou_metric = IntersectionOverUnion()

    with torch.no_grad():
        avg_size = 0
        i = 0
        print('=== Validation epoch: ', 0, '===')
        sys.stdout.flush()
        for data in val_loader:#tqdm(val_loader):#

            # print(data)

            for d in data[1]:
                if 'image_id' in d:
                    del d['image_id']

            images = list(image.to(device) for image in data[0])#list(image.to for image in data[0])#
            targets = [{k: v.to(device) for k, v in t.items()} if t else {k: v.to(device) for k, v in {"boxes": torch.zeros(0,4).to(device), "labels": torch.zeros(1).type(torch.int64).to(device)}.items()} for t in data[1]]
            #[{k: v for k, v in t.items()} if t else {k: v for k, v in {"boxes": torch.zeros(0,4), "labels": torch.zeros(1).type(torch.int64)}.items()} for t in data[1]]
    
            # if use_nu:
            #     for k in range(len(targets)):
            #         for i in range(len(targets[k]['labels'])):
            #             targets[k]['labels'][i] += 1

            features_from_head, likelihoods_from_head, images_sizes, images_size, original_image_sizes, _ = model_head(images)
            print(features_from_head)
            # print(len(features_from_head[0]))
            features_from_head[0]= features_from_head[0][:34] + features_from_head[0][-(len(features_from_head[0])-35):]
            features_from_head[0]= features_from_head[0][:34] + features_from_head[0][-(len(features_from_head[0])-35):]
            features_from_head[0]= features_from_head[0][:34] + features_from_head[0][-(len(features_from_head[0])-35):]
            features_from_head[0]= features_from_head[0][:34] + features_from_head[0][-(len(features_from_head[0])-35):]
            features_from_head[0]= features_from_head[0][:34] + features_from_head[0][-(len(features_from_head[0])-35):]
            features_from_head[0]= features_from_head[0][:34] + features_from_head[0][-(len(features_from_head[0])-35):]


            features_from_head[0]= features_from_head[0][:88] + features_from_head[0][-(len(features_from_head[0])-89):]
            features_from_head[0]= features_from_head[0][:88] + features_from_head[0][-(len(features_from_head[0])-89):]
            features_from_head[0]= features_from_head[0][:88] + features_from_head[0][-(len(features_from_head[0])-89):]
            features_from_head[0]= features_from_head[0][:88] + features_from_head[0][-(len(features_from_head[0])-89):]
            features_from_head[0]= features_from_head[0][:88] + features_from_head[0][-(len(features_from_head[0])-89):]
            features_from_head[0]= features_from_head[0][:88] + features_from_head[0][-(len(features_from_head[0])-89):]

            features_from_head[0]= features_from_head[0][:135] + features_from_head[0][-(len(features_from_head[0])-136):]
            features_from_head[0]= features_from_head[0][:135] + features_from_head[0][-(len(features_from_head[0])-136):]
            features_from_head[0]= features_from_head[0][:135] + features_from_head[0][-(len(features_from_head[0])-136):]
            features_from_head[0]= features_from_head[0][:135] + features_from_head[0][-(len(features_from_head[0])-136):]
            features_from_head[0]= features_from_head[0][:135] + features_from_head[0][-(len(features_from_head[0])-136):]
            features_from_head[0]= features_from_head[0][:135] + features_from_head[0][-(len(features_from_head[0])-136):]
            # print(len(features_from_head[0]), '-----')
            quit()
            if entropy_model and training_stage == '2':
                avg_size += sys.getsizeof(features_from_head[0])
            detections = model_tail(features_from_head,  images_sizes, images_size, original_image_sizes, size=likelihoods_from_head)

            # if use_nu:
            #     for k in range(len(detections)):
            #         for i in range(len(detections[k]['labels'])):
            #             if detections[k]['labels'][i] in coco_to_nu_label_map.keys():
            #                 detections[k]['labels'][i] = coco_to_nu_label_map[detections[k]['labels'][i]]

            metric.update(detections, targets)
            i+=1

            # if i%1000 == 0:
            #     print('mAP:', metric.compute()['map'])
            #     print('mAP_50%:', metric.compute()['map_50'])

        if (entropy_model and training_stage == '2') or quality is not None:
            print('average_compressed_size:', avg_size/(len(val_loader)*eval_batch_size))
        print('Total mAP:', metric.compute()['map'], '---',  metric.compute()['map_50'], '\n\n') 
        sys.stdout.flush()

    return loss_list

learning_rate = 2e-2 #2e-2
n_epochs = 30

loss_list = training_loop(detector_head, detector_tail, learning_rate, None, n_epochs, val_loader, train_batch_size)

