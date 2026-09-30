import sys 
import math
import random

import torch
import torch.utils.data
from copy import deepcopy
# from torch import multiprocessing
from torchvision.transforms import v2
# import torchvision.transforms as trans

from torchmetrics.detection import MeanAveragePrecision

from rpn_updated import RegionProposalNetwork

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
path_type = 'coco'

use_nu = False
use_rsud = False

if use_nu:
    train_data_dir = "../dataset/nu_data/data/"
    path_type = 'rsud'
elif use_rsud:
    train_data_dir = "../dataset/rsud/rsud20k/images/"
    path_type = 'nuscenes'

coco_dataset = get_coco(root=train_data_dir,
                          image_set='train',
                          transforms = transforms,
                          path_type=path_type
                          )

val_coco_dataset = get_coco(root=train_data_dir,
                          image_set='val',
                          transforms = transforms,
                          path_type=path_type
                          )

#If we are checking jpeg qualities test this to the quality else set to None
quality = None
coco_dataset.quality = None
val_coco_dataset.quality = quality

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

# Batch size
train_batch_size = 1 #2
eval_batch_size = 1
accum = 4 #4
clip_value = 5

# indices = torch.randperm(len(coco_dataset))[:17000]
# subset_coco = torch.utils.data.Subset(coco_dataset, indices)

# own DataLoader
data_loader = torch.utils.data.DataLoader(coco_dataset,#subset_coco,#
                                          batch_size=train_batch_size,
                                          collate_fn = collate_fn,
                                          shuffle=True,
                                          num_workers=2)

val_loader = torch.utils.data.DataLoader(val_coco_dataset,
                                          batch_size=eval_batch_size,
                                          collate_fn = collate_fn,
                                          shuffle=False,
                                          num_workers=2)

channels = 12
training_stage = '2' # 1, 2
entropy_model = True
testing = False
test_teacher = False
train_teacher = False
mask_threshold = 0.7

device = torch.device('cuda:0') # torch.device('cpu') #
if training_stage == '1' or train_teacher:
    teacher = MaskRCNN_wrapper().to(device).eval()#torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="COCO_V1").to(device).eval()
    if use_nu and not train_teacher:
        teacher.load_state_dict(torch.load('models/teachers/nu_teacher_seg.pth'),strict=False)
    if use_rsud and not train_teacher:
        teacher.load_state_dict(torch.load('models/teachers/rsud_teacher_seg.pth'),strict=False)

detector_head = MaskRCNN_wrapper_HEAD(bottleneck_channel=channels, is_training=(training_stage=='1'), entropy_split=entropy_model).to(device)#torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="COCO_V1").to(device)#
detector_tail = MaskRCNN_wrapper_TAIL(bottleneck_channel=channels, entropy_split=entropy_model).to(device)

if not train_teacher:
    detector_head = MaskRCNN_wrapper_HEAD(bottleneck_channel=channels, is_training=(training_stage=='1'), entropy_split=entropy_model).to(device)#torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="COCO_V1").to(device)#
    detector_tail = MaskRCNN_wrapper_TAIL(bottleneck_channel=channels, entropy_split=entropy_model).to(device)

    if training_stage == '1' and (use_nu or use_rsud):
        base_model = MaskRCNN_wrapper().to('cpu')
        if use_nu:
            base_model.load_state_dict(torch.load('models/teachers/nu_teacher_seg.pth'),strict=False)
        elif use_rsud:
            base_model.load_state_dict(torch.load('models/teachers/rsud_teacher_seg.pth'),strict=False)

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
            detector_head.load_state_dict(torch.load('models/nu_models/FRCNN_head-S1-'+str(channels)+'CH-L0500.pth'))
            detector_tail.load_state_dict(torch.load('models/nu_models/FRCNN_tail-S1-'+str(channels)+'CH-L0500.pth'), strict=False)
        elif use_rsud:
            detector_head.load_state_dict(torch.load('models/rsud_models/FRCNN_head-S1-'+str(channels)+'CH-L0500.pth'))
            detector_tail.load_state_dict(torch.load('models/rsud_models/FRCNN_tail-S1-'+str(channels)+'CH-L0500.pth'), strict=False)
        else:
            detector_head.backbone_head.encoder.entropy_bb._offset = torch.zeros([12], dtype=torch.int32)
            detector_head.backbone_head.encoder.entropy_bb._quantized_cdf = torch.zeros([12,189], dtype=torch.int32)
            detector_head.backbone_head.encoder.entropy_bb._cdf_length = torch.zeros(12, dtype=torch.int32)

            detector_tail.backbone_tail.decoder.entropy_bb_for_decode._offset = torch.zeros([12], dtype=torch.int32)
            detector_tail.backbone_tail.decoder.entropy_bb_for_decode._quantized_cdf = torch.zeros([12,189], dtype=torch.int32)
            detector_tail.backbone_tail.decoder.entropy_bb_for_decode._cdf_length = torch.zeros([12], dtype=torch.int32)

            detector_head.load_state_dict(torch.load('models/entrop/MaskRCNN_head-S2-'+str(channels)+'CH-L0500.pth'))
            detector_tail.load_state_dict(torch.load('models/entrop/MaskRCNN_tail-S2-'+str(channels)+'CH-L0500.pth'), strict=False)
        
        detector_tail.backbone_tail.decoder.entropy_bb_for_decode = deepcopy(detector_head.backbone_head.encoder.entropy_bb)
        detector_tail.backbone_tail.decoder.entropy_bb_for_decode.requires_grad = False

        detector_head.backbone_head.encoder.entropy_bb.update()
        detector_tail.backbone_tail.decoder.entropy_bb_for_decode.update()
        detector_tail.backbone_tail.decoder.apply_grad = True
    elif training_stage == '2':
        detector_head.load_state_dict(torch.load('models/standard/MaskRCNN_head-S1-'+str(channels)+'CH.pth'))
        detector_tail.load_state_dict(torch.load('models/standard/MaskRCNN_tail-S1-'+str(channels)+'CH.pth'))
else:
    detector_head = None
    detector_tail = None

def training_loop(model_head, model_tail, learning_rate, train_dataloader, n_epochs, val_loader, batch_size):
    
    if train_teacher:
        optimizer = torch.optim.SGD(teacher.parameters(), lr = learning_rate, momentum=0.5, weight_decay=1e-5)
    else:
        if training_stage == '1':
            optimizer = torch.optim.SGD((list(model_head.parameters()) + list(model_tail.backbone_tail.decoder.parameters())), lr=learning_rate, momentum=0.5, weight_decay=1e-5)
        elif training_stage == '2':
            optimizer = torch.optim.SGD((list(model_tail.backbone_tail.bit_processor.parameters()) + list(model_tail.backbone_tail.layer2.parameters()) + 
                               list(model_tail.backbone_tail.layer3.parameters()) + list(model_tail.backbone_tail.layer4.parameters()) + 
                               list(model_tail.fpn.parameters()) + list(model_tail.rpn.parameters()) + 
                               list(model_tail.roi_heads.parameters())), lr=learning_rate, momentum=0.5, weight_decay=1e-5)

    mse_loss = torch.nn.MSELoss()

    lr2 = learning_rate/(n_epochs+3) 

    loss_list = []
    best_loss = 9999999
    
    num_pixels = 100000#5136384

    for epoch in range(n_epochs):#tqdm(range(n_epochs)):#
        if train_teacher:
            teacher.train()
        else:
            if training_stage == '1':
                model_head.train()
                model_tail.backbone_tail.decoder.train()
            elif training_stage == '2':
                model_head.eval()
                model_tail.train()
                model_tail.backbone_tail.decoder.eval()
                if entropy_model:
                    model_head.backbone_head.encoder.is_testing = True
                    model_head.backbone_head.encoder.is_training = False # so we can train over converted bits
                    model_tail.backbone_tail.decoder.is_testing = True
                    model_tail.backbone_tail.decoder.entropy_bb_for_decode.eval()
                
        batch_loss = 0
        total_loss = 0

        for g in optimizer.param_groups:
            print(g['lr'])
            g['lr'] = g['lr'] - lr2

        # map_val = 0
        i = 0
        accum_counter = 1
        printed = False
        if not testing:
            print('=== Training epoch: ', epoch, '===')
            for data in train_dataloader: #tqdm(train_dataloader): #
                if i * accum >= 3000:
                    break

                for d in data[1]:
                    if 'image_id' in d:
                        del d['image_id']

                images = list(image.to(device) for image in data[0])#list(image for image in data[0])#
                # print(data[1])
                targets = [{k: v.to(device) for k, v in t.items()} if t else {k: v.to(device) for k, v in {"boxes": torch.zeros(0,4).to(device), "labels": torch.zeros(1).type(torch.int64).to(device)}.items()} for t in data[1]]
                #[{k: v for k, v in t.items()} if t else {k: v for k, v in {"boxes": torch.zeros(0,4), "labels": torch.zeros(1).type(torch.int64)}.items()} for t in data[1]]

                if use_rsud:
                    image = cv2.imread(data[2][0])
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # masks = mask_generator.generate(image)
                    mask_generator.set_image(image)

                    transformed_boxes = mask_generator.transform.apply_boxes_torch(targets[0]["boxes"], image.shape[:2])
                    masks, _, _ = mask_generator.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes,
                        multimask_output=False,
                    )

                    # print(targets[0]['masks'])
                    # print('\n\n===================================================================================\n\n')
                    masks = masks.squeeze(1)
                    masks = masks.long()
                    # print(masks)

                    targets[0]['masks'] = masks

                if train_teacher:
                    loss, detections = teacher(images, targets)
                    batch_loss += (loss['loss_classifier'] + loss['loss_box_reg'] + loss['loss_mask'] + loss['loss_objectness'] + loss["loss_rpn_box_reg"])
                else:
                    # forward pass
                    if training_stage == '1':
                        detections = teacher(images)

                    features_from_head, likelihoods_from_head, images_sizes, images_size, original_image_sizes, targets = model_head(images, targets)
                    loss_1, detections_1 = model_tail(features_from_head,  images_sizes, images_size, original_image_sizes, targets, size=likelihoods_from_head)
                    mse_target = (model_tail.backbone_tail.decoder_out.clone().detach().cpu().cuda())/10
                    if training_stage == '2':
                        fix_prob = 0.25
                        byte_loss_mean = 0
                        byte_loss_var = 0.33
                        drop_prob = 0.75
                        for f_indx in range(len(features_from_head)):
                            hex_object = features_from_head[f_indx].hex()
                            hex_object = list(hex_object)
                            lost_bytes = 0
                            for s_indx in range(len(hex_object)):
                                we_drop = abs(random.gauss(byte_loss_mean, byte_loss_var)) + fix_prob
                                if we_drop >= drop_prob:
                                    hex_object[s_indx] = '0'
                                    lost_bytes += 1
                                   
                            hex_object = ''.join(hex_object)
                            features_from_head[f_indx] = bytes.fromhex(hex_object)
                            print("packets_lost", lost_bytes)
                    loss, detections = model_tail(features_from_head,  images_sizes, images_size, original_image_sizes, targets, size=likelihoods_from_head)
                    print(loss)
                    # loss caculations
                    if training_stage == '1':
                        if entropy_model:
                            N, C, H, W = features_from_head.size()
                            num_pixels = N * C * H * W

                            
                            bpp_loss = (0.01*(torch.log(likelihoods_from_head).sum() / (-math.log(2) * num_pixels)))
                            aux_loss = model_head.backbone_head.encoder.entropy_bb.loss()
                        else:
                            bpp_loss = 0
                            aux_loss = 0

                        instace_mse_loss = 10*(mse_loss(model_tail.backbone_tail.decoder_out, teacher.l1_out))
                        recon_loss = 10*(mse_loss(model_tail.backbone_tail.l2_out, teacher.l2_out))
                        recon_loss += 10*(mse_loss(model_tail.backbone_tail.l3_out, teacher.l3_out))
                        recon_loss += 10*(mse_loss(model_tail.backbone_tail.l4_out, teacher.l4_out))

                        # print(f'mse_loss: %.5f | bpp_loss: %.5f | aux_loss: %.5f | recon_loss %.5f |' %(instace_mse_loss.item(), bpp_loss.item(), aux_loss.item(), recon_loss.item()))
                        batch_loss += instace_mse_loss + recon_loss + bpp_loss + aux_loss

                    elif training_stage == '2':
                        # batch_loss += 0 if torch.isnan(loss['loss_classifier']).item() else loss['loss_classifier']
                        # batch_loss += 0 if torch.isnan(loss['loss_box_reg']).item() else loss['loss_box_reg']
                        # batch_loss += 0 if torch.isnan(loss['loss_mask']).item() else loss['loss_mask']
                        # batch_loss += 0 if torch.isnan(loss['loss_objectness']).item() else loss['loss_objectness'] 
                        # batch_loss += 0 if torch.isnan(loss['loss_rpn_box_reg']).item() else loss['loss_rpn_box_reg']
                        # batch_loss += (loss_1['loss_classifier'] + loss_1['loss_box_reg'] + loss_1['loss_mask'] + loss_1['loss_objectness'] + loss_1["loss_rpn_box_reg"]) 
                        batch_loss += (loss['loss_classifier'] + loss['loss_box_reg'] + loss['loss_mask'] + loss['loss_objectness'] + loss["loss_rpn_box_reg"]) 
                        # print(mse_target)
                        # print(model_tail.backbone_tail.decoder_out)
                        # print(torch.isnan(model_tail.backbone_tail.decoder_out).any())
                        # mloss = mse_loss(mse_target, model_tail.backbone_tail.decoder_out)
                        # print(mloss)
                        # batch_loss += 0 if (torch.isnan(mloss).item() or torch.isinf(mloss).item()) else mloss

                # backpropagation
                batch_loss.backward()
                total_loss += batch_loss.item()/accum
                batch_loss = 0
                
                if accum_counter == accum:
                    if not train_teacher:
                        torch.nn.utils.clip_grad_norm_(model_head.parameters(), clip_value)
                        torch.nn.utils.clip_grad_norm_(model_tail.parameters(), clip_value)
                    optimizer.step()
                    optimizer.zero_grad()
                    accum_counter = 1
                    i+=1
                    printed = False
                else:
                    accum_counter += 1

                #print statments
                if i != 0 and i % 33 == 0 and printed == False:
                    print('Batch: ', (i+1))
                    if training_stage == '1' and not train_teacher:
                        if entropy_model:
                            print(f'mse_loss: %.5f | bpp_loss: %.5f | aux_loss: %.5f | recon_loss %.5f |' %(instace_mse_loss.item(), bpp_loss.item(), aux_loss.item(), recon_loss.item()))
                        else:
                            print(f'mse_loss: %.5f | recon_loss %.5f |' %(instace_mse_loss.item(), recon_loss.item()))

                    print(loss)
                    print('loss:', total_loss/(i+1))
                    sys.stdout.flush()
                    printed = True

                    #i+= 1
            
            loss_list.append(total_loss/(i+1))

            if train_teacher:
                if use_nu:
                    torch.save(teacher.state_dict(), "models/teachers/nu_teacher_seg.pth")
                elif use_rsud:
                    torch.save(teacher.state_dict(), "models/teachers/rsud_teacher_seg.pth")
            else:
                if training_stage == '1':
                    if entropy_model:
                        torch.save(model_head.state_dict(), 'models/coco_byte/MaskRCNN_head-S1-'+str(channels)+'CH-L0500.pth')
                        torch.save(model_tail.state_dict(), 'models/coco_byte/MaskRCNN_tail-S1-'+str(channels)+'CH-L0500.pth')
                    else:
                        torch.save(model_head.state_dict(), 'models/coco_byte/stand/MaskRCNN_head-S1-'+str(channels)+'CH.pth')
                        torch.save(model_tail.state_dict(), 'models/coco_byte/stand/MaskRCNN_tail-S1-'+str(channels)+'CH.pth')
                elif training_stage == '2':
                    if entropy_model:
                        torch.save(model_head.state_dict(), 'models/coco_byte/MaskRCNN_head-S2-'+str(channels)+'CH-L0500.pth')
                        torch.save(model_tail.state_dict(), 'models/coco_byte/MaskRCNN_tail-S2-'+str(channels)+'CH-L0500.pth')
                    else:
                        torch.save(model_head.state_dict(), 'models/coco_byte/stand/MaskRCNN_head-S2-'+str(channels)+'CH.pth')
                        torch.save(model_tail.state_dict(), 'models/coco_byte/stand/MaskRCNN_tail-S2-'+str(channels)+'CH.pth')
            print('saving model')
            print('Total Loss:', total_loss/(i+1))


        #BEGIN EVAL SECTION
        if train_teacher:
            teacher.eval()
        else:
            model_head.eval()
            model_tail.eval()
            if entropy_model and training_stage == '2':
                model_head.backbone_head.encoder.is_testing = True
                model_tail.backbone_tail.decoder.is_testing = True  
                model_head.backbone_head.encoder.entropy_bb.update()
                model_tail.backbone_tail.decoder.entropy_bb_for_decode.update()

        metric = MeanAveragePrecision(iou_type="segm")

        with torch.no_grad():
            avg_size = 0
            i = 0
            print('=== Validation epoch: ', epoch, '===')
            sys.stdout.flush()
            for data in val_loader:#tqdm(val_loader):#

                # print(data)

                for d in data[1]:
                    if 'image_id' in d:
                        del d['image_id']

                images = list(image.to(device) for image in data[0])#list(image.to for image in data[0])#
                targets = [{k: v.to(device) for k, v in t.items()} if t else {k: v.to(device) for k, v in {"boxes": torch.zeros(0,4).to(device), "labels": torch.zeros(1).type(torch.int64).to(device)}.items()} for t in data[1]]
                #[{k: v for k, v in t.items()} if t else {k: v for k, v in {"boxes": torch.zeros(0,4), "labels": torch.zeros(1).type(torch.int64)}.items()} for t in data[1]]

                if use_rsud:
                    image = cv2.imread(data[2][0])
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # masks = mask_generator.generate(image)
                    mask_generator.set_image(image)

                    transformed_boxes = mask_generator.transform.apply_boxes_torch(targets[0]["boxes"], image.shape[:2])
                    masks, _, _ = mask_generator.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes,
                        multimask_output=False,
                    )

                    masks = masks.squeeze(1)
                    masks = masks.to(torch.uint8)
                    targets[0]['masks'] = masks

                if (training_stage == '1' or training_stage == '2') and not test_teacher:
                    features_from_head, likelihoods_from_head, images_sizes, images_size, original_image_sizes, _ = model_head(images)
                    if entropy_model and training_stage == '2':
                        avg_size += sys.getsizeof(features_from_head[0])
                        fix_prob = 0.25
                        byte_loss_mean = 0
                        byte_loss_var = 0.33
                        drop_prob = 0.75
                        for f_indx in range(len(features_from_head)):
                            total_size = len(features_from_head[f_indx])
                            lost_bytes = 0
                            for s_indx in range(len(features_from_head[f_indx])):
                                if s_indx >= len(features_from_head[f_indx]):
                                    break
                                we_drop = abs(random.gauss(byte_loss_mean, byte_loss_var)) + fix_prob
                                if we_drop >= drop_prob:
                                    features_from_head[f_indx] = features_from_head[f_indx][:(s_indx-lost_bytes)] + features_from_head[f_indx][-(len(features_from_head[f_indx])-(s_indx-lost_bytes+1)):]
                                    lost_bytes += 1
                    detections = model_tail(features_from_head,  images_sizes, images_size, original_image_sizes, size=likelihoods_from_head)
                else:
                    detections = teacher(images)

                for index in range(len(detections)):
                    B, C, H, W = detections[index]["masks"].shape
                    detections[index]["masks"] = (detections[index]["masks"]>mask_threshold).to(torch.uint8).reshape(B, H, W)

                    detections[index] = {"masks": detections[index]["masks"], "labels": detections[index]["labels"], "scores": detections[index]["scores"]}
                    targets[index] = {"masks": targets[index]["masks"], "labels": targets[index]["labels"]}

                # detections = teacher(images)

                metric.update(detections, targets)
                if quality is not None and test_teacher:
                    avg_size += data[3][0]#[2][0]
                else:
                    avg_size += 0
                i+=1

                # if i%1000 == 0:
                #     print('mAP:', metric.compute()['map'])
                #     print('mAP_50%:', metric.compute()['map_50'])

            if (entropy_model and training_stage == '2') or quality is not None:
                print('average_compressed_size:', avg_size/(len(val_loader)*eval_batch_size))

            print(metric.compute()) 
            sys.stdout.flush()

    return loss_list

learning_rate = 2e-2 #2e-2
n_epochs = 30

loss_list = training_loop(detector_head, detector_tail, learning_rate, data_loader, n_epochs, val_loader, train_batch_size)

