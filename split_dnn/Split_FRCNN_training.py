import gc
import random
import sys 
import math

from boxcoder import process_raw
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
train_batch_size = 2 #2
eval_batch_size = 1
accum = 4 #4
clip_value = 5

learning_rate = 2e-2 #2e-2
n_epochs = 5
eval_epoch = 1

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

channels = 3
training_stage = '2' # 1, 2
entropy_model = False

testing_quant_with_noise = False
new_standard = False
new_standard_ft = None # 1, 2, None #should only be not None when training stage is set to 2
# train_over_noise = False
packet_size_values = [512, 256, 128]
drop_probs = [0.25899, 0.48810, 0.930145]
#-4,-2,0,2,4,6,8,10
bers = [7.86496035e-02, 5.62819520e-02, 3.75061284e-02, 2.28784076e-02,
 1.25008180e-02, 5.95386715e-03, 2.38829078e-03, 7.72674815e-04,
 1.90907774e-04, 3.36272284e-05, 3.87210822e-06, 2.61306795e-07,
 9.00601035e-09, 1.33293102e-10, 6.81018913e-13, 9.12395736e-16,
 2.26739584e-19, 6.75896977e-24, 1.39601431e-29, 1.00107397e-36,
 1.04424379e-45]

first_loop = [False] #[True]

testing = False
train_teacher = False
test_teacher = False

device = torch.device('cuda:0') 
if training_stage == '1' or training_stage == '2' or train_teacher or new_standard_ft == '1':
    teacher = FRCNN_wrapper().to(device).eval()#torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="COCO_V1").to(device).eval()
    if use_nu:
        teacher.load_state_dict(torch.load('models/teachers/nu_teacher.pth'),strict=False)
    if use_rsud:
        teacher.load_state_dict(torch.load('models/teachers/rsud_teacher.pth'),strict=False)

if not train_teacher:
    detector_head = FRCNN_wrapper_HEAD(bottleneck_channel=channels, is_training=(training_stage=='1'), entropy_split=entropy_model, new_standard=new_standard).to(device)#torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="COCO_V1").to(device)#
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
            detector_head.load_state_dict(torch.load('models/nu_models/FRCNN_head-S1-'+str(channels)+'CH-L0500.pth'))
            detector_tail.load_state_dict(torch.load('models/nu_models/FRCNN_tail-S1-'+str(channels)+'CH-L0500.pth'), strict=False)
        elif use_rsud:
            detector_head.load_state_dict(torch.load('models/rsud_models/FRCNN_head-S1-'+str(channels)+'CH-L0500.pth'))
            detector_tail.load_state_dict(torch.load('models/rsud_models/FRCNN_tail-S1-'+str(channels)+'CH-L0500.pth'), strict=False)
        else:
            detector_head.load_state_dict(torch.load('models/entrop/FRCNN_head-S1-'+str(channels)+'CH-L0500.pth'))
            detector_tail.load_state_dict(torch.load('models/entrop/FRCNN_tail-S1-'+str(channels)+'CH-L0500.pth'), strict=False)
        detector_tail.backbone_tail.decoder.entropy_bb_for_decode = deepcopy(detector_head.backbone_head.encoder.entropy_bb)
        detector_tail.backbone_tail.decoder.entropy_bb_for_decode.requires_grad = False
    elif new_standard_ft == '2':
        detector_head.load_state_dict(torch.load('models/new_stand/stand/FRCNN_head-S2ft1-'+str(channels)+'CH.pth'), strict=False)
        detector_tail.load_state_dict(torch.load('models/new_stand/stand/FRCNN_tail-S2ft1-'+str(channels)+'CH.pth'))
    elif new_standard and training_stage == '2':
        detector_head.load_state_dict(torch.load('models/new_stand/stand//FRCNN_head-S1-'+str(channels)+'CH.pth'), strict=False)
        detector_tail.load_state_dict(torch.load('models/new_stand/stand/FRCNN_tail-S1-'+str(channels)+'CH.pth'))
    elif training_stage == '2' or new_standard:
        detector_head.load_state_dict(torch.load('models/standard/FRCNN_head-S1-'+str(channels)+'CH.pth'), strict=False)
        detector_tail.load_state_dict(torch.load('models/standard/FRCNN_tail-S1-'+str(channels)+'CH.pth'))
else:
    detector_head = None
    detector_tail = None

def training_loop(model_head, model_tail, learning_rate, train_dataloader, n_epochs, val_loader, batch_size):
    
    if train_teacher:
        optimizer = torch.optim.SGD(teacher.parameters(), lr = learning_rate, momentum=0.5, weight_decay=1e-5)
    else:
        if training_stage == '1':
            optimizer = torch.optim.SGD((list(model_head.parameters()) + list(model_tail.backbone_tail.decoder.parameters())), lr=learning_rate, momentum=0.5, weight_decay=1e-5)
        elif new_standard_ft == '1':
            optimizer = torch.optim.SGD((list(model_tail.backbone_tail.decoder.parameters())), lr=learning_rate, momentum=0.5, weight_decay=1e-5)
        elif (training_stage == '2' and new_standard) or new_standard_ft == '2':
            optimizer = torch.optim.SGD((list(model_tail.backbone_tail.layer2.parameters()) + 
                               list(model_tail.backbone_tail.layer3.parameters()) + list(model_tail.backbone_tail.layer4.parameters()) + 
                               list(model_tail.fpn.parameters()) + list(model_tail.rpn.parameters()) + 
                               list(model_tail.roi_heads.parameters())), lr=learning_rate, momentum=0.5, weight_decay=1e-5)
        elif training_stage == '2':
            optimizer = torch.optim.SGD((list(model_tail.backbone_tail.decoder.parameters()) + list(model_tail.backbone_tail.layer2.parameters()) + 
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
                if new_standard:
                    model_head.backbone_head.encoder.quant_thing.is_training = True
                    model_head.backbone_head.encoder.to_bits = False
            elif training_stage == '2':
                model_head.eval()
                model_tail.train()
                if entropy_model:
                    model_head.backbone_head.encoder.is_testing = False
                    model_head.backbone_head.encoder.is_training = False # so we can train over converted bits
                    model_tail.backbone_tail.decoder.is_testing = False
                    model_tail.backbone_tail.decoder.entropy_bb_for_decode.eval()
                if new_standard:
                    model_head.backbone_head.encoder.quant_thing.is_training = True
                    model_head.backbone_head.encoder.to_bits = False
                if new_standard_ft == '1':
                    model_tail.eval()
                    model_tail.backbone_tail.decoder.train()

        batch_loss = 0
        total_loss = 0

        for g in optimizer.param_groups:
            print(g['lr'])
            g['lr'] = g['lr'] - lr2

        # map_val = 0
        i = 0
        accum_counter = 1
        datasize_accum = 0
        printed = False
        if not testing and not first_loop[0]:
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

                # if use_nu:
                #     for k in range(len(targets)):
                #         for j in range(len(targets[k]['labels'])):
                #             targets[k]['labels'][j] += 1
                    

                # forward pass
                if train_teacher:
                    loss, detections = teacher(images, targets)
                    batch_loss += (loss['loss_classifier'] + loss['loss_box_reg']) 
                else:
                    if training_stage == '1' or new_standard_ft == '1':
                        detections = teacher(images)

                    if new_standard_ft == '1':
                        with torch.no_grad():
                            features_from_head, likelihoods_from_head, images_sizes, images_size, original_image_sizes, targets = model_head(images, targets)
                        detections = model_tail(features_from_head,  images_sizes, images_size, original_image_sizes, targets)
                    else:
                        features_from_head, likelihoods_from_head, images_sizes, images_size, original_image_sizes, targets = model_head(images, targets)
                        loss, detections = model_tail(features_from_head,  images_sizes, images_size, original_image_sizes, targets)
                    
                    # loss caculations
                    if training_stage == '1':
                        if entropy_model:
                            N, C, H, W = features_from_head.size()
                            num_pixels = N * C * H * W

                            
                            bpp_loss = (0.01*(torch.log(likelihoods_from_head).sum() / (-math.log(2) * num_pixels)))
                            aux_loss = model_head.backbone_head.encoder.entropy_bb.loss()
                        elif new_standard:
                            bpp_loss = 0.032 * model_head.backbone_head.encoder.quant_thing.bpp #0.15, 0.075, 0.0375
                            aux_loss = 0
                        else:
                            bpp_loss = 0
                            aux_loss = 0

                        instace_mse_loss = 10*(mse_loss(model_tail.backbone_tail.decoder_out, teacher.l1_out))
                        recon_loss = 20*(mse_loss(model_tail.backbone_tail.l2_out, teacher.l2_out))
                        recon_loss += 20*(mse_loss(model_tail.backbone_tail.l3_out, teacher.l3_out))
                        recon_loss += 20*(mse_loss(model_tail.backbone_tail.l4_out, teacher.l4_out))

                        # print(f'mse_loss: %.5f | bpp_loss: %.5f | aux_loss: %.5f | recon_loss %.5f |' %(instace_mse_loss.item(), bpp_loss.item(), aux_loss.item(), recon_loss.item()))
                        batch_loss += instace_mse_loss + recon_loss + bpp_loss + aux_loss

                    elif training_stage == '2':
                        if new_standard_ft == '1':
                            # instace_mse_loss = 10*(mse_loss(model_tail.backbone_tail.decoder_out, teacher.l1_out))
                            # recon_loss = 20*(mse_loss(model_tail.backbone_tail.l2_out, teacher.l2_out))
                            # recon_loss += 20*(mse_loss(model_tail.backbone_tail.l3_out, teacher.l3_out))
                            # recon_loss += 20*(mse_loss(model_tail.backbone_tail.l4_out, teacher.l4_out))

                            # batch_loss += 0.95 * (instace_mse_loss + recon_loss) 
                            # batch_loss += 0.95 * (loss['loss_classifier'] + loss['loss_box_reg'] + loss['loss_mask'] + loss['loss_objectness'] + loss["loss_rpn_box_reg"]) 

                            # batch_loss.backward()
                            # total_loss += batch_loss.item()/accum
                            # primary_display_loss = batch_loss.item()
                            batch_loss = 0

                            # soft_targets = [
                            #     model_tail.backbone_tail.decoder_out.clone().detach().to(device),
                            #     model_tail.backbone_tail.l2_out.clone().detach().to(device),
                            #     model_tail.backbone_tail.l3_out.clone().detach().to(device),
                            #     model_tail.backbone_tail.l4_out.clone().detach().to(device),
                            # ]

                            training_pairs = [[4,4]]#,[2, 2]]]
                            # random_pair = [2,0]
                            # while random_pair == training_pairs[0]: #or random_pair != training_pairs[1]:
                            #     random_pair = [2, random.randint(1,3)]

                            # training_pairs.append(random_pair)

                            secondary_display_loss = 0
                            model_head.backbone_head.encoder.to_bits = True
                            for pair_idx in range(len(training_pairs)):
                                pack_size = packet_size_values[training_pairs[pair_idx][0]]
                                # drop_prob = drop_probs[training_pairs[pair_idx][1]]
                                ber = bers[training_pairs[pair_idx][1]]

                                model_head.backbone_head.encoder.packet_size_value = pack_size
                                # model_head.backbone_head.encoder.drop_prob = drop_prob
                                model_head.backbone_head.encoder.ber = ber

                                with torch.no_grad():
                                    features_from_head, likelihoods_from_head, images_sizes, images_size, original_image_sizes, targets = model_head(images, targets)
                                loss, detections = model_tail(features_from_head,  images_sizes, images_size, original_image_sizes, targets)
                                # detections = model_tail(features_from_head,  images_sizes, images_size, original_image_sizes, targets)

                                # instace_mse_loss = 10*(mse_loss(model_tail.backbone_tail.decoder_out, soft_targets[0]))
                                # recon_loss = 20*(mse_loss(model_tail.backbone_tail.l2_out, soft_targets[1]))
                                # recon_loss += 20*(mse_loss(model_tail.backbone_tail.l3_out, soft_targets[2]))
                                # recon_loss += 20*(mse_loss(model_tail.backbone_tail.l4_out, soft_targets[3]))

                                # instace_mse_loss = 10*(mse_loss(model_tail.backbone_tail.decoder_out, teacher.l1_out))
                                # recon_loss = 20*(mse_loss(model_tail.backbone_tail.l2_out, teacher.l2_out))
                                # recon_loss += 20*(mse_loss(model_tail.backbone_tail.l3_out, teacher.l3_out))
                                # recon_loss += 20*(mse_loss(model_tail.backbone_tail.l4_out, teacher.l4_out))

                                # batch_loss +=  0.05 * ((instace_mse_loss + recon_loss)/(len(training_pairs))) #(0.8 if (pair_idx == 0) else 0.2)
                                batch_loss += (loss['loss_classifier'] + loss['loss_box_reg'] + loss['loss_objectness'] + loss["loss_rpn_box_reg"]) 


                                batch_loss.backward()
                                total_loss += batch_loss.item()/accum
                                secondary_display_loss += batch_loss.item()
                                batch_loss = 0
                                
                            model_head.backbone_head.encoder.to_bits = False
                            # del soft_targets
                            
                        elif new_standard_ft == '2':
                            ...
                        else:
                            batch_loss += (loss['loss_classifier'] + loss['loss_box_reg'] + loss['loss_objectness'] + loss["loss_rpn_box_reg"]) 

                # backpropagation
                if new_standard_ft is None:
                    batch_loss.backward()
                    total_loss += batch_loss.item()/accum
                if new_standard:
                    datasize_accum += model_head.backbone_head.encoder.quant_thing.byte_size
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
                    gc.collect()
                    torch.cuda.empty_cache()
                    # print('we step')
                else:
                    accum_counter += 1
                #print statments
                if i != 0 and i % 10 == 0 and printed == False:
                    print('Batch: ', (i+1))
                    if training_stage == '1' and not train_teacher:
                        if entropy_model:
                            print(f'mse_loss: %.5f | bpp_loss: %.5f | aux_loss: %.5f | recon_loss %.5f |' %(instace_mse_loss.item(), bpp_loss.item(), aux_loss.item(), recon_loss.item()))
                        elif new_standard:
                            print(f'mse_loss: %.5f | bpp_loss: %.5f | recon_loss %.5f |' %(instace_mse_loss.item(), bpp_loss.item(), recon_loss.item()))
                            print(f'compressed_datasize: %.3f' %(datasize_accum/(10*accum)))
                            datasize_accum = 0
                        else:
                            print(f'mse_loss: %.5f | recon_loss %.5f |' %(instace_mse_loss.item(), recon_loss.item()))

                    if training_stage == '2' and new_standard:
                        if new_standard_ft == '1':
                            primary_display_loss = 0.0
                            print(f'primary_loss: %.5f | secondary_loss %.5f |' %(primary_display_loss, secondary_display_loss))
                        elif new_standard_ft == '2':
                            ...
                        print(f'compressed_datasize: %.3f' %(datasize_accum/(10*accum)))
                        datasize_accum = 0

                    print('loss:', total_loss/(i+1))
                    sys.stdout.flush()
                    printed = True

                    #i+= 1
            
            loss_list.append(total_loss/(i+1))

            if train_teacher:
                if use_nu:
                    torch.save(teacher.state_dict(), "models/teachers/nu_teacher.pth")
                elif use_rsud:
                    torch.save(teacher.state_dict(), "models/teachers/rsud_teacher.pth")
            else:
                if training_stage == '1':
                    if entropy_model:
                        if use_nu:
                            torch.save(model_head.state_dict(), 'models/nu_models/FRCNN_head-S1-'+str(channels)+'CH-L0500.pth')
                            torch.save(model_tail.state_dict(), 'models/nu_models/FRCNN_tail-S1-'+str(channels)+'CH-L0500.pth')
                        elif use_rsud:
                            torch.save(model_head.state_dict(), 'models/rsud_models/FRCNN_head-S1-'+str(channels)+'CH-L0500.pth')
                            torch.save(model_tail.state_dict(), 'models/rsud_models/FRCNN_tail-S1-'+str(channels)+'CH-L0500.pth')
                        else:
                            torch.save(model_head.state_dict(), 'models/entrop/FRCNN_head-S1-'+str(channels)+'CH-L0500.pth')
                            torch.save(model_tail.state_dict(), 'models/entrop/FRCNN_tail-S1-'+str(channels)+'CH-L0500.pth')
                    if new_standard:
                        torch.save(model_head.state_dict(), 'models/new_stand/stand/FRCNN_head-S1-'+str(channels)+'CH.pth')
                        torch.save(model_tail.state_dict(), 'models/new_stand/stand/FRCNN_tail-S1-'+str(channels)+'CH.pth')
                    else:
                        torch.save(model_head.state_dict(), 'models/standard/FRCNN_head-S1-'+str(channels)+'CH.pth')
                        torch.save(model_tail.state_dict(), 'models/standard/FRCNN_tail-S1-'+str(channels)+'CH.pth')
                elif training_stage == '2':
                    if entropy_model:
                        if use_nu:
                            torch.save(model_head.state_dict(), 'models/nu_models/FRCNN_head-S2-'+str(channels)+'CH-L0500.pth')
                            torch.save(model_tail.state_dict(), 'models/nu_models/FRCNN_tail-S2-'+str(channels)+'CH-L0500.pth')
                        elif use_rsud:
                            torch.save(model_head.state_dict(), 'models/rsud_models/FRCNN_head-S2-'+str(channels)+'CH-L0500.pth')
                            torch.save(model_tail.state_dict(), 'models/rsud_models/FRCNN_tail-S2-'+str(channels)+'CH-L0500.pth')
                        else:
                            torch.save(model_head.state_dict(), 'models/entrop/FRCNN_head-S2-'+str(channels)+'CH-L0500.pth')
                            torch.save(model_tail.state_dict(), 'models/entrop/FRCNN_tail-S2-'+str(channels)+'CH-L0500.pth')
                    elif new_standard_ft == '1':
                        torch.save(model_head.state_dict(), 'models/new_stand/stand/FRCNN_head-S2ft1-'+str(channels)+'CH.pth')
                        torch.save(model_tail.state_dict(), 'models/new_stand/stand/FRCNN_tail-S2ft1-'+str(channels)+'CH.pth')
                    elif new_standard_ft == '2':
                        torch.save(model_head.state_dict(), 'models/new_stand/stand/FRCNN_head-S2ft2-'+str(channels)+'CH.pth')
                        torch.save(model_tail.state_dict(), 'models/new_stand/stand/FRCNN_tail-S2ft2-'+str(channels)+'CH.pth')
                    if new_standard:
                        torch.save(model_head.state_dict(), 'models/new_stand/stand/FRCNN_head-S2-'+str(channels)+'CH.pth')
                        torch.save(model_tail.state_dict(), 'models/new_stand/stand/FRCNN_tail-S2-'+str(channels)+'CH.pth')
                    else:
                        torch.save(model_head.state_dict(), 'models/standard/FRCNN_head-S2-'+str(channels)+'CH.pth')
                        torch.save(model_tail.state_dict(), 'models/standard/FRCNN_tail-S2-'+str(channels)+'CH.pth')
            print('saving model')
            print('Total Loss:', total_loss/(i+1))


        if first_loop[0]:
            first_loop[0] = False

        #BEGIN EVAL SECTION
        if train_teacher:
            teacher.eval()
        else:
            model_head.eval()
            model_tail.eval()
            if new_standard:
                model_head.backbone_head.encoder.quant_thing.is_training = False
                model_head.backbone_head.encoder.to_bits = False #train_over_noise #training_stage == '2'
                if new_standard_ft == '1':
                    model_tail.backbone_tail.use_gt = False #True
                    model_head.backbone_head.encoder.to_bits = True

                    model_head.backbone_head.encoder.packet_size_value = packet_size_values[-1]
                    # model_head.backbone_head.encoder.drop_prob = drop_probs[-1]
                    model_head.backbone_head.encoder.ber = bers[12] #0,2,4,6,8,10,12
                    print('its true')
            if entropy_model and training_stage == '2':
                model_head.backbone_head.encoder.is_testing = True
                model_tail.backbone_tail.decoder.is_testing = True  
                model_head.backbone_head.encoder.entropy_bb.update()
                model_tail.backbone_tail.decoder.entropy_bb_for_decode.update()
        metric = MeanAveragePrecision(iou_type="bbox")
        # iou_metric = IntersectionOverUnion()

        if ((epoch+1)%eval_epoch==0):
            with torch.no_grad():
                avg_compressed = 0
                avg_size = 0
                i = 0
                print('=== Validation epoch: ', epoch, '===')
                sys.stdout.flush()
                for data in val_loader:#tqdm(val_loader):#
                    if (i+1)%250 == 0:
                        # if i >10:
                        #     break
                        print(i)
                        if training_stage =='2' and new_standard:
                            print(model_head.backbone_head.encoder.quant_thing.byte_size)
                            print(model_head.backbone_head.encoder.quant_thing.compressed_size)
                        if testing_quant_with_noise:
                            print(avg_size/(i+1))
                        mdict = metric.compute()
                        print('map@COCO:', mdict['map'], 'map@50:', mdict['map_50'], 'map@75:', mdict['map_75'])
                        sys.stdout.flush()
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


                    if (training_stage == '1' or training_stage == '2') and (not test_teacher and not train_teacher):
                        features_from_head, likelihoods_from_head, images_sizes, images_size, original_image_sizes, _ = model_head(images)
                        if entropy_model and training_stage == '2':
                            avg_size += sys.getsizeof(features_from_head[0])
                        if testing_quant_with_noise and training_stage == '2':
                            features_from_head, size_holder = process_raw(features_from_head, bers[0])
                            avg_size += size_holder
                        detections = model_tail(features_from_head,  images_sizes, images_size, original_image_sizes, size=likelihoods_from_head)
                    else:
                        detections = teacher(images)

                    # if use_nu:
                    #     for k in range(len(detections)):
                    #         for i in range(len(detections[k]['labels'])):
                    #             if detections[k]['labels'][i] in coco_to_nu_label_map.keys():
                    #                 detections[k]['labels'][i] = coco_to_nu_label_map[detections[k]['labels'][i]]

                    metric.update(detections, targets)
                    if quality is not None and test_teacher:
                        avg_size += data[2][0]
                    elif new_standard:
                        avg_size += model_head.backbone_head.encoder.quant_thing.byte_size
                        avg_compressed += model_head.backbone_head.encoder.quant_thing.compressed_size
                    else:
                        avg_size += 0
                    i+=1

                    # if i%1000 == 0:
                    #     print('mAP:', metric.compute()['map'])
                    #     print('mAP_50%:', metric.compute()['map_50'])

                if (entropy_model and training_stage == '2') or quality is not None or new_standard or testing_quant_with_noise:
                    print('average_compressed_size:', avg_size/(len(val_loader)*eval_batch_size))
                    if new_standard:
                        print('raw compressed size:', avg_compressed/(len(val_loader)*eval_batch_size))

                print(metric.compute()) 
                sys.stdout.flush()

    return loss_list

loss_list = training_loop(detector_head, detector_tail, learning_rate, data_loader, n_epochs, val_loader, train_batch_size)

