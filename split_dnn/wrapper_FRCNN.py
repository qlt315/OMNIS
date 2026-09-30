import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union
import time

import torch
import torchvision
import torch.nn.functional as F
from torch import nn, Tensor
from split_resnet50 import ResNetHead, ResNetTail, ResidualBlock50

from rpn_updated import RegionProposalNetwork

#This to wrap the pytorch frcnn model to extract inter-layer features for knowledge distilation
class FRCNN_wrapper(nn.Module):
    def __init__(self):
        super().__init__()

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="COCO_V1", pretrained=True)
  
        self.transform = model.transform
        self.backbone_body = model.backbone.body
        self.backbone_fpn = model.backbone.fpn
        self.rpn = model.rpn
        self.roi_heads = model.roi_heads

        # used only on torchscript mode
        self._has_warned = False

        del model

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses, detections

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """

        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    #if target:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        features = self.backbone_body.conv1(images.tensors)
        features = self.backbone_body.bn1(features)
        features = self.backbone_body.relu(features)
        features = self.backbone_body.maxpool(features)


        self.l1_out = self.backbone_body.layer1(features)
        # print(self.l1_out.shape)
        self.l2_out = self.backbone_body.layer2(self.l1_out)
        self.l3_out = self.backbone_body.layer3(self.l2_out)
        self.l4_out = self.backbone_body.layer4(self.l3_out)

        features2 = OrderedDict([("0", self.l1_out), ("1", self.l2_out), ("2", self.l3_out), ("3", self.l4_out)])

        features2 = self.backbone_fpn(features2)

        #If needed for fpn distilation
        # self.fp1_out = features2["0"]
        # self.fp2_out = features2["1"]
        # self.fp3_out = features2["2"]
        # self.fp4_out = features2["3"]

        proposals, proposal_losses = self.rpn(images, features2, targets)
        detections, detector_losses = self.roi_heads(features2, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class FRCNN_wrapper_HEAD(nn.Module):
    def __init__(self, bottleneck_channel=6, is_training=True, entropy_split=True, use_nu=False, new_standard=False): #we only need from_old if using model before we created new ROIHeads
        super().__init__()

        self.is_traing = False

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="COCO_V1", pretrained=True)


        self.transform = model.transform
        self.backbone_head = ResNetHead(bottleneck_channel, is_training, entropy_split=entropy_split, new_standard=new_standard) 
        self.backbone_head.conv1 = model.backbone.body.conv1

        self.entropy_split = entropy_split

        del model

    def forward(self, images, targets=None):
        #start_time = time.time()

        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    #if target:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )


        if self.entropy_split:
            head_features, head_likelihoods = self.backbone_head(images.tensors)
            #to_split_time = time.time()
            return head_features, head_likelihoods, images.image_sizes, images.tensors.shape[-2:], original_image_sizes, targets
        else:
            head_features = self.backbone_head(images.tensors)
            #to_split_time = time.time()
            return head_features, None, images.image_sizes, images.tensors.shape[-2:], original_image_sizes, targets

class FRCNN_wrapper_TAIL(nn.Module):
    def __init__(self, bottleneck_channel=6, entropy_split=True):
        super().__init__()

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="COCO_V1", pretrained=True)

        self.transform = model.transform
        
        self.backbone_tail = ResNetTail(ResidualBlock50, [3, 4, 6, 3], 1000, bottleneck_channel, keep_heads=False, entropy_split=entropy_split)
        self.backbone_tail.layer2 = model.backbone.body.layer2
        self.backbone_tail.layer3 = model.backbone.body.layer3
        self.backbone_tail.layer4 = model.backbone.body.layer4

        self.fpn = model.backbone.fpn

        self.rpn = RegionProposalNetwork(model.rpn) 
        self.roi_heads = model.roi_heads

        del model

    @torch.jit.unused
    def eager_outputs(self, losses, detections,):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses, detections

        return detections

    # passed image sizes over bc quantization changes the images
    def forward(self, features_from_head,  images_sizes, images_size, original_image_sizes, targets=None, size=None):

        decompressed = self.backbone_tail(features_from_head, size)
        decompressed = OrderedDict([("0", self.backbone_tail.decoder_out), ("1", self.backbone_tail.l2_out), ("2", self.backbone_tail.l3_out), ("3", self.backbone_tail.l4_out)])

        decompressed = self.fpn(decompressed)

        #If needed for fpn distilation
        # self.fp1_out = decompressed["0"]
        # self.fp2_out = decompressed["1"]
        # self.fp3_out = decompressed["2"]
        # self.fp4_out = decompressed["3"]

        if isinstance(decompressed, torch.Tensor):
            decompressed = OrderedDict([("0", decompressed)])

        # proposals, proposal_losses = self.rpn(images_size, decompressed, targets)
        # detections, detector_losses = self.roi_heads(decompressed, proposals, images_sizes, targets)
        # detections = self.transform.postprocess(detections, images_sizes, original_image_sizes)

        proposals, proposal_losses = self.rpn(images_size, images_sizes, decompressed, targets)
        detections, detector_losses = self.roi_heads(decompressed, proposals, images_sizes, targets)
        detections = self.transform.postprocess(detections, images_sizes, original_image_sizes) 

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        # jit = torch's compiler (efficient binary code)
        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)
        
# if __name__ == "__main__":
#     model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="COCO_V1")
#     print(model.backbone.body.conv1)

#     randvar1 = torch.rand(1,4,100,100)
#     randvar2 = torch.rand(1,4,100,100)
    
#     randvar1.requires_grad = True
#     randvar2.requires_grad = True

#     print(randvar1)
#     print(randvar2)

#     metric = StructuralSimilarityIndexMeasure(data_range=1.0)
#     print(metric(randvar1,randvar2))
#     # print(ssim_loss(randvar1, randvar2, window_size=11))


#     input_tensor = [torch.rand(3,640,480), torch.rand(3,640,480)]    # batch, channels, height, width

#     model = FRCNN_wrapper().eval()
#     print(model.backbone.body.conv1)
#     model_out = model(input_tensor)
#     # wrapper = FRCNN_wrapper(max_size=700,min_size=700).eval()
#     # print(wrapper)
#     frcnn_head = FRCNN_wrapper_HEAD().eval()     # not in training mode when testing (default is)
#     # frcnn_tail = FRCNN_wrapper_TAIL().eval()

#     wrapper_out, _, _1, _2 = frcnn_head(input_tensor)

#     # print(model_out)
#     # print(wrapper_out)

#     print('----------------------------')

#     input_tensor = [torch.rand(3,480,600), torch.rand(3,640,480)]    # batch, channels, height, width

#     model_out = model(input_tensor)
#     wrapper_out, _, _1, _2 = frcnn_head(input_tensor)

#     # print(model_out)
#     # print(wrapper_out)

#     print('----------------------------')

#     input_tensor = [torch.rand(3,600,550), torch.rand(3,575,420)]    # batch, channels, height, width

#     model_out = model(input_tensor)
#     wrapper_out, _, _1, _2 = frcnn_head(input_tensor)

#     # print(model_out)
#     # print(wrapper_out)

#     # print(ssim_loss(model.l1_kd_out, wrapper.l1_out, window_size=11))

#     # features, images_sizes, images_size, original_img_sizes = frcnn_head(input_tensor)
#     # detections = frcnn_tail(images_size, images_sizes, original_img_sizes, features)
#     # print(wrapper_out)
#     # print(detections)
