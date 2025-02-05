# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from engine import train_one_epoch, evaluate
import utils
import transforms as T


class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

#def get_model_instance_segmentation(num_classes):
#    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
#
#    # get number of input features for the classifier
#    in_features = model.roi_heads.box_predictor.cls_score.in_features
#    # replace the pre-trained head with a new one
#    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#
#    # now get the number of input features for the mask classifier
#    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
#    hidden_layer = 256
#    # and replace the mask predictor with a new one
#    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
#                                                       hidden_layer,
#                                                       num_classes)
#
#    return model

def get_model_instance_segmentation(num_classes):
    # load a pre-trained model for classification and return
    # only the features
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
#    backbone = torchvision.models.resnet50(pretrained=True)
#    backbone = torchvision.models.resnet34(pretrained=True)
#    backbone = torch.nn.Sequential(*(list(backbone.children())[:-1]))

    # FasterRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 1280
#    backbone.out_channels = 512
    
    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    # 每个格生成15个anchor，rpn head 就有一部分用来分类（15通道输出），一部分用来回归（15*4通道输出）
    """Base anchor generator.
    The job of the anchor generator is to create (or load) a collection
    of bounding boxes to be used as anchors.
    Generated anchors are assumed to match some convolutional grid or list of grid
    shapes.  For example, we might want to generate anchors matching an 8x8
    feature map and a 4x4 feature map.  If we place 3 anchors per grid location
    on the first feature map and 6 anchors per grid location on the second feature
    map, then 3*8*8 + 6*4*4 = 288 anchors are generated in total.
    To support fully convolutional settings, feature map shapes are passed
    dynamically at generation time.  The number of anchors to place at each location
    is static --- implementations of AnchorGenerator must always be able return
    the number of anchors that it uses per location for each feature map.
    """   
    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0], output_size=7, sampling_ratio=2)

    """
    Multi-scale RoIAlign pooling, which is useful for detection with or without FPN.
    It infers the scale of the pooling via the heuristics present in the FPN paper.
    Arguments:
        featmap_names (List[str]): the names of the feature maps that will be used for the pooling.
        output_size (List[Tuple[int, int]] or List[int]): output size for the pooled region
        sampling_ratio (int): sampling ratio for ROIAlign
    Examples::
        from collections import OrderedDict
        m = torchvision.ops.MultiScaleRoIAlign(featmap_names=['feat1', 'feat3'], output_size=3, sampling_ratio=2)
        i = OrderedDict()
        # 不同尺度的feature map默认channel数是一样的
        i['feat1'] = torch.rand(1, 5, 64, 64)
        i['feat2'] = torch.rand(1, 5, 32, 32)  # this feature won't be used in the pooling
        i['feat3'] = torch.rand(1, 5, 16, 16)
        # create some random bounding boxes
        boxes = torch.rand(6, 4) * 256; boxes[:, 2:] += boxes[:, :2]
        # original image size, before computing the feature maps
        image_sizes = [(512, 512)] # 输入图像的size, boxes的坐标值是与之对应的,预测出的左边也是按这个来的
        
        output = m(i, [boxes], image_sizes)
        print(output.shape) # torch.Size([6, 5, 3, 3])
        # 
    """
    

    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
    from torchvision.models.detection.rpn import RPNHead

    return model

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

    
if __name__ == "__main__":
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
    dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))
#    1/0
    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=2, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")    
## test
#root = 'PennFudanPed'
#imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
#masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
#idx = 1
#image = Image.open(os.path.join('PennFudanPed', "PNGImages", imgs[idx]))
#mask = Image.open(os.path.join('PennFudanPed', "PedMasks", masks[idx]))
## mask 是一个单通道图像，0代表background，1,2,3代表不同的object，一共有np.max(mask)个objects