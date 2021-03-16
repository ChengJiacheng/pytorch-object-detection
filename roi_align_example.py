import torch
import torchvision

# create 10 random boxes
boxes = torch.rand(10, 4) * 100
# they need to be in [x0, y0, x1, y1] format
# x0 < x1, y0 < y1
boxes[:, 2:] += boxes[:, :2]
# create a random image
image = torch.rand(1, 3, 200, 200)
# extract regions in `image` defined in `boxes`, rescaling
# them to have a size of 3x3
pooled_regions = torchvision.ops.roi_align(image, [boxes], output_size=(3, 3))
# check the size
print(pooled_regions.shape)
# torch.Size([10, 3, 3, 3])

# or compute the intersection over union between
# all pairs of boxes
print(torchvision.ops.box_iou(boxes, boxes).shape)
# torch.Size([10, 10])