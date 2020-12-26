import numpy as np
import torch
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import random

from pprint import pprint

# DEVICE = torch.device("cuda:0")
# #DEVICE = torch.device("cpu")

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'trafficlight', 'firehydrant', 'N/A', 'stopsign',
    'parkingmeter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sportsball',
    'kite', 'baseballbat', 'baseballglove', 'skateboard', 'surfboard', 'tennisracket',
    'bottle', 'N/A', 'wineglass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hotdog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'pottedplant', 'bed', 'N/A', 'diningtable',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cellphone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddybear', 'hairdrier', 'toothbrush'
]

INV_COCO_DICT = {
    0: "__background",
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}


VOC_DICT = {
    "__background__": 0,
    "person": 1,
    "bird": 2,
    "cat": 3,
    "cow": 4,
    "dog": 5,
    "horse": 6,
    "sheep": 7,
    "aeroplane": 8,
    "bicycle": 9,
    "boat": 10,
    "bus": 11,
    "car": 12,
    "motorbike": 13,
    "train": 14,
    "bottle": 15,
    "chair": 16,
    "diningtable": 17,
    "pottedplant": 18,
    "sofa": 19,
    "tvmonitor": 20
}

INV_VOC_DICT = {
    0:"__background__",
    1:"person",
    2:"bird",
    3:"cat",
    4:"cow",
    5:"dog",
    6:"horse",
    7:"sheep",
    8:"aeroplane",
    9:"bicycle",
    10:"boat",
    11:"bus",
    12:"car",
    13:"motorbike",
    14:"train",
    15:"bottle",
    16:"chair",
    17:"diningtable",
    18:"pottedplant",
    19:"sofa",
    20:"tvmonitor"
}



def get_voc_generator(path, year, image_set, batch_size, device, shuffle=True):
    
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor()
    ])
    
    def detection_collate_fn(sample_list):
        img_batch = []
        target_batch = []
        for (x, y) in sample_list:
            x_scale = 300 / x.size[0]
            y_scale = 300 / x.size[1]
            img_batch.append(transform(x))
            y = torch.stack([y[:, 0], y[:, 1] * x_scale, y[:, 2] * y_scale, y[:, 3] * x_scale,y[:, 4] * y_scale], dim=1)
            target_batch.append(y)
        img_batch = torch.stack(img_batch).to(device)
        return img_batch, target_batch
    
    def voc_target_transform(y):
        truths = []
        for elem in y["annotation"]["object"]:
            truth = torch.zeros((5, ), device=device, dtype=torch.float)
            truth[0] = VOC_DICT[elem["name"]]
            truth[1] = 0.5 * (int(elem["bndbox"]["xmax"]) + int(elem["bndbox"]["xmin"]))
            truth[2] = 0.5 * (int(elem["bndbox"]["ymax"]) + int(elem["bndbox"]["ymin"]))
            truth[3] = int(elem["bndbox"]["xmax"]) - int(elem["bndbox"]["xmin"])
            truth[4] = int(elem["bndbox"]["ymax"]) - int(elem["bndbox"]["ymin"])
            truths.append(truth)
        if truths:
            truth_array = torch.stack(truths, dim=0)
        else:
            truth_array = torch.zeros((0, 5), device=device, dtype=torch.float)

        return truth_array
    
    voc_data = dset.VOCDetection(root = path,
                                  year = year,
                                  image_set = image_set,
                                  target_transform=voc_target_transform)
    voc_generator = DataLoader(voc_data, batch_size=batch_size, collate_fn=detection_collate_fn, shuffle=shuffle, drop_last=True)
    return voc_generator

# def voc_target_transform(y):
#     truths = []
#     for elem in y["annotation"]["object"]:
#         truth = torch.zeros((5, ), device=device, dtype=torch.float)
#         truth[0] = VOC_DICT[elem["name"]]
#         truth[1] = 0.5 * (int(elem["bndbox"]["xmax"]) + int(elem["bndbox"]["xmin"]))
#         truth[2] = 0.5 * (int(elem["bndbox"]["ymax"]) + int(elem["bndbox"]["ymin"]))
#         truth[3] = int(elem["bndbox"]["xmax"]) - int(elem["bndbox"]["xmin"])
#         truth[4] = int(elem["bndbox"]["ymax"]) - int(elem["bndbox"]["ymin"])
#         truths.append(truth)
#     if truths:
#         truth_array = torch.stack(truths, dim=0)
#     else:
#         truth_array = torch.zeros((0, 5), device=device, dtype=torch.float)

#     return truth_array

def get_coco_generator(img_path, ann_path, batch_size, device, shuffle=True, isTrain=True):
    transform = transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor()])
    def detection_collate_fn(sample_list):
        img_batch = []
        target_batch = []
        for (x, y) in sample_list:
            if not isTrain:
                x_scale = 300 / x.size[0]
                y_scale = 300 / x.size[1]
                x = transform(x)
                y = torch.stack([y[:, 0], y[:, 1] * x_scale, y[:, 2] * y_scale, y[:, 3] * x_scale,y[:, 4] * y_scale], dim=1)
            else:
                x, y = augment(x, y)
            
            img_batch.append(x)
            target_batch.append(y)
            
        img_batch = torch.stack(img_batch).to(device)
        return img_batch, target_batch
    
    def coco_target_transform(y):
        truths = []
        for elem in y:
            box = elem["bbox"]
            truth = torch.zeros((5, ), device=device, dtype=torch.float)
            truth[0] = elem["category_id"]
            truth[1] = box[0] + 0.5 * box[2]
            truth[2] = box[1] + 0.5 * box[3]
            truth[3] = box[2]
            truth[4] = box[3]
            truths.append(truth)
        if truths:
            truth_array = torch.stack(truths, dim=0)
        else:
            truth_array = torch.zeros((0, 5), device=device, dtype=torch.float)

        return truth_array
    
    coco_data = dset.CocoDetection(root = img_path,
                                   annFile = ann_path,
                                   target_transform=coco_target_transform)
    coco_generator = DataLoader(coco_data, batch_size=batch_size, collate_fn=detection_collate_fn, shuffle=shuffle, drop_last=True)
    return coco_generator


# def coco_target_transform(y):
#     truths = []
#     for elem in y:
#         box = elem["bbox"]
#         truth = torch.zeros((5, ), device=device, dtype=torch.float)
#         truth[0] = elem["category_id"]
#         truth[1] = box[0] + 0.5 * box[2]
#         truth[2] = box[1] + 0.5 * box[3]
#         truth[3] = box[2]
#         truth[4] = box[3]
#         truths.append(truth)
#     if truths:
#         truth_array = torch.stack(truths, dim=0)
#     else:
#         truth_array = torch.zeros((0, 5), device=device, dtype=torch.float)

#     return truth_array

def augment(x, y):
    # assume x and y are not resized yet
    
    # randomly choose a size between certain bounds, maybe between 150 and 450
    # 3 sided die toss to choose crop, shrink, or unchanged
    # select a random size between 150 and 300
    # if 0, crop img to the random size at a random spot
    # if 1, resize img to random size then place at random spot in blank canvas
    # if 2, dont change, just scale to 300
    # coin toss to flip horizontally
    
    die = random.randrange(0, 3)
    
    if die == 0:
        transform = transforms.Compose([
        transforms.Resize((300, 300)),
            transforms.ToTensor()
        ])
        # crop and resize
        crop_width = random.randrange(int(0.5 * x.size[0]), x.size[0])
        crop_height = random.randrange(int(0.5 * x.size[1]), x.size[1])
        x1 = random.randrange(0, x.size[0] - crop_width + 1)
        y1 = random.randrange(0, x.size[1] - crop_height + 1)
        x2 = x1 + crop_width
        y2 = y1 + crop_height
        crop = x.crop((x1, y1, x2, y2))
        x = transform(crop)
        
        x_scale = 300 / crop_width
        y_scale = 300 / crop_height
        
        y = torch.stack([
            y[:, 0],
            x_scale * (y[:, 1] - x1),
            y_scale * (y[:, 2] - y1),
            x_scale * y[:, 3],
            y_scale * y[:, 4],
        ], dim=1)
        
        # throw out centers that arent in the pic anymore
        mask1 = y[:, 1] > 0
        mask2 = y[:, 1] < 300
        mask3 = y[:, 2] > 0
        mask4 = y[:, 2] < 300
        mask = torch.logical_and(torch.logical_and(mask1, mask2), torch.logical_and(mask3, mask4))
        index_helper = torch.arange(0, mask1.shape[0])
        indices = torch.masked_select(index_helper, mask)
        y = y[indices]
        
        # clip widths and heights to be within bounds
        x1_vec = y[:, 1] - 0.5 * y[:, 3]
        y1_vec = y[:, 2] - 0.5 * y[:, 4]
        x2_vec = y[:, 1] + 0.5 * y[:, 3]
        y2_vec = y[:, 2] + 0.5 * y[:, 4]
        
        x1_vec[x1_vec < 0] = 0
        y1_vec[y1_vec < 0] = 0
        x2_vec[x2_vec > 300] = 300
        y2_vec[y2_vec > 300] = 300
        
        y = torch.stack([
            y[:, 0],
            0.5 * (x1_vec + x2_vec),
            0.5 * (y1_vec + y2_vec),
            x2_vec - x1_vec,
            y2_vec - y1_vec,
        ], dim=1)
    elif die == 1:
        # shrink and place
        width = random.randrange(150, 300)
        height = random.randrange(150, 300)
        x_scale = width / x.size[0]
        y_scale = height / x.size[1]
        
        transform1 = transforms.Resize((height, width))
        x = transform1(x)
        
        x1 = random.randrange(0, 300 - width + 1)
        y1 = random.randrange(0, 300 - height + 1)
        
        transform2 = transforms.Compose([
            transforms.Pad((x1, y1, 300 - (x.size[0] + x1), 300 - (x.size[1] + y1))),
            transforms.ToTensor()
        ])
        x = transform2(x)
        
        y = torch.stack([
            y[:, 0],
            x_scale * y[:, 1] + x1,
            y_scale * y[:, 2] + y1,
            x_scale * y[:, 3],
            y_scale * y[:, 4],
        ], dim=1)
    else:
        transform = transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor()])
        x_scale = 300 / x.size[0]
        y_scale = 300 / x.size[1]
        x = transform(x)
        y = torch.stack([y[:, 0], y[:, 1] * x_scale, y[:, 2] * y_scale, y[:, 3] * x_scale,y[:, 4] * y_scale], dim=1)
    
    if random.randrange(0, 2) == 1:
        x = torch.flip(x, (2,))
        y = torch.stack([y[:, 0], 300 - y[:, 1], y[:, 2], y[:, 3], y[:, 4]], dim=1)
    
    return x, y
        