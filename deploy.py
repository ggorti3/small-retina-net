import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.ops import nms
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import os
from data import get_coco_generator, get_voc_generator, INV_COCO_DICT, INV_VOC_DICT
from model import SmallRetinaNet, refit_SSD


def display(model, img_batch, label_batch):
    np_imgs = []
    for k in range(img_batch.shape[0]):
        np_img = img_batch[k].detach().cpu().numpy()
        np_imgs.append(np.stack([np_img[0], np_img[1], np_img[2]], axis=2))
    model.eval()
    softmax = nn.Softmax(dim=1)
    out = model.forward(img_batch)
    for j in range(img_batch.shape[0]):
        label_array = label_batch[j]
        flat_out = out[j]
        class_scores = flat_out[:, :model.num_classes]
        class_scores = softmax(class_scores)
        class_scores, class_labels = torch.max(class_scores, 1)
        index_helper = torch.arange(0, class_labels.shape[0], device=model.device, dtype=torch.long)
        valid_out_mask = class_labels > 0.4
        valid_out_indices = torch.masked_select(index_helper, valid_out_mask)

        raw_boxes = flat_out[:, model.num_classes:]
        reparam_boxes = torch.stack([
            model.wa_vec * raw_boxes[:, 0] + model.xa_vec,
            model.ha_vec * raw_boxes[:, 1] + model.ya_vec,
            model.wa_vec * torch.exp(raw_boxes[:, 2]),
            model.ha_vec * torch.exp(raw_boxes[:, 3])
        ], dim=1)

        reparam_boxes = torch.stack([
            reparam_boxes[:, 0] - 0.5 * reparam_boxes[:, 2],
            reparam_boxes[:, 1] - 0.5 * reparam_boxes[:, 3],
            reparam_boxes[:, 0] + 0.5 * reparam_boxes[:, 2],
            reparam_boxes[:, 1] + 0.5 * reparam_boxes[:, 3]
        ], dim=1)

        detected_boxes = reparam_boxes[valid_out_indices]
        detected_scores = class_scores[valid_out_indices]
        detected_labels = class_labels[valid_out_indices]

        good_box_indices = nms(detected_boxes, detected_scores, iou_threshold=0.5)
        detected_boxes = detected_boxes[good_box_indices]
        detected_scores = detected_scores[good_box_indices]
        detected_labels = detected_labels[good_box_indices]
        
        fig, ax = plt.subplots(1)
        ax.imshow(np_imgs[j])
        print(detected_boxes.shape[0])
        for n in range(detected_boxes.shape[0]):
            x1 = detected_boxes[n, 0]
            y1 = detected_boxes[n, 1]
            x2 = detected_boxes[n, 2]
            y2 = detected_boxes[n, 3]
            rect_corner = x1, y1
            rect = patches.Rectangle(rect_corner, x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            #ax.text(x1, y1, "{}, {:.3f}".format(INV_COCO_DICT[int(detected_labels[n])], float(detected_scores[n])), color='r')
            ax.text(x1, y1, "{}, {:.3f}".format(INV_VOC_DICT[int(detected_labels[n])], float(detected_scores[n])), color='r')
            
        for n in range(label_array.shape[0]):
            xt = label_array[n, 1]
            yt = label_array[n, 2]
            wt = label_array[n, 3]
            ht = label_array[n, 4]
            trect_corner = xt - 0.5 * wt, yt - 0.5 * ht
            trect = patches.Rectangle(trect_corner, wt, ht, linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(trect)
        plt.show()

def test(model, img_path):
    
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor()
    ])
    
    pil_image = Image.open(img_path)#.rotate(270)
    img_batch = transform(pil_image).unsqueeze(0)
    img_batch = img_batch.to(model.device)
    
    np_imgs = []
    for k in range(img_batch.shape[0]):
        np_img = img_batch[k].detach().cpu().numpy()
        np_imgs.append(np.stack([np_img[0], np_img[1], np_img[2]], axis=2))
        
        
        
    model.eval()
    softmax = nn.Softmax(dim=1)
    out = model.forward(img_batch)
    for j in range(img_batch.shape[0]):
        flat_out = out[j]
        class_scores = flat_out[:, :model.num_classes]
        class_scores = softmax(class_scores)
        class_scores, class_labels = torch.max(class_scores, 1)
        index_helper = torch.arange(0, class_labels.shape[0], device=model.device, dtype=torch.long)
        valid_out_mask = class_labels > 0.4
        valid_out_indices = torch.masked_select(index_helper, valid_out_mask)

        raw_boxes = flat_out[:, model.num_classes:]
        reparam_boxes = torch.stack([
            model.wa_vec * raw_boxes[:, 0] + model.xa_vec,
            model.ha_vec * raw_boxes[:, 1] + model.ya_vec,
            model.wa_vec * torch.exp(raw_boxes[:, 2]),
            model.ha_vec * torch.exp(raw_boxes[:, 3])
        ], dim=1)

        reparam_boxes = torch.stack([
            reparam_boxes[:, 0] - 0.5 * reparam_boxes[:, 2],
            reparam_boxes[:, 1] - 0.5 * reparam_boxes[:, 3],
            reparam_boxes[:, 0] + 0.5 * reparam_boxes[:, 2],
            reparam_boxes[:, 1] + 0.5 * reparam_boxes[:, 3]
        ], dim=1)

        detected_boxes = reparam_boxes[valid_out_indices]
        detected_scores = class_scores[valid_out_indices]
        detected_labels = class_labels[valid_out_indices]

        good_box_indices = nms(detected_boxes, detected_scores, iou_threshold=0.5)
        detected_boxes = detected_boxes[good_box_indices]
        detected_scores = detected_scores[good_box_indices]
        detected_labels = detected_labels[good_box_indices]
        
        fig, ax = plt.subplots(1)
        ax.imshow(np_imgs[j])
        print(detected_boxes.shape[0])
        for n in range(detected_boxes.shape[0]):
            x1 = detected_boxes[n, 0]
            y1 = detected_boxes[n, 1]
            x2 = detected_boxes[n, 2]
            y2 = detected_boxes[n, 3]
            rect_corner = x1, y1
            rect = patches.Rectangle(rect_corner, x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            #ax.text(x1, y1, "{}, {:.3f}".format(INV_COCO_DICT[int(detected_labels[n])], float(detected_scores[n])), color='r')
            ax.text(x1, y1, "{}, {:.3f}".format(INV_VOC_DICT[int(detected_labels[n])], float(detected_scores[n])), color='r')
            
        plt.show()    
        
def deploy(model):
    
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor()
    ])
    
    img_list = []
    img_names = []
    for root, _, files in os.walk("./sample_images"):
        for name in files:
            if not name.startswith("."):
                img_names.append(name)
                img = Image.open(os.path.join(root, name))
                img = transform(img)
                img_list.append(img)
        
    
    img_batch = torch.stack(img_list).to(model.device)
    
    
    np_imgs = []
    for k in range(img_batch.shape[0]):
        np_img = img_batch[k].detach().cpu().numpy()
        np_imgs.append(np.stack([np_img[0], np_img[1], np_img[2]], axis=2))
        
        
        
    model.eval()
    softmax = nn.Softmax(dim=1)
    out = model.forward(img_batch)
    for j in range(img_batch.shape[0]):
        flat_out = out[j]
        class_scores = flat_out[:, :model.num_classes]
        class_scores = softmax(class_scores)
        class_scores, class_labels = torch.max(class_scores, 1)
        index_helper = torch.arange(0, class_labels.shape[0], device=model.device, dtype=torch.long)
        valid_out_mask = class_labels > 0.4
        valid_out_indices = torch.masked_select(index_helper, valid_out_mask)

        raw_boxes = flat_out[:, model.num_classes:]
        reparam_boxes = torch.stack([
            model.wa_vec * raw_boxes[:, 0] + model.xa_vec,
            model.ha_vec * raw_boxes[:, 1] + model.ya_vec,
            model.wa_vec * torch.exp(raw_boxes[:, 2]),
            model.ha_vec * torch.exp(raw_boxes[:, 3])
        ], dim=1)

        reparam_boxes = torch.stack([
            reparam_boxes[:, 0] - 0.5 * reparam_boxes[:, 2],
            reparam_boxes[:, 1] - 0.5 * reparam_boxes[:, 3],
            reparam_boxes[:, 0] + 0.5 * reparam_boxes[:, 2],
            reparam_boxes[:, 1] + 0.5 * reparam_boxes[:, 3]
        ], dim=1)

        detected_boxes = reparam_boxes[valid_out_indices]
        detected_scores = class_scores[valid_out_indices]
        detected_labels = class_labels[valid_out_indices]

        good_box_indices = nms(detected_boxes, detected_scores, iou_threshold=0.5)
        detected_boxes = detected_boxes[good_box_indices]
        detected_scores = detected_scores[good_box_indices]
        detected_labels = detected_labels[good_box_indices]
        
        fig, ax = plt.subplots(1)
        ax.imshow(np_imgs[j])
        print(detected_boxes.shape[0])
        for n in range(detected_boxes.shape[0]):
            x1 = detected_boxes[n, 0]
            y1 = detected_boxes[n, 1]
            x2 = detected_boxes[n, 2]
            y2 = detected_boxes[n, 3]
            rect_corner = x1, y1
            rect = patches.Rectangle(rect_corner, x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            #ax.text(x1, y1, "{}, {:.3f}".format(INV_COCO_DICT[int(detected_labels[n])], float(detected_scores[n])), color='r')
            ax.text(x1, y1, "{}, {:.3f}".format(INV_VOC_DICT[int(detected_labels[n])], float(detected_scores[n])), color='r')
        
        plt.savefig("./sample_detections/" + img_names[j])
            
        #plt.show()    
        
        
if __name__ == "__main__":
    DEVICE = torch.device("cpu")
    net = SmallRetinaNet(21, DEVICE)
    net.load_state_dict(torch.load("models/small_retinanet_pascalvoc_1.pt", map_location=DEVICE))
    #test(net, "./sample_images/IMG_8171.JPG")
    deploy(net)




