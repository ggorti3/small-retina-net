import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.ops import nms
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from data import get_coco_generator, get_voc_generator, INV_COCO_DICT, INV_VOC_DICT
from model import SmallRetinaNet, refit_SSD

def train(model, data_generator, lr=1e-5, epochs=6):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr)
    for e in range(epochs):
        cum_pos_class_loss = 0
        cum_neg_class_loss = 0
        cum_box_loss = 0
        count = 0
        for i, (img_batch, label_batch) in enumerate(data_generator):
            out = net.forward(img_batch)
            loss, pos_class_loss, neg_class_loss, box_loss = net.loss(out, label_batch)
            with torch.no_grad():
                cum_pos_class_loss += pos_class_loss
                cum_neg_class_loss += neg_class_loss
                cum_box_loss += box_loss
                count += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 50 == 0 and i != 0:
                print("    Iteration {}\n        pos_class_loss: {}\n        neg_class_loss: {}\n        box_loss: {}".format(i, cum_pos_class_loss/count, cum_neg_class_loss/count, cum_box_loss/count))
                cum_pos_class_loss = 0
                cum_neg_class_loss = 0
                cum_box_loss = 0
                count = 0
        print("Epoch {} complete".format(e))
        
        

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    coco_train_generator = get_coco_generator(img_path='/your_coco_path_here/images/train2017',
                                                ann_path='/your_coco_path_here/annotations/instances_train2017.json',
                                                batch_size=8,
                                                device=DEVICE,
                                                shuffle=True,
                                                isTrain=True)

    #voc_train_generator = get_voc_generator('your_path_here/pascalvoc-2012', "2012", "val", 3, device=DEVICE, shuffle=True, isTrain=True)

    
    net = SmallRetinaNet(91, DEVICE)
    
    train(net, coco_train_generator, lr=1e-5, epochs=3)







