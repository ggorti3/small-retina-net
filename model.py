import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import log

from resnet import ResNet34_fmap, ResNet18_fmap

class SmallRetinaNet(nn.Module):
    """
    Object detection network modeled after RetinaNet. Uses a resnet 34 backbone that feeds into a feature pyramid network,
    producing feature maps of different scales. Each feature map has a corresponding head that predicts boxes and classes.
    
    Only accepts RGB images transformed to size 300x300
    """
    
    def __init__(self, num_classes, device,
                         aspect_ratios = [1/3, 1/2, 1, 2, 3], 
                         scales = (0.1, 0.9), 
                         stride=8):
        """
        Initializes attributes such as hyperparameters and layers.
        
        :params:
        
        num_classes: the number of classes, including background class
        aspect_ratios: List containing different aspect ratios to describe anchor boxes
        stride: cumulative stride of the network used to produce the shared feature map
        """
        
        super(SmallRetinaNet, self).__init__()
        
        # set class attributes for important parameters
        
        self.k = 6
        self.show = False
        self.num_anchors = len(aspect_ratios) * self.k
        self.num_classes = num_classes
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.stride = stride
        self.device= device
        
        # create layers
        
        self.relu = nn.ReLU(inplace=True)
        
        self.resnet = ResNet18_fmap()

        self.pconv1 = nn.ConvTranspose2d(512, 512, stride=2, kernel_size=2)
        self.pconv2 = nn.ConvTranspose2d(512, 512, stride=2, kernel_size=3, padding=1)
        self.cconv1 = nn.Conv2d(128, 512, kernel_size=1)
        self.cconv2 = nn.Conv2d(256, 512, kernel_size=1)
        
        
        self.bn0 = nn.BatchNorm2d(512)
        self.bn1 = nn.BatchNorm2d(512)
        self.head0 = nn.Conv2d(512, len(aspect_ratios) * (num_classes + 4), kernel_size=3, padding=1)
        self.head1 = nn.Conv2d(512, len(aspect_ratios) * (num_classes + 4), kernel_size=3, padding=1)
        self.head2 = nn.Conv2d(512, len(aspect_ratios) * (num_classes + 4), kernel_size=3, padding=1)
        
        self.conv3a = nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1)
        self.conv3b = nn.Conv2d(256, 256, kernel_size=1)
        self.bn3a = nn.BatchNorm2d(256)
        self.bn3b = nn.BatchNorm2d(256)
        self.head3 = nn.Conv2d(256, len(aspect_ratios) * (num_classes + 4), kernel_size=3, padding=1)
        
        self.conv4a = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.conv4b = nn.Conv2d(256, 256, kernel_size=1)
        self.bn4a = nn.BatchNorm2d(256)
        self.bn4b = nn.BatchNorm2d(256)
        self.head4 = nn.Conv2d(256, len(aspect_ratios) * (num_classes + 4), kernel_size=3, padding=1)
        
        self.conv5a = nn.Conv2d(256, 256, kernel_size=3)
        self.conv5b = nn.Conv2d(256, 256, kernel_size=1)
        self.bn5a = nn.BatchNorm2d(256)
        self.bn5b = nn.BatchNorm2d(256)
        self.head5 = nn.Conv2d(256, len(aspect_ratios) * (num_classes + 4), kernel_size=1)
        
        torch.nn.init.normal_(self.pconv1.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.pconv1.bias)
        torch.nn.init.normal_(self.pconv2.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.pconv2.bias)
        torch.nn.init.normal_(self.cconv1.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.cconv1.bias)
        torch.nn.init.normal_(self.cconv2.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.cconv2.bias)
        
        # initialize weight values -- the initialization of the output layer biases with an appropriate
        # "prior" is very important for focal loss
        
        pi = 0.01
        fill_val = -1 * log((1 - pi) / pi)
        
        torch.nn.init.normal_(self.head0.weight, mean=0.0, std=0.01)
        torch.nn.init.constant_(self.head0.bias, fill_val)
        with torch.no_grad():
            self.head0.bias[:len(aspect_ratios)] = 0
            self.head0.bias[num_classes * len(aspect_ratios):] = 0
        
        torch.nn.init.normal_(self.head1.weight, mean=0.0, std=0.01)
        torch.nn.init.constant_(self.head1.bias, fill_val)
        with torch.no_grad():
            self.head1.bias[:len(aspect_ratios)] = 0
            self.head1.bias[num_classes * len(aspect_ratios):] = 0
        
        torch.nn.init.normal_(self.head2.weight, mean=0.0, std=0.01)
        torch.nn.init.constant_(self.head2.bias, fill_val)
        with torch.no_grad():
            self.head2.bias[:len(aspect_ratios)] = 0
            self.head2.bias[num_classes * len(aspect_ratios):] = 0
        
        torch.nn.init.normal_(self.head3.weight, mean=0.0, std=0.01)
        torch.nn.init.constant_(self.head3.bias, fill_val)
        with torch.no_grad():
            self.head3.bias[:len(aspect_ratios)] = 0
            self.head3.bias[num_classes * len(aspect_ratios):] = 0
        
        torch.nn.init.normal_(self.head4.weight, mean=0.0, std=0.01)
        torch.nn.init.constant_(self.head4.bias, fill_val)
        with torch.no_grad():
            self.head4.bias[:len(aspect_ratios)] = 0
            self.head4.bias[num_classes * len(aspect_ratios):] = 0
        
        torch.nn.init.normal_(self.head5.weight, mean=0.0, std=0.01)
        torch.nn.init.constant_(self.head5.bias, fill_val)
        with torch.no_grad():
            self.head5.bias[:len(aspect_ratios)] = 0
            self.head5.bias[num_classes * len(aspect_ratios):] = 0
        
        torch.nn.init.normal_(self.conv3a.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.conv3a.bias)
        torch.nn.init.normal_(self.conv3b.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.conv3b.bias)
        torch.nn.init.normal_(self.conv4a.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.conv4a.bias)
        torch.nn.init.normal_(self.conv4b.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.conv4b.bias)
        torch.nn.init.normal_(self.conv5a.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.conv5a.bias)
        torch.nn.init.normal_(self.conv5b.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.conv5b.bias)
        
        # generate anchor boxes, necessary for training the network
        self.generate_anchors(38, 19, 10, 5, 3, 1)
        
        # send the model to the appropriate device
        self.to(device)
        
    def forward(self, image_batch):
        # ASSUMES SQUARE FEATURE MAPS ALWAYS (input images must be transformed to square images)
        
        if self.show:
            self.np_imgs = []
            for k in range(image_batch.shape[0]):
                np_img = image_batch[k].detach().cpu().numpy()
                self.np_imgs.append(np.stack([np_img[0], np_img[1], np_img[2]], axis=2))
        
        batch_size = image_batch.shape[0]
        
        fmap0, fmap1, fmap2 = self.resnet(image_batch)
        
        
        fmap1 = self.pconv2(fmap2) + self.cconv2(fmap1)
        self.bn1(fmap1)
        self.relu(fmap1)
        fmap0 = self.pconv1(fmap1) + self.cconv1(fmap0)
        self.bn0(fmap0)
        self.relu(fmap0)
        
        
        s0 = fmap0.shape[2]
        out0 = self.head0(fmap0).view(batch_size, self.num_classes + 4, s0**2 * len(self.aspect_ratios))
        out0 = torch.transpose(out0, 1, 2)
        
        s1 = fmap1.shape[2]
        out1 = self.head1(fmap1).view(batch_size, self.num_classes + 4, s1**2 * len(self.aspect_ratios))
        out1 = torch.transpose(out1, 1, 2)
        
        s2 = fmap2.shape[2]
        out2 = self.head2(fmap2).view(batch_size, self.num_classes + 4, s2**2 * len(self.aspect_ratios))
        out2 = torch.transpose(out2, 1, 2)
        
        fmap = self.conv3a(fmap2)
        fmap = self.bn3a(fmap)
        self.relu(fmap)
        fmap = self.conv3b(fmap)
        fmap = self.bn3b(fmap)
        self.relu(fmap)
        s3 = fmap.shape[2]
        out3 = self.head3(fmap).view(batch_size, self.num_classes + 4, s3**2 * len(self.aspect_ratios))
        out3 = torch.transpose(out3, 1, 2)
        
        fmap = self.conv4a(fmap)
        fmap = self.bn4a(fmap)
        self.relu(fmap)
        fmap = self.conv4b(fmap)
        fmap = self.bn4b(fmap)
        self.relu(fmap)
        s4 = fmap.shape[2]
        out4 = self.head4(fmap).view(batch_size, self.num_classes + 4, s4**2 * len(self.aspect_ratios))
        out4 = torch.transpose(out4, 1, 2)
        
        fmap = self.conv5a(fmap)
        fmap = self.bn5a(fmap)
        self.relu(fmap)
        fmap = self.conv5b(fmap)
        fmap = self.bn5b(fmap)
        self.relu(fmap)
        s5 = fmap.shape[2]
        out5 = self.head5(fmap).view(batch_size, self.num_classes + 4, s5**2 * len(self.aspect_ratios))
        out5 = torch.transpose(out5, 1, 2)
        
        out_batch = torch.cat([out0, out1, out2, out3, out4, out5], dim=1)
        return out_batch
        
    def generate_anchors(self, s0, s1, s2, s3, s4, s5):
        # ASSUME INPUT SIZE IS 300 x 300
        
        aspect_array = torch.tensor(self.aspect_ratios, device=self.device)
        scale_vec = torch.arange(self.scales[0], self.scales[1] + 0.01, (self.scales[1] - self.scales[0])/self.k, device=self.device)
        
        y_vec0 = torch.arange(0, 300 - 0.001, 300 / s0, device=self.device) + (150 / s0)
        y_vec0 = torch.repeat_interleave(y_vec0, s0).repeat(aspect_array.shape[0])
        x_vec0 = (torch.arange(0, 300 - 0.001, 300 / s0, device=self.device) + (150 / s0)).repeat(s0 * aspect_array.shape[0])
        w_vec0 = scale_vec[0] * torch.sqrt(torch.repeat_interleave(aspect_array, s0**2)) * 300
        h_vec0 = scale_vec[0] / torch.sqrt(torch.repeat_interleave(aspect_array, s0**2)) * 300
        
        y_vec1 = torch.arange(0, 300 - 0.001, 300 / s1, device=self.device) + (150 / s1)
        y_vec1 = torch.repeat_interleave(y_vec1, s1).repeat(aspect_array.shape[0])
        x_vec1 = (torch.arange(0, 300 - 0.001, 300 / s1, device=self.device) + (150 / s1)).repeat(s1 * aspect_array.shape[0])
        w_vec1 = scale_vec[1] * torch.sqrt(torch.repeat_interleave(aspect_array, s1**2)) * 300
        h_vec1 = scale_vec[1] / torch.sqrt(torch.repeat_interleave(aspect_array, s1**2)) * 300
        
        y_vec2 = torch.arange(0, 300 - 0.001, 300 / s2, device=self.device) + (150 / s2)
        y_vec2 = torch.repeat_interleave(y_vec2, s2).repeat(aspect_array.shape[0])
        x_vec2 = (torch.arange(0, 300 - 0.001, 300 / s2, device=self.device) + (150 / s2)).repeat(s2 * aspect_array.shape[0])
        w_vec2 = scale_vec[2] * torch.sqrt(torch.repeat_interleave(aspect_array, s2**2)) * 300
        h_vec2 = scale_vec[2] / torch.sqrt(torch.repeat_interleave(aspect_array, s2**2)) * 300
        
        y_vec3 = torch.arange(0, 300 - 0.001, 300 / s3, device=self.device) + (150 / s3)
        y_vec3 = torch.repeat_interleave(y_vec3, s3).repeat(aspect_array.shape[0])
        x_vec3 = (torch.arange(0, 300 - 0.001, 300 / s3, device=self.device) + (150 / s3)).repeat(s3 * aspect_array.shape[0])
        w_vec3 = scale_vec[3] * torch.sqrt(torch.repeat_interleave(aspect_array, s3**2)) * 300
        h_vec3 = scale_vec[3] / torch.sqrt(torch.repeat_interleave(aspect_array, s3**2)) * 300
        
        y_vec4 = torch.arange(0, 300 - 0.001, 300 / s4, device=self.device) + (150 / s4)
        y_vec4 = torch.repeat_interleave(y_vec4, s4).repeat(aspect_array.shape[0])
        x_vec4 = (torch.arange(0, 300 - 0.001, 300 / s4, device=self.device) + (150 / s4)).repeat(s4 * aspect_array.shape[0])
        w_vec4 = scale_vec[4] * torch.sqrt(torch.repeat_interleave(aspect_array, s4**2)) * 300
        h_vec4 = scale_vec[4] / torch.sqrt(torch.repeat_interleave(aspect_array, s4**2)) * 300
        
        y_vec5 = torch.arange(0, 300 - 0.001, 300 / s5, device=self.device) + (150 / s5)
        y_vec5 = torch.repeat_interleave(y_vec5, s5).repeat(aspect_array.shape[0])
        x_vec5 = (torch.arange(0, 300 - 0.001, 300 / s5, device=self.device) + (150 / s5)).repeat(s5 * aspect_array.shape[0])
        w_vec5 = scale_vec[5] * torch.sqrt(torch.repeat_interleave(aspect_array, s5**2)) * 300
        h_vec5 = scale_vec[5] / torch.sqrt(torch.repeat_interleave(aspect_array, s5**2)) * 300
        
        self.xa_vec = torch.cat([x_vec0, x_vec1, x_vec2, x_vec3, x_vec4, x_vec5])
        self.ya_vec = torch.cat([y_vec0, y_vec1, y_vec2, y_vec3, y_vec4, y_vec5])
        self.wa_vec = torch.cat([w_vec0, w_vec1, w_vec2, w_vec3, w_vec4, w_vec5])
        self.ha_vec = torch.cat([h_vec0, h_vec1, h_vec2, h_vec3, h_vec4, h_vec5])
    
    def loss(self, out_batch, label_array_list):
        batch_size = len(label_array_list)
        # iterate through the batch
        pos_outs_list = []
        neg_outs_list = []
        reparam_pos_labels_list = []
        total_num_positives=0
        for i in range(batch_size):
            # calculate IoU array
            with torch.no_grad():
                label_array = label_array_list[i]
                if label_array.shape[0] == 0:
                    continue
                xt_array = label_array[:, 1].unsqueeze(0).repeat(self.xa_vec.shape[0], 1)
                yt_array = label_array[:, 2].unsqueeze(0).repeat(self.ya_vec.shape[0], 1)
                wt_array = label_array[:, 3].unsqueeze(0).repeat(self.wa_vec.shape[0], 1)
                ht_array = label_array[:, 4].unsqueeze(0).repeat(self.ha_vec.shape[0], 1)

                xa_array = self.xa_vec.unsqueeze(1).repeat(1, label_array.shape[0])
                ya_array = self.ya_vec.unsqueeze(1).repeat(1, label_array.shape[0])
                wa_array = self.wa_vec.unsqueeze(1).repeat(1, label_array.shape[0])
                ha_array = self.ha_vec.unsqueeze(1).repeat(1, label_array.shape[0])
                
                x_left = torch.max(xt_array - 0.5 * wt_array, xa_array - 0.5 * wa_array)
                y_top = torch.max(yt_array - 0.5 * ht_array, ya_array - 0.5 * ha_array)
                x_right = torch.min(xt_array + 0.5 * wt_array, xa_array + 0.5 * wa_array)
                y_bottom = torch.min(yt_array + 0.5 * ht_array, ya_array + 0.5 * ha_array)
                
                wi_array = x_right - x_left
                wi_array[wi_array < 0] = 0
                hi_array = y_bottom - y_top
                hi_array[hi_array < 0] = 0

                intersection_array = (wi_array * hi_array)
                union_array = wa_array * ha_array + wt_array * ht_array - intersection_array
                IoU_array = intersection_array / union_array
            
                base_mask = torch.zeros((IoU_array.shape[0], IoU_array.shape[1]), dtype=torch.bool, device=self.device)
                index_helper = torch.arange(0, IoU_array.shape[1], dtype=torch.long, device=self.device)
                nearest_box_indices = torch.argmax(IoU_array, dim=0)
                base_mask[nearest_box_indices, index_helper] = True
                positives_mask = torch.logical_or(base_mask, IoU_array >= 0.5)
                num_positives = torch.sum(positives_mask)
                total_num_positives += num_positives
            
            # select positive anchors and matching labels
            label_idx_array = torch.arange(0, IoU_array.shape[1],
                                           dtype=torch.long, device=self.device).unsqueeze(0).repeat(IoU_array.shape[0], 1)
            out_idx_array = torch.arange(0, IoU_array.shape[0],
                                         dtype=torch.long, device=self.device).unsqueeze(1).repeat(1, IoU_array.shape[1])
            pos_label_idx_vec = torch.masked_select(label_idx_array, positives_mask)
            pos_out_idx_vec = torch.masked_select(out_idx_array, positives_mask)
            
            neg_idx_helper = torch.arange(0, IoU_array.shape[0], dtype=torch.long, device=self.device)
            neg_out_mask = (torch.sum(positives_mask, dim=1) == 0)
            neg_out_idx_vec = torch.masked_select(neg_idx_helper, neg_out_mask)
            
            out_array = out_batch[i]
            pos_outs = out_array[pos_out_idx_vec]
            neg_outs = out_array[neg_out_idx_vec][:, :self.num_classes]
            
            if self.show:
                print("guesses")
                for elem in torch.argmax(pos_outs[:, :self.num_classes], 1):
                    print("    {}".format(INV_COCO_DICT[int(elem)]))
                print("truths")
                for elem in label_array[pos_label_idx_vec][:, 0]:
                    print("    {}".format(INV_COCO_DICT[int(elem)]))
                fig, ax = plt.subplots(1)
                ax.imshow(self.np_imgs[i])
                xa_vec = self.xa_vec[pos_out_idx_vec]
                ya_vec = self.ya_vec[pos_out_idx_vec]
                wa_vec = self.wa_vec[pos_out_idx_vec]
                ha_vec = self.ha_vec[pos_out_idx_vec]
                for k in range(pos_outs.shape[0]):

                    x = wa_vec[k] * pos_outs[k, self.num_classes] + xa_vec[k]
                    y = ha_vec[k] * pos_outs[k, self.num_classes + 1] + ya_vec[k]
                    w = wa_vec[k] * torch.exp(pos_outs[k, self.num_classes + 2]) 
                    h = ha_vec[k] * torch.exp(pos_outs[k, self.num_classes + 3])
                    rect_corner = x - 0.5 * w, y - 0.5 * h
                    rect = patches.Rectangle(rect_corner, w, h, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)

                for k in range(label_array.shape[0]):
                    xt = label_array[k, 1]
                    yt = label_array[k, 2]
                    wt = label_array[k, 3]
                    ht = label_array[k, 4]
                    trect_corner = xt - 0.5 * wt, yt - 0.5 * ht
                    trect = patches.Rectangle(trect_corner, wt, ht, linewidth=1, edgecolor='g', facecolor='none')
                    ax.add_patch(trect)
                plt.show()
            
            pos_labels = label_array[pos_label_idx_vec]
            reparam_pos_labels = torch.stack([
                pos_labels[:, 0],
                (pos_labels[:, 1] - self.xa_vec[pos_out_idx_vec]) / self.wa_vec[pos_out_idx_vec],
                (pos_labels[:, 2] - self.ya_vec[pos_out_idx_vec]) / self.ha_vec[pos_out_idx_vec],
                torch.log(pos_labels[:, 3] / self.wa_vec[pos_out_idx_vec]),
                torch.log(pos_labels[:, 4] / self.ha_vec[pos_out_idx_vec])
            ], dim=1)
            
            # for outs, the first self.num_classes columns are class scores, then the last 4 columns are the boxes
            # for labels, the first column is the class label, the last 4 columns describe the box
            pos_outs_list.append(pos_outs)
            neg_outs_list.append(neg_outs)
            reparam_pos_labels_list.append(reparam_pos_labels)
            
        if pos_outs_list:
            cum_pos_outs = torch.cat(pos_outs_list, dim=0)
            cum_neg_outs = torch.cat(neg_outs_list, dim=0)
            cum_reparam_pos_labels = torch.cat(reparam_pos_labels_list, dim=0)
            pos_class_outs = cum_pos_outs[:, :self.num_classes]
            pos_class_labels = cum_reparam_pos_labels[:, 0]
            pos_box_outs = cum_pos_outs[:, self.num_classes:]
            pos_truth_boxes = cum_reparam_pos_labels[:, 1:]
            neg_class_outs = cum_neg_outs[:, :self.num_classes]
            neg_labels = torch.zeros((neg_class_outs.shape[0], ), device=self.device, dtype=torch.long)

            gamma = 2
            alpha = 0.25
            SL1Loss = nn.SmoothL1Loss()
            pos_class_loss = torch.sum(focal_loss(pos_class_outs, pos_class_labels.long(), gamma, alpha, self.device)) / pos_class_outs.shape[0]
            box_loss = SL1Loss(pos_box_outs, pos_truth_boxes)
            neg_class_loss = torch.sum(focal_loss(neg_class_outs, neg_labels, gamma, alpha, self.device)) / pos_class_outs.shape[0]
                
            beta = 1
            loss = (pos_class_loss + neg_class_loss) + beta * box_loss
            return loss, pos_class_loss, neg_class_loss, box_loss
            
        else:
            return 0, 0, 0, 0
        
# helper functions

def refit_SSD(model, new_num_classes):
    """
    Changes prediction layers of the network to work for "new_num_classes" number of classes.
    Necessary in order to train the same SSD on different datasets, such as training on coco then pascalvoc. 
    """
    aspect_ratios = model.aspect_ratios
    model.num_classes = new_num_classes
    model.head0 = nn.Conv2d(512, len(aspect_ratios) * (new_num_classes + 4), kernel_size=3, padding=1)
    model.head1 = nn.Conv2d(512, len(aspect_ratios) * (new_num_classes + 4), kernel_size=3, padding=1)
    model.head2 = nn.Conv2d(512, len(aspect_ratios) * (new_num_classes + 4), kernel_size=3, padding=1)
    model.head3 = nn.Conv2d(256, len(aspect_ratios) * (new_num_classes + 4), kernel_size=3, padding=1)
    model.head4 = nn.Conv2d(256, len(aspect_ratios) * (new_num_classes + 4), kernel_size=3, padding=1)
    model.head5 = nn.Conv2d(256, len(aspect_ratios) * (new_num_classes + 4), kernel_size=1)
    
    pi = 0.01
    fill_val = -1 * log((1 - pi) / pi)

    torch.nn.init.normal_(net.head0.weight, mean=0.0, std=0.01)
    torch.nn.init.constant_(net.head0.bias, fill_val)
    torch.nn.init.normal_(net.head1.weight, mean=0.0, std=0.01)
    torch.nn.init.constant_(net.head1.bias, fill_val)
    torch.nn.init.normal_(net.head2.weight, mean=0.0, std=0.01)
    torch.nn.init.constant_(net.head2.bias, fill_val)
    torch.nn.init.normal_(net.head3.weight, mean=0.0, std=0.01)
    torch.nn.init.constant_(net.head3.bias, fill_val)
    torch.nn.init.normal_(net.head4.weight, mean=0.0, std=0.01)
    torch.nn.init.constant_(net.head4.bias, fill_val)
    torch.nn.init.normal_(net.head5.weight, mean=0.0, std=0.01)
    torch.nn.init.constant_(net.head5.bias, fill_val)
    with torch.no_grad():
        net.head0.bias[:len(aspect_ratios)] = 0
        net.head0.bias[new_num_classes * len(aspect_ratios):] = 0
        net.head1.bias[:len(aspect_ratios)] = 0
        net.head1.bias[new_num_classes * len(aspect_ratios):] = 0
        net.head2.bias[:len(aspect_ratios)] = 0
        net.head2.bias[new_num_classes * len(aspect_ratios):] = 0
        net.head3.bias[:len(aspect_ratios)] = 0
        net.head3.bias[new_num_classes * len(aspect_ratios):] = 0
        net.head4.bias[:len(aspect_ratios)] = 0
        net.head4.bias[new_num_classes * len(aspect_ratios):] = 0
        net.head5.bias[:len(aspect_ratios)] = 0
        net.head5.bias[new_num_classes * len(aspect_ratios):] = 0
    return model

def focal_loss(out, target, gamma, alpha, device):
    """
    Focal loss implementation.
    """
    softmax = nn.Softmax(dim=1)
    out = softmax(out)
    
    mask = torch.zeros((out.shape[0], out.shape[1]), device=device, dtype=torch.bool)
    idx_helper = torch.arange(0, out.shape[0], device=device, dtype=torch.long)
    mask[idx_helper, target] = True
    
    p_array = torch.where(mask, out, (1 - out))
    p_array[p_array < 1e-38] = 1e-38
    loss_array = -1 * torch.pow(1 - p_array, gamma) * torch.log(p_array)
    alpha_array = torch.full((loss_array.shape[0], loss_array.shape[1]), alpha, device=device)
    alpha_array[0] = 1 - alpha
    loss_array *= alpha_array
    losses = torch.sum(loss_array, 1)
    
    return losses





