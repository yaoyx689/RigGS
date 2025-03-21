# Copyright 2023 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from config import cfg
from data_utils import *
from extractor import *
from clustering import *
import os 
from skimage.morphology import thin


def get_2d_skeleton(mask):     
    thinned = thin((mask>0).astype(np.uint8)).astype(np.float32)
    thinned = thinned.astype(np.int32)
    new_thinned = thinned.copy()
    # for v in range(thinned.shape[1]):
    #     new_thinned[:,v] = thinned[:,thinned.shape[1]-1-v]
    select = new_thinned > 0
    new_thinned[select] = 255    
    return new_thinned

def extract_skeleton(masks): 
    thinned_images = [] 
    for idx in tqdm(range(0,masks.shape[0])):
        thinned_img = get_2d_skeleton(masks[idx,0])   
        thinned_images.append(thinned_img)
        
    return thinned_images 
    
    
def preprocess_data(data_dir, image_folder_name, mask_folder_name=None):        
    images = []
    
    # judge if provided the background
    provided_bg = False 
    image_folder = os.path.join(data_dir, image_folder_name)
    out_thinned_folder = os.path.join(data_dir, 'train_thinned')
    out_segment_folder = os.path.join(data_dir, 'semantic_seg')
    os.makedirs(out_thinned_folder, exist_ok=True)
    os.makedirs(out_segment_folder, exist_ok=True)
    
    files = os.listdir(image_folder)
    files.sort()
    
    images = []
    image_names = []
    masks = []
    for image_name in files:
        if image_name[-3:] == 'png' or image_name[-3:] == 'jpg':
            image = cv2.imread(os.path.join(image_folder, image_name), -1) 
            
            if image.shape[-1]==4:
                mask = np.tile(np.expand_dims(image[...,3], axis=-1), (1, 1, 3))
                masks.append((mask>0).astype(float))
            
            images.append(image)
            image_names.append(image_name)
    
    if mask_folder_name is not None and masks ==[]:
        for idx in range(len(image_names)):
            image_name = image_names[idx]
            mask = cv2.imread(os.path.join(data_dir, mask_folder_name, image_name))
            masks.append((mask>0).astype(float))
    
    if masks != []:
        provided_bg = True
    
    crop_images = []
    crop_masks = []
    for i in range(len(images)):
        
        img = images[i][:,:,2::-1]/255.
        left, top, width, height = 0, 0, img.shape[1], img.shape[0]
        bb = process_bbox(left, top, width, height,1,1)
        img = crop_and_resize(img.copy(), bb, cfg.crop_size, rgb=True)
        
        if provided_bg:
            mask = masks[i][:,:,2::-1]
            mask = crop_and_resize(mask.copy(), bb, cfg.crop_size, rgb=True)
            crop_masks.append(mask)
        
        crop_images.append(img)
    
    input_size = cfg.input_size
    print("Extracting DINO features...")
    extractor = VitExtractor(cfg.dino_model, cfg.device)
    if provided_bg is False:
        with torch.no_grad():
            features, saliency = extractor.extract_feat_hr(crop_images)

        print("Clustering DINO features...")
        masks_vit, part_centers, centroids = cluster_features(features, saliency, crop_images)
        masks_fg = [F.interpolate((m>0).float(), cfg.input_size, mode='nearest') for m in masks_vit]
    else:
        crop_images = [F.interpolate(img, cfg.input_size, mode='bilinear', align_corners=False) for img in crop_images]
        crop_masks = [F.interpolate(img, cfg.input_size, mode='bilinear', align_corners=False) for img in crop_masks]
        masks_fg = [torch.sum(mask**2, dim=1, keepdim=True)>0 for mask in crop_masks]
    
    print("Extracting low-res DINO features...")
    crop_images = [F.interpolate(img, cfg.input_size, mode='bilinear', align_corners=False) for img in crop_images]
    # print('crop_images.shape = ', crop_images.shape)
    with torch.no_grad():
        features, saliency = extractor.extract_feat(crop_images)

    print("Clustering low-res features...")
    masks_vit, part_centers, centroids = cluster_features(features, saliency, crop_images, masks_fg)
    
    input_size = (width, height)
    print("Collecting input batch...")
    inputs = {}
    inputs['images'] = F.interpolate(torch.cat(crop_images, 0), input_size, mode='bilinear', align_corners=False)
    inputs['masks'] = F.interpolate(torch.cat(masks_vit, 0), input_size, mode='nearest')
    inputs['masks_lr'] = F.interpolate(torch.cat(masks_vit, 0), (cfg.hw,cfg.hw), mode='nearest')
    inputs['part_cent'] = torch.stack(part_centers, 0)
        
    # Reduce feature dimension
    d = extractor.get_embedding_dim()
    feat_img = torch.stack([k.permute(1,0).view(d,cfg.hw,cfg.hw) for k in features], 0)
    feat_sal = feat_img.permute(0,2,3,1)[inputs['masks_lr'][:,0]>0]
    _, _, V = torch.pca_lowrank(feat_sal, q=cfg.d_feat, center=True, niter=2)
    feat_img = feat_img.permute(1,0,2,3).reshape(d,-1).permute(1,0)
    inputs['feat_img'] = torch.matmul(feat_img, V).permute(1,0).view(cfg.d_feat,-1,cfg.hw,cfg.hw).permute(1,0,2,3)
    inputs['feat_part'] = torch.matmul(centroids, V)
    
    masks = inputs['masks'].cpu().numpy()
    thinned_images = extract_skeleton(masks) 
    
    for i in range(len(images)):
        mask = inputs['masks'][i].permute(1,2,0).cpu().numpy()
        cmask = part_mask_to_image(mask[:,:,0], part_colors)        
        img = np.concatenate([cmask[:,:,2::-1], cmask[:,:,3:]], 2)
        cv2.imwrite(osp.join(out_segment_folder, image_names[i][:-4] + '_seg.png'), img)
        cv2.imwrite(osp.join(out_thinned_folder, image_names[i][:-4] + '_thinned.png'), thinned_images[i])
        np.save(osp.join(out_segment_folder, image_names[i][:-4] + '_seg.npy'), masks[i])
    
    return len(images)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/jumpingjacks/')
    parser.add_argument('--image_folder', type=str, default='train')
    parser.add_argument('--inst', type=bool, default=False, dest='opt_instance')
    parser.add_argument('--idx', type=int, default=0, dest='instance_idx')
    parser.add_argument('--cls', type=str, default='', dest='cls')
    parser.add_argument('--mask_folder', type=str, default=None)
    args = parser.parse_args()
    cfg.set_args(args)
    
    num_imgs = preprocess_data(args.data_dir, args.image_folder, args.mask_folder)
    print("Finished preprocessing %d images." % num_imgs)
    