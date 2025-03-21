import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
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
    os.makedirs(out_thinned_folder, exist_ok=True)
    
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
    
    if not provided_bg:
        print('Error: No mask provided')
        return 
    
    masks = [(mask**2).sum(-1)>0 for mask in masks]
    masks = np.stack(masks, axis=0)
    masks = np.expand_dims(masks, axis=1)
    
    thinned_images = extract_skeleton(masks) 
    
    
    
    for i in range(len(thinned_images)):
        cv2.imwrite(osp.join(out_thinned_folder, image_names[i][:-4] + '_thinned.png'), thinned_images[i])
    
    return len(thinned_images)
    



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/jumpingjacks/')
    parser.add_argument('--image_folder', type=str, default='train')
    parser.add_argument('--mask_folder', type=str, default=None)
    args = parser.parse_args()
    
    num_imgs = preprocess_data(args.data_dir, args.image_folder, args.mask_folder)
    print("Finished preprocessing %d images." % num_imgs)
    