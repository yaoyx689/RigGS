import numpy as np
import os 
import pickle 
from argparse import ArgumentParser

def main(processed_dir, data_dir):

    TRAIN_CAMERA_ID = [1, 5, 9, 13, 17, 21] 
    
    out_dir = os.path.join(data_dir,"train/")
    
    image_dir = os.path.join(out_dir, "images/")
    mask_dir = os.path.join(out_dir, "masks/")
    
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    new_cameras = {}
    new_mesh_infos = {}

    all_cameras = {} 
    all_mesh_infos = {} 

    for cam in TRAIN_CAMERA_ID:
        path = processed_dir + "/view_" + str(cam).zfill(2) + "/"
        with open(os.path.join(path, 'cameras.pkl'), 'rb') as f: 
            cameras = pickle.load(f)
        
        all_cameras[str(cam)] = cameras 
        
        with open(os.path.join(path, 'mesh_infos.pkl'), 'rb') as f: 
            mesh_infos = pickle.load(f)
        
        all_mesh_infos[str(cam)] = mesh_infos 


    idx = 0
    for fname in all_mesh_infos[str(TRAIN_CAMERA_ID[0])]:
        cam = TRAIN_CAMERA_ID[idx]
        print(cam)
        path = processed_dir + "/view_" + str(cam).zfill(2) + "/"
        _, stridx = fname.split('_')
        idx = int(stridx)
        image_path = os.path.join(path, "images", fname + ".png") 
        out_img_path = os.path.join(image_dir, fname + ".png") 
        
        mask_path = os.path.join(path, "masks", fname + ".png") 
        out_mask_path = os.path.join(mask_dir, fname + ".png") 
        
        
        runf = "cp " + image_path + " " + out_img_path 
        os.system(runf)
        
        runf = "cp " + mask_path + " " + out_mask_path 
        os.system(runf)
        
        new_cameras[fname] = all_cameras[str(cam)][fname]
        new_mesh_infos[fname] = all_mesh_infos[str(cam)][fname]
        
        idx = (idx + 1)%len(TRAIN_CAMERA_ID)


    # write camera infos
    output_path = out_dir
    with open(os.path.join(output_path, 'cameras.pkl'), 'wb') as f:   
        pickle.dump(new_cameras, f)
        
    # write mesh infos
    with open(os.path.join(output_path, 'mesh_infos.pkl'), 'wb') as f:   
        pickle.dump(new_mesh_infos, f)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--processed_dir', type=str, default='./377/test/')
    parser.add_argument('--data_dir', type=str, default='./377/')
    args = parser.parse_args()
    
    main(args.processed_dir, args.data_dir)