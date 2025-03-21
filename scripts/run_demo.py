import os 

def main(data_dir, out_dir):
    
    ###### Settings #####
    train_stage1 = True  # Whether to train the initialization (Sec. 3.1), keep it True when starting training.
    render_stage1 = False # Whether to render the results of initialization (Sec. 3.1), keep it True when checking results.
    train_stage2 = True  # Whether to train skeleton-driven dynamic model (Sec. 3.3). 
    render_stage2 = True  # Whether to render the results of skeleton-driven dynamic model (Sec. 3.3).
    render_stage2_interpolation = False # Whether to render the interpolation results of skeleton-driven dynamic model (Sec. 3.3). 
    render_stage2_random_motion = False # Whether to render the a continuous sequence of random actions. 
    view_id="0" # The viewpoint subscript corresponding to the training image is used to render the results of interpolation and random actions at that viewpoint.
    use_gui = False 
    #####################
    nnode="512"
    
    pretrain_model_path= os.path.join(out_dir, 'stage1')
    outname=out_dir 

    runf = "python train_gui.py --source_path " + data_dir + " --model_path " + pretrain_model_path + " --deform_type node --node_num " + nnode + " --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --iterations_node_rendering 10000 --data_type ZJU --use_isotropic_gs"
    if train_stage1 and not use_gui:
        os.system(runf) 
    
    
    runf = "python render.py --source_path " + data_dir + " --model_path " + pretrain_model_path + " --deform_type node --node_num " + nnode + " --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --use_isotropic_gs"
    
    if render_stage1 and not use_gui:
        os.system(runf) 
        
    knn=-1 
    # "CUDA_VISIBLE_DEVICES=0"
    runf = "python run_train_rig.py --source_path " + data_dir + " --model_path " + outname + " --pretrain_model_path " + pretrain_model_path + " --deform_type node --node_num 512 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --iterations 100000 --use_isotropic_gs --use_skinning_weight_mlp --use_template_offsets"
    if use_gui:
        runf = runf + " --gui"
    
    if train_stage2:
        os.system(runf) 
    
    runf = "python render_rig.py --source_path " + data_dir + " --model_path " + outname + " --pretrain_model_path " + pretrain_model_path + " --deform_type node --node_num 512 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --skeleton_weight_knn " + str(knn) 
    
    if render_stage2 and not use_gui:
        os.system(runf) 
    
    runf = "python render_rig.py --source_path " + data_dir + " --model_path " + outname + " --pretrain_model_path " + pretrain_model_path + " --deform_type node --node_num 512 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --skeleton_weight_knn " + str(knn) + " --mode time --view_id " + view_id + " --skip_test" 
    
    if render_stage2_interpolation and not use_gui:
        os.system(runf) 
    
    runf = "python render_rig.py --source_path " + data_dir + " --model_path " + outname + " --pretrain_model_path " + pretrain_model_path + " --deform_type node --node_num 512 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --skeleton_weight_knn " + str(knn) + " --mode motion --view_id " + view_id + " --skip_test" 
    
    if render_stage2_random_motion and not use_gui:
        os.system(runf) 


import argparse 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='example')
    parser.add_argument('data_dir', type=str, help='input data path')
    parser.add_argument('out_dir', type=str, help='output path')
    args = parser.parse_args()
    
    main(args.data_dir, args.out_dir)