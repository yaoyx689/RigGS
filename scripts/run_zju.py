import os 

def main(select_idx):
    #       0        1       2       3     4      5 
    names = ['377', '386', '387', '392', '393', '394']
    viewid = [0,0,0,0,0,0]

    idxs = [select_idx]
    
    train_stage1 = False  
    render_stage1 = False    
    train_stage2 = False 
    render_stage2 = True    
    render_stage2_interpolation = False  
    render_stage2_random_motion = False  

    for idx in idxs:
        name = names[idx]
        nnode="512"
        view_id=str(viewid[idx])
        pretrain_model_path="./outputs/stage1_res/zju_" + name + "_" + nnode
        outname="./saved_final_results/zju_" + name + "_" + nnode + "_final"


        runf = "python train_gui.py --source_path ./data/zju_processed_mv/" + name + " --model_path " + pretrain_model_path + " --deform_type node --node_num " + nnode + " --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --iterations_node_rendering 10000 --data_type ZJU --use_isotropic_gs"
        if train_stage1:
            os.system(runf) 
        
        
        runf = "python render.py --source_path ./data/zju_processed_mv/" + name + " --model_path " + pretrain_model_path + " --deform_type node --node_num " + nnode + " --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --use_isotropic_gs"
        
        if render_stage1:
            os.system(runf) 
            
        
        knn=-1 
        # "CUDA_VISIBLE_DEVICES=6" + cuda_id    
        runf = "python run_train_rig.py --source_path ./data/zju_processed_mv/" + name + " --model_path " + outname + " --pretrain_model_path " + pretrain_model_path + "_node --deform_type node --node_num 512 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --iterations 100000 --use_isotropic_gs --use_skinning_weight_mlp --use_template_offsets --data_type ZJU --skeleton_warm_up 5000"

        if train_stage2:
            os.system(runf) 
        
        runf = "python render_rig.py --source_path ./data/zju_processed_mv/" + name + " --model_path " + outname + " --pretrain_model_path " + pretrain_model_path + " --deform_type node --node_num 512 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --skeleton_weight_knn " + str(knn) 
        
        if render_stage2:
            os.system(runf) 
        
        runf = "python render_rig.py --source_path ./data/zju_processed_mv/" + name + " --model_path " + outname + " --pretrain_model_path " + pretrain_model_path + " --deform_type node --node_num 512 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --skeleton_weight_knn " + str(knn) + " --mode time --view_id " + view_id + " --skip_test"
        
        if render_stage2_interpolation:
            os.system(runf) 
        
        runf = "python render_rig.py --source_path ./data/zju_processed_mv/" + name + " --model_path " + outname + " --pretrain_model_path " + pretrain_model_path + " --deform_type node --node_num 512 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --skeleton_weight_knn " + str(knn) + " --mode motion --view_id " + view_id + " --skip_test"
        
        if render_stage2_random_motion:
            os.system(runf) 


import argparse 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='example')
    parser.add_argument('idx', type=int, help='input data index (0~5)')
    args = parser.parse_args()
    
    main(args.idx)
    