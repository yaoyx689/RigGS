# Pre-processing data 

## Compute the semantic segmentation and 2D skeleton 

For the data used in the paper, you can download the preprocessed semantic segmentation and 2D skeleton from [here](https://drive.google.com/file/d/1P5YOLkqm2a4pEqkEvf0pq3UHb-hrlK_g/view?usp=sharing).
If you want to use our method on the other data, you can refer to the following.

1. (**Used in the paper**) Download the [Hi-LASSIE](https://github.com/google/hi-lassie) and install the dependences according to its introduction. Then put the `cal_semantic_seg.py` in the `hi-lassie/main/` folder and run 
    ```
    python cal_semantic_seg.py --data_dir ${DATA_PATH} --image_folder ${NAME_OF_IMAGE_FOLDER} --mask_folder ${NAME_OF_MASK_FOLDER} 
    ``` 
    Then the semantic segmentation and 2D skeleton will be obtained in `${DATA_PATH}/semantic_seg` and `${DATA_PATH}/train_thinned`. If the input image is RGBA type, then you can ignore the `--mask_folder`. 

2. (**2D skeleton only**) If you do not need to use the skeleton's symmetry correction, you can run the following code to extract only the 2D skeleton. In this case no additional libraries need to be installed. 
    ```
    python cal_2d_skeleton.py --data_dir ${DATA_PATH} --image_folder ${NAME_OF_IMAGE_FOLDER} --mask_folder ${NAME_OF_MASK_FOLDER} 
    ``` 
    Then the 2D skeleton will be obtained in `${DATA_PATH}/train_thinned`. If the input image is in RGBA type, then you can ignore the `--mask_folder`. Our code can be run without semantic segmentation. 


## Details of preprocessing ZJU-MoCap dataset 
Since our template-free method performs reconstruction and rigging simultaneously, it faces challenges with videos captured by a fixed camera. Improved results can be achieved when the camera is allowed to move. Therefore, we used 6 cameras (1, 5, 9, 13, 17, 21) to simulate monocular videos with camera movement. 
The specific steps are as follows:
1. Follow [HumanNeRF](https://github.com/chungyiweng/humannerf) to preprocess the data.
Note that the training view angle is set to 1 by default (see line 73 on `humannerf/tools/prepare_zju_mocap
/prepare_dataset.py` ). We need to change its value from 1 to 23, and run 
    ```
    python prepare_dataset.py --cfg xxx.yaml
    ```
    23 times to get the processed images and camera parameters for each view angle. Except for the selected 6 viewpoints, the other data will be used to evaluate the results. The results for each view can be saved in the `${seq_name}_processed/test/view_xx` folder.
2. Run the following commond and the training data will be saved in `${seq_name}_processed/train/`. 
    ```
    python construct_zju_train_cam.py --processed_dir ${seq_name}_processed/test/ --data_dir ${seq_name}_processed/ 
    ```
3. Process SMPL vertices. 
   - Download and install [SMPL3](https://github.com/DogeStudio/SMPL3) and put `process_SMPL_vertices.py` in folder `SMPL3/smpl_webuser
/hello_world/`; 
   - Download SMPL model from [here](https://smpl.is.tue.mpg.de/) and run 
   ```
   python process_SMPL_vertices.py --data_dir ${seq_name}_processed/ --smpl_model_path ${smpl_model_path}
   ```
   The results will be saved in `${seq_name}_processed/SMPL_prior` folder. 

## Folder structure
The structure of the data and pre-trained model can be referred to as follows:

```
./RigGS
├── saved_final_results
│   ├── jumpingjacks_512_final_node
│   │   ├── point_cloud
│   │   ├── skeleton
│   │   ├── skeleton_tree.npz
│   │   └── cfg_args
│   ├── hook_512_final_node
│   └── ...
└── data
    ├── dy_syn_data # For D-NeRF dataset and DG-Mesh dataset 
    │   ├── jumpingjacks
    │   │   ├── train
    │   │   ├── test
    │   │   ├── train_thinned 
    │   │   ├── semantic_seg 
    │   │   └── ...
    │   ├── hook 
    │   └── ...
    └── zju_processed # For ZJU-MoCap dataset
        ├── 377
        |   ├── train
        |   |   ├── images
        |   |   ├── masks
        |   |   ├── train_thinned
        |   |   ├── semantic_seg
        |   |   ├── cameras.pkl
        |   |   └── mesh_infos.pkl
        |   ├── test 
        |   |   ├── view_01
        |   |   |   ├── images
        |   |   |   ├── images
        |   |   |   ├── cameras.pkl
        |   |   |   └── mesh_infos.pkl
        |   |   └── ...
        |   ├── SMPL_prior 
        |   |   ├── frame_000000.npy # (the corresponding SMPL vertices (6890,3),only used during training.)
        |   |   └── ... 
        ├── 386
        └── ...
```
