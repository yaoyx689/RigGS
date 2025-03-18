```
./RigGS
├── checkpoints
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