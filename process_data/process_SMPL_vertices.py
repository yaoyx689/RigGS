import sys
# sys.path.append('${ROOT_dir}/SMPL3/smpl_webuser/')
import os 
import pickle 
from serialization import load_model
import pickle
import numpy as np
from argparse import ArgumentParser

def pose2obj(pose, shape, trans, outmesh_path, smpl_model_path):
    ## Load SMPL model (here we load the female model)
    ## Make sure path is correct
    m = load_model(os.path.join(smpl_model_path, 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl'))
    
    m.pose[:] = pose 
    m.betas[:] = shape
    m.trans[:] = trans 
    
    posed_J = m.J_transformed 
    tree = m.kintree_table 
    
    np.save(outmesh_path[:-4] + '.npy', m.r)
    
    with open(outmesh_path, 'w') as fp:
        for v in m.r:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in m.f + 1: 
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    return posed_J, tree


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./377/')
    parser.add_argument('--smpl_model_path', type=str, default='./smpl_models/')
    args = parser.parse_args()
    
    data_dir = args.data_dir
    pose_file = os.path.join(data_dir, "train", "mesh_infos.pkl")
    out_path = os.path.join(data_dir, "SMPL_prior/")
    os.makedirs(out_path, exist_ok=True) 
    
    with open(pose_file, 'rb') as f:
        mesh_infos = pickle.load(f)
    
    smpl_model_path = args.smpl_model_path
    posed_Js = []
    for fname in mesh_infos:
        poses = mesh_infos[fname]["poses"]
        shape = np.zeros(10)
        trans = np.zeros(3)
        out_mesh = out_path + fname + ".obj"
        posed_J, tree = pose2obj(poses, shape, trans, out_mesh, smpl_model_path)
        posed_Js.append(posed_J)
    
