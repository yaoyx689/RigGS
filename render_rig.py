#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene, DeformModel
from scene.skeleton_model import SkeletonModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.pose_utils import pose_spherical
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel
import imageio
import numpy as np
from pytorch_msssim import ms_ssim
from piq import LPIPS
lpips = LPIPS()
from utils.image_utils import ssim as ssim_func
from utils.image_utils import psnr, lpips, alex_lpips
from skeleton_utils.visualization import vis_blending_weight_all, write_to_obj
from scipy.spatial.transform import Rotation
import random 
import cv2
import math 
from skeleton_utils.visualization import get_color_for_skinning_weights 


def project_nodes_to_2d_withnodes(viewpoint_cam, template_nodes, nodes, parents, render_img,out_path,thickness=1):
    
    viewmatrix = viewpoint_cam.world_view_transform
    tan_fovx = math.tan(viewpoint_cam.FoVx * 0.5)
    tan_fovy = math.tan(viewpoint_cam.FoVy * 0.5)
    height = viewpoint_cam.image_height 
    width = viewpoint_cam.image_width
    fy = height / (2 * tan_fovy)
    fx = width / (2 * tan_fovx)
    # cx = width / 2
    # cy = height / 2
    
    if viewpoint_cam.K is not None:
        cx = viewpoint_cam.K[0,2]
        cy = viewpoint_cam.K[1,2]
    else:
        cx = width / 2
        cy = height / 2
            
    trans_nodes = torch.matmul(nodes, viewmatrix[:3,:3]) + viewmatrix[3,:3] 
    
    proj_x = (fx * trans_nodes[:,0]/trans_nodes[:,2] + cx+0.5).int()
    proj_y = (fy * trans_nodes[:,1]/trans_nodes[:,2] + cy+0.5).int()
    
    proj_xy = torch.stack([proj_x, proj_y], dim=-1) 
    parents = parents.long()
    edges = torch.stack([proj_xy[1:], proj_xy[parents[1:]]], dim=-2)
    
    proj_xy = proj_xy.detach().cpu().numpy()
    edges = edges.detach().cpu().numpy()
    
    render_img_cv = render_img.permute(1, 2, 0).cpu().numpy()
    alpha_img = render_img_cv[...,3:]
    alpha_img = np.repeat(alpha_img, 3, axis=2).astype(np.float32)
    
    rgb_img = np.zeros([render_img_cv.shape[0], render_img_cv.shape[1],3], dtype=np.float32) + render_img_cv[...,:3].astype(np.float32)
    
    color = [0,0,0]  
    for i in range(nodes.shape[0]-1):
        alpha_img = cv2.polylines(img=alpha_img, pts=[edges[i].astype(np.int32)], isClosed=False, color=[1, 1, 1], thickness=thickness)
        
        rgb_img = cv2.polylines(img=rgb_img, pts=[edges[i].astype(np.int32)], isClosed=False, color=[float(color[0]), float(color[1]), float(color[2])], thickness=thickness)
    
    # from skeleton_utils.visualization import get_geometric_color 
    # # colors = get_geometric_color(template_nodes)
    for i in range(nodes.shape[0]):
        alpha_img = cv2.circle(img=alpha_img, center=proj_xy[i].astype(np.int32), radius=3, color=[1, 1, 1], thickness=-1)
        rgb_img = cv2.circle(img=rgb_img, center=proj_xy[i].astype(np.int32), radius=3, color=[float(color[0]), float(color[1]), float(color[2])], thickness=-1)
    
    
    rgb_img = np.concatenate([rgb_img, alpha_img[..., :1]], axis=-1)
    
    rgb_img = torch.tensor(rgb_img).permute(2, 0, 1)
    torchvision.utils.save_image(rgb_img, out_path)
    return rgb_img 

def sampling_skeleton_points(joints, parents, num_sample=512):
    points_c = joints[1:]
    points_p = joints[parents[1:].long()]
    
    distance = (points_c - points_p).norm(dim=-1).detach()
    each_distance = distance.sum()/num_sample 
    max_distance = distance.max() 
    
    t = torch.linspace(0,1,int(max_distance/each_distance), device=points_c.device)[:,None,None]
    new_ps = t * joints[1:] + (1-t)*joints[parents[1:].long()]
    sampling_points = new_ps.reshape(-1,3)
    
    return sampling_points
    

def render_set(model_path, load2gpt_on_the_fly, name, iteration, views, gaussians, pipeline, background, deform, skeleton, template_idx, template_offsets=None,view_id=0):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    pc_path = os.path.join(model_path, name, "ours_{}".format(iteration), "pc")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(pc_path, exist_ok=True)

    # Measurement
    psnr_list, ssim_list, lpips_list = [], [], []
    ms_ssim_list, alex_lpips_list = [], []

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
    renderings = []
    rendering_skinnings = []
    gts = []
    
    write_to_obj(skeleton.deform.nodes.detach().cpu(), os.path.join(pc_path, 'nodes_template.obj'), parents=skeleton.deform.parents)
    write_to_obj((gaussians.get_xyz.detach()).cpu(), os.path.join(pc_path, 'canonical_xyz.obj'))
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpt_on_the_fly:
            view.load2device()
        fid = view.fid
        xyz = gaussians.get_xyz
        
        time_input = skeleton.deform.expand_time(fid)
        canonical_xyz = xyz.detach() 
        
        d_values = skeleton.step(canonical_xyz, time_input, motion_mask=gaussians.motion_mask)
        d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']
    
        vis_rig_path = os.path.join(pc_path, 'rig')
        
        d_scaling = torch.zeros_like(d_scaling).to(d_scaling.device)
        if gaussians.use_isotropic_gs:
            d_rotation = torch.zeros_like(d_rotation).to(d_rotation.device)
        
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=deform.d_rot_as_res)
        alpha = results["alpha"]
        rendering = torch.clamp(torch.cat([results["render"], alpha]), 0.0, 1.0)
        
        vn_idx = d_values['nn_idx']
        vn_weight = d_values['nn_weight']
        skinning_color = get_color_for_skinning_weights(canonical_xyz, vn_idx=vn_idx, vn_weight=vn_weight,control_points=skeleton.deform.nodes.detach()[:,:3])
        results_skinning = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=deform.d_rot_as_res, override_color=skinning_color)
        rendering_skinning = torch.clamp(torch.cat([results_skinning["render"], results_skinning["alpha"]]), 0.0, 1.0)
        
        # Measurement
        image = rendering[:3]
        gt_image = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
        psnr_list.append(psnr(image[None], gt_image[None]).mean())
        ssim_list.append(ssim_func(image[None], gt_image[None], data_range=1.).mean())
        lpips_list.append(lpips(image[None], gt_image[None]).mean())
        ms_ssim_list.append(ms_ssim(image[None], gt_image[None], data_range=1.).mean())
        alex_lpips_list.append(alex_lpips(image[None], gt_image[None]).mean())

        # for i in range(5):
        renderings.append(to8b(rendering.cpu().numpy()))
        gts.append(to8b(gt_image.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)
        

        gt = view.original_image[0:4, :, :]
        gt = torch.cat([gt, view.gt_alpha_mask])
        
        rendering_skinning = project_nodes_to_2d_withnodes(view, skeleton.deform.nodes[:,:3], d_values['d_nodes'], skeleton.deform.parents, rendering_skinning, os.path.join(render_path, '{0:05d}'.format(idx) + "_rig.png"))
        rendering_skinnings.append(to8b(rendering_skinning.cpu().numpy()))
        
        
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        
        if not os.path.exists(vis_rig_path + '_skinning_weight.ply'):
            vn_idx = d_values['nn_idx']
            vn_weight = d_values['nn_weight']
            
            vis_blending_weight_all(vis_rig_path, canonical_xyz, None, vn_idx, vn_weight, skeleton.deform.nodes.detach()[:,:3], None, None)
        
        
    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    gts = np.stack(gts, 0).transpose(0,2,3,1)
    rendering_skinnings = np.stack(rendering_skinnings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)
    imageio.mimwrite(os.path.join(render_path, 'video_rig.mp4'), rendering_skinnings, fps=30, quality=8)
    imageio.mimwrite(os.path.join(gts_path, 'video.mp4'), renderings, fps=30, quality=8)

    # Measurement
    psnr_test = torch.stack(psnr_list).mean()
    ssim_test = torch.stack(ssim_list).mean()
    lpips_test = torch.stack(lpips_list).mean()
    ms_ssim_test = torch.stack(ms_ssim_list).mean()
    alex_lpips_test = torch.stack(alex_lpips_list).mean()
    print("\n[ITER {}] Evaluating {}: PSNR {} SSIM {} LPIPS {} MS SSIM{} ALEX_LPIPS {}".format(iteration, name, psnr_test, ssim_test, lpips_test, ms_ssim_test, alex_lpips_test))
    
    # each_results
    fid = open(os.path.join(model_path, name, "ours_{}".format(iteration), 'numerical_res.txt'), 'w')
    print('ID\tpsnr\tssim\tlpips\tms_ssim\talex_lpips', file=fid)
    for i in range(len(psnr_list)):
        print('%d\t%.2f\t%.4f\t%.4f\t%.4f\t%.4f'%(i,psnr_list[i], ssim_list[i], lpips_list[i], ms_ssim_list[i], alex_lpips_list[i]), file=fid)
    print('mean\t%.2f\t%.4f\t%.4f\t%.4f\t%.4f'%(psnr_test, ssim_test, lpips_test, ms_ssim_test, alex_lpips_test), file=fid)
    fid.close()   
    return template_offsets


def generate_random_quaternion():
    axis = np.random.rand(3)
    axis /= np.linalg.norm(axis)

    angle = np.random.uniform(0, 0.1 * np.pi)

    r = Rotation.from_rotvec(axis * angle)
    q = r.as_quat()

    return q 


def generate_continuous_random_quaternion(num):
    axis = np.random.rand(3)
    axis /= np.linalg.norm(axis)

    start_angle = -np.pi/6  
    end_angle = np.pi/6 

    qs = []
    for k in range(num):
        each_angle = k*(end_angle - start_angle)/num 
        r = Rotation.from_rotvec(axis * (start_angle + each_angle))
        q = r.as_quat()
        qs.append(np.array([q[3],q[0],q[1],q[2]]))

    return qs


def generate_random_motion(model_path, load2gpt_on_the_fly, name, iteration, views, gaussians, pipeline, background, deform, skeleton, template_idx, template_offsets=None,view_id=0):
    render_path = os.path.join(model_path, name, "motion_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "motion_{}".format(iteration), "depth")
    pc_path = os.path.join(model_path, name, "motion_{}".format(iteration), "pc")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(pc_path, exist_ok=True)
        
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = view_id
    view = views[idx]
    renderings = []
    xyz = gaussians.get_xyz
    
    template_view = views[template_idx]
    
    pose_num = 60
    n_nodes = skeleton.deform.nodes.shape[0]
    change_node_num = int(0.3*n_nodes)
    
    all_qs = []
    random_node = random.sample(range(5, n_nodes), change_node_num)
    for k in range(change_node_num):
        qs = generate_continuous_random_quaternion(pose_num)
        all_qs.append(qs)
    
    all_node_attrs = []
    for i in range(pose_num):
        node_attrs = {}
        node_attrs['d_xyz'] = torch.zeros(skeleton.deform.nodes.shape[0], 3).to(xyz.device)
        node_attrs['local_rotation'] = torch.zeros(skeleton.deform.nodes.shape[0], 4).to(xyz.device)
        node_attrs['local_rotation'][:,0] = 1
        node_attrs['d_scaling'] = torch.zeros(skeleton.deform.nodes.shape[0], 3).to(xyz.device)
        node_attrs['d_rotation'] = None
            
        for j in range(change_node_num):
            nj = random_node[j]
            node_attrs['local_rotation'][nj,:] = torch.tensor(all_qs[j][i])
            
        all_node_attrs.append(node_attrs)
        
        
    
    canonical_xyz = gaussians.get_xyz.detach()
    rendering_skinnings = []
    for idx in tqdm(range(len(all_node_attrs))):
        
        node_attrs = all_node_attrs[idx]
        node_attrs['global_trans'] = torch.zeros(1,3).to(skeleton.deform.nodes.device)
        node_attrs['t'] = template_view.fid
        
        d_values = skeleton.deform.deform_by_pose(canonical_xyz, node_attrs, motion_mask=gaussians.motion_mask)
        d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']
        d_scaling = torch.zeros_like(d_scaling).to(d_scaling.device)
        
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=deform.d_rot_as_res)
        alpha = results["alpha"]
        rendering = torch.clamp(torch.cat([results["render"], alpha]), 0.0, 1.0)
        for i in range(1):
            renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        write_to_obj(d_values['d_nodes'], os.path.join(pc_path, 'nodes_{0:05d}'.format(idx) + '.obj'), parents=skeleton.deform.parents)
        
        vn_idx = d_values['nn_idx']
        vn_weight = d_values['nn_weight']
        skinning_color = get_color_for_skinning_weights(canonical_xyz, vn_idx=vn_idx, vn_weight=vn_weight,control_points=skeleton.deform.nodes.detach()[:,:3])
        results_skinning = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=deform.d_rot_as_res, override_color=skinning_color)
        rendering_skinning = torch.clamp(torch.cat([results_skinning["render"], results_skinning["alpha"]]), 0.0, 1.0)
        
        rendering_skinning = project_nodes_to_2d_withnodes(view, skeleton.deform.nodes[:,:3], d_values['d_nodes'], skeleton.deform.parents, rendering_skinning, os.path.join(render_path, '{0:05d}'.format(idx) + "_rig.png"))
        
        for i in range(1):
            rendering_skinnings.append(to8b(rendering_skinning.cpu().numpy()))
        

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    rendering_skinnings = np.stack(rendering_skinnings, 0).transpose(0,2,3,1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)
    imageio.mimwrite(os.path.join(render_path, 'video_rig.mp4'), rendering_skinnings, fps=30, quality=8)


def interpolate_time(model_path, load2gpt_on_the_fly, name, iteration, views, gaussians, pipeline, background, deform, skeleton, template_idx, template_offsets=None,view_id=0):
    render_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
        

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    frame = 200
    idx = view_id
    view = views[idx]
    renderings = []
    rendering_skinnings = []
    for t in tqdm(range(0, frame, 1), desc="Rendering progress"):
        fid = torch.Tensor([t / (frame - 1)]).cuda()
        xyz = gaussians.get_xyz
        if deform.name == 'deform':
            time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        elif deform.name == 'node':
            time_input = skeleton.deform.expand_time(fid)
        
        
        canonical_xyz = gaussians.get_xyz.detach() 
        d_values = skeleton.step(canonical_xyz, time_input, motion_mask=gaussians.motion_mask)
        
        d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']
        
        d_scaling = torch.zeros_like(d_scaling).to(d_scaling.device)
        
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=deform.d_rot_as_res)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(t) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(t) + ".png"))
        
        rendering = torch.clamp(torch.cat([results["render"], results["alpha"]]), 0.0, 1.0)
        
        vn_idx = d_values['nn_idx']
        vn_weight = d_values['nn_weight']
        skinning_color = get_color_for_skinning_weights(canonical_xyz, vn_idx=vn_idx, vn_weight=vn_weight,control_points=skeleton.deform.nodes.detach()[:,:3])
        results_skinning = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=deform.d_rot_as_res, override_color=skinning_color)
        rendering_skinning = torch.clamp(torch.cat([results_skinning["render"], results_skinning["alpha"]]), 0.0, 1.0)
        
        rendering_skinning = project_nodes_to_2d_withnodes(view, skeleton.deform.nodes[:,:3], d_values['d_nodes'], skeleton.deform.parents, rendering_skinning, os.path.join(render_path, '{0:05d}'.format(t) + "_rig.png"))
        
        rendering_skinnings.append(to8b(rendering_skinning.cpu().numpy())) 
        
    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)
    
    rendering_skinnings = np.stack(rendering_skinnings,0).transpose(0,2,3,1)
    imageio.mimwrite(os.path.join(render_path, 'video_rig.mp4'), rendering_skinnings, fps=30, quality=8)



def interpolate_all(model_path, load2gpt_on_the_fly, name, iteration, views, gaussians, pipeline, background, deform, skeleton, template_idx, template_offsets=None,view_id=0):
    render_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
        

    frame = 150
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, frame + 1)[:-1]], 0)
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]  # Choose a specific time for rendering

    renderings = []
    for i, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        fid = torch.Tensor([i / (frame - 1)]).cuda()

        matrix = np.linalg.inv(np.array(pose))
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        view.reset_extrinsic(R, T)

        canonical_xyz = gaussians.get_xyz.detach() 
        time_input = skeleton.deform.expand_time(fid)

        d_values = skeleton.step(canonical_xyz, time_input, motion_mask=gaussians.motion_mask)
        
        d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']
        
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=deform.d_rot_as_res)
        rendering = torch.clamp(results["render"], 0.0, 1.0)
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(i) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def render_sets(dataset: ModelParams, iteration: int, skeleton_knn: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool, mode: str, view_id: int, load2device_on_the_fly=False):
    with torch.no_grad():
        
        deform = DeformModel(K=dataset.K, deform_type=dataset.deform_type, is_blender=dataset.is_blender, skinning=dataset.skinning, hyper_dim=dataset.hyper_dim, node_num=dataset.node_num, pred_opacity=dataset.pred_opacity, pred_color=dataset.pred_color, use_hash=dataset.use_hash, hash_time=dataset.hash_time, d_rot_as_res=dataset.d_rot_as_res, local_frame=dataset.local_frame, progressive_brand_time=dataset.progressive_brand_time, max_d_scale=dataset.max_d_scale)
        check = deform.load_weights(dataset.pretrain_model_path, iteration=-1)
        print('load_deform weight', check)
        
        skeleton_path = dataset.model_path + '/skeleton_tree.npz'
        tree = np.load(skeleton_path, allow_pickle=True)
        joints = torch.from_numpy(tree['nodes']).to(deform.deform.nodes.device)
        parents = torch.from_numpy(tree['parents']).to(deform.deform.nodes.device) 
        template_idx = tree['template_idx']
        
        control_nodes = torch.zeros(512,3).to(joints.device)
        skeleton = SkeletonModel(control_nodes=control_nodes,K=skeleton_knn,is_blender=dataset.is_blender, skinning=dataset.skinning, hyper_dim=dataset.hyper_dim, joints=joints, parent_indices=parents, pred_opacity=dataset.pred_opacity, pred_color=dataset.pred_color, use_hash=dataset.use_hash, hash_time=dataset.hash_time, d_rot_as_res=dataset.d_rot_as_res, local_frame=dataset.local_frame, progressive_brand_time=dataset.progressive_brand_time, max_d_scale=dataset.max_d_scale, use_skinning_weight_mlp=dataset.use_skinning_weight_mlp, use_template_offsets=dataset.use_template_offsets)
        skeleton.load_weights(dataset.model_path, iteration=iteration)
        

        gs_fea_dim = deform.deform.node_num if dataset.skinning and deform.name == 'node' else dataset.hyper_dim
        gaussians = GaussianModel(dataset.sh_degree, fea_dim=gs_fea_dim, with_motion_mask=dataset.gs_with_motion_mask, use_isotropic_gs=dataset.use_isotropic_gs)
        scene = Scene(dataset, gaussians, load_iteration=iteration,shuffle=False,use_pretrain_model=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if mode == "render":
            render_func = render_set
        elif mode == "time":
            render_func = interpolate_time
        elif mode == 'motion':
            render_func = generate_random_motion 
        else:
            render_func = interpolate_all

        if not skip_train:
            render_func(dataset.model_path, load2device_on_the_fly, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, deform, skeleton, template_idx, None, view_id)

        if not skip_test:
            render_func(dataset.model_path, load2device_on_the_fly, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, deform, skeleton, template_idx, None)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render', 'time', 'view', 'all', 'pose', 'original', 'motion'])
    parser.add_argument("--view_id", default=0, help="set a view for interpolation or generate new motion")
    
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--elevation', type=float, default=0, help="default GUI camera elevation")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000] + list(range(10000, 80_0001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40000])
    parser.add_argument("--deform-type", type=str, default='mlp')

    args = get_combined_args(parser)
    if not args.model_path.endswith(args.deform_type):
        args.model_path = os.path.join(os.path.dirname(os.path.normpath(args.model_path)), os.path.basename(os.path.normpath(args.model_path)) + f'_{args.deform_type}')
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, args.skeleton_weight_knn, pipeline.extract(args), args.skip_train, args.skip_test, args.mode, int(args.view_id), load2device_on_the_fly=args.load2gpu_on_the_fly)
