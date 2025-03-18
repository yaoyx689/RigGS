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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, network_gui 
# import sys
from scene import Scene, GaussianModel, DeformModel
from scene.skeleton_model import SkeletonModel 
from utils.general_utils import get_linear_noise_func
import uuid
import tqdm
from argparse import  Namespace
from train_utils import skeleton_training_report
import numpy as np
import dearpygui.dearpygui as dpg
from pytorch3d.loss import chamfer_distance
from skeleton_utils.extract_skeleton_utils import obtain_skeleton_tree
from utils.system_utils import searchForMaxIteration
from utils.time_utils import farthest_point_sample
from utils.other_utils import project_nodes_to_2d_elements

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


class TrainRig:
    def __init__(self, args, dataset, opt, pipe, testing_iterations, saving_iterations) -> None:
        self.dataset = dataset
        self.args = args
        self.opt = opt
        self.pipe = pipe
        self.testing_iterations = testing_iterations
        self.saving_iterations = saving_iterations
        self.device = 'cuda'

        if self.opt.progressive_train:
            self.opt.iterations_node_sampling = max(self.opt.iterations_node_sampling, int(self.opt.progressive_stage_steps / self.opt.progressive_stage_ratio))
            self.opt.iterations_node_rendering = max(self.opt.iterations_node_rendering, self.opt.iterations_node_sampling + 2000)
            print(f'Progressive trian is on. Adjusting the iterations node sampling to {self.opt.iterations_node_sampling} and iterations node rendering {self.opt.iterations_node_rendering}')

        self.tb_writer = prepare_output_and_logger(dataset)

        gs_fea_dim = self.dataset.hyper_dim
        self.gaussians = GaussianModel(dataset.sh_degree, fea_dim=gs_fea_dim, with_motion_mask=self.dataset.gs_with_motion_mask, use_isotropic_gs=self.dataset.use_isotropic_gs)

        load_iteration = -1
        loaded_iter = searchForMaxIteration(os.path.join(self.dataset.model_path, "point_cloud"))
    
        if loaded_iter is None or loaded_iter < self.opt.skeleton_warm_up:
            self.scene = Scene(dataset, self.gaussians, load_iteration=load_iteration, use_pretrain_model=True)
            self.iteration = 1
            self.init_with_pretrain_model = True 
            print('use pretrained deform model to init, #iters=', self.scene.loaded_iter)
        else:
            self.scene = Scene(dataset, self.gaussians, load_iteration=load_iteration, use_pretrain_model=False)
            self.iteration = 1 if self.scene.loaded_iter is None else self.scene.loaded_iter
            self.init_with_pretrain_model = False 
            print('load checkpoint, #iters=', self.scene.loaded_iter)

        self.gaussians.training_setup(opt)

        deform, self.pretrain_deform_info, skeleton_tree_info = self.init_skeleton_info()
        self.template_idx = skeleton_tree_info['template_idx']
        print('template_idx = ', self.template_idx)

        joints, parent_indices, joint_node_indices = skeleton_tree_info['joints'], skeleton_tree_info['parent_indices'], skeleton_tree_info['joint_node_indices'].long()
        self.joints = joints             

        self.skeleton = SkeletonModel(K=self.opt.skeleton_weight_knn, is_blender=self.dataset.is_blender, skinning=self.args.skinning, hyper_dim=self.dataset.hyper_dim, joints=joints, parent_indices=parent_indices, pred_opacity=self.dataset.pred_opacity, pred_color=self.dataset.pred_color, use_hash=self.dataset.use_hash, hash_time=self.dataset.hash_time, d_rot_as_res=self.dataset.d_rot_as_res and not self.dataset.d_rot_as_rotmat, local_frame=self.dataset.local_frame, progressive_brand_time=self.dataset.progressive_brand_time, with_arap_loss=not self.opt.no_arap_loss, max_d_scale=self.dataset.max_d_scale, enable_densify_prune=self.opt.node_enable_densify_prune, is_scene_static=self.dataset.is_scene_static,  use_skinning_weight_mlp=self.dataset.use_skinning_weight_mlp, use_template_offsets=self.dataset.use_template_offsets)
        self.skeleton.train_setting(self.opt)
        
        if self.init_with_pretrain_model:
            self.skeleton.deform._node_radius.data = deform.deform._node_radius[joint_node_indices]

            self.iteration = 1
        else:
            skeleton_load = self.skeleton.load_weights(dataset.model_path, iteration=load_iteration)
            self.iteration = 1 if self.scene.loaded_iter is None else self.scene.loaded_iter
            print('skeleton loaded_iter = ', loaded_iter)

        

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.iter_start = torch.cuda.Event(enable_timing=True)
        self.iter_end = torch.cuda.Event(enable_timing=True)

        self.viewpoint_stack = None
        self.ema_loss_for_log = 0.0
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        self.best_ms_ssim = 0.0
        self.best_lpips = np.inf
        self.best_alex_lpips = np.inf
        self.best_iteration = 0
        self.progress_bar = tqdm.tqdm(range(opt.iterations), desc="Training progress")
        self.smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)        
        self.all_nodes_projection_loss = 1.0e5*torch.ones(len(self.scene.getTrainCameras())).to(self.device)
        
    
    def init_skeleton_info(self):
        deform = None 
        if self.init_with_pretrain_model:
            deform = DeformModel(K=self.dataset.K, deform_type=self.dataset.deform_type, is_blender=self.dataset.is_blender, skinning=self.args.skinning, hyper_dim=self.dataset.hyper_dim, node_num=self.dataset.node_num, pred_opacity=self.dataset.pred_opacity, pred_color=self.dataset.pred_color, use_hash=self.dataset.use_hash, hash_time=self.dataset.hash_time, d_rot_as_res=self.dataset.d_rot_as_res and not self.dataset.d_rot_as_rotmat, local_frame=self.dataset.local_frame, progressive_brand_time=self.dataset.progressive_brand_time, with_arap_loss=not self.opt.no_arap_loss, max_d_scale=self.dataset.max_d_scale, enable_densify_prune=self.opt.node_enable_densify_prune, is_scene_static=self.dataset.is_scene_static)
            deform_loaded = deform.load_weights(self.dataset.pretrain_model_path, iteration=-1)
            deform.train_setting(self.opt)

            if not deform_loaded:
                print('pre-train_model path = ', self.dataset.pretrain_model_path, 'data_loaded = ', deform_loaded)
                print('Error: no pre-trained node-based deform model!')
                exit(1)

            if self.scene.getTrainCameras()[0] is not None:
                self.nodes_semantic_label = torch.zeros((len(self.scene.getTrainCameras()),deform.deform.node_num)).float().to(self.device)
            else:
                self.nodes_semantic_label = None 
            pretrain_deform_info, skeleton_tree_info = self.precompute_deformations(deform, True)
            
            joint_node_indices = skeleton_tree_info['joint_node_indices'].long()
            
            pretrain_deform_info['d_joints'] = pretrain_deform_info['d_nodes'][:, joint_node_indices]
            template_idx = skeleton_tree_info['template_idx']
            template_offsets = self.cal_template_offsets(deform, template_idx)
            self.gaussians._xyz.data = self.gaussians.get_xyz.detach() + template_offsets 
            pretrain_deform_info['d_xyz'] = pretrain_deform_info['d_xyz'] - template_offsets[None]
        else:
            pretrain_deform_info = None 
            _, skeleton_tree_info = self.precompute_deformations(None, False)
        
        return deform, pretrain_deform_info, skeleton_tree_info

    
    def cal_max_coverage_view(self, indices):
        sorted_train_cams = sorted(self.scene.getTrainCameras().copy(), key=lambda x: x.fid) 
        num_nonzeros = []
        for idx in range(len(indices)):
            viewpoint_cam = sorted_train_cams[indices[idx]]
            mask = viewpoint_cam.gt_alpha_mask
            num_nonzero = mask.sum()
            num_nonzeros.append(num_nonzero)
            
        num_nonzeros = torch.stack(num_nonzeros)
        max_index = torch.argmax(num_nonzeros)
        return indices[max_index]
            

    # d_nodes: (n_frames, n_nodes, 3)
    def select_key_frame(self, d_nodes):
        if self.opt.manually_key_frame >= 0:
            return self.opt.manually_key_frame

        mean_nodes = d_nodes.mean(dim=0)[None]
        distances = (d_nodes - mean_nodes).norm(dim=-1)
        mean_distances = distances.mean(dim=-1)
        k = 5
        _, min_indices = torch.topk(mean_distances, k=k, largest=False)
        select_idx = self.cal_max_coverage_view(min_indices)
        return int(select_idx)

    
    def set_semantic_label(self, elements, viewpoint_cam):
        if viewpoint_cam.semantic_seg is None:
            return 
        proj_positions = elements.detach().clone().long()
        height = viewpoint_cam.image_height 
        width = viewpoint_cam.image_width
        select = proj_positions[:,0] >= height 
        proj_positions[select, 0] = height-1
        select = proj_positions[:,1] >= width 
        proj_positions[select, 1] = width - 1 
        select = proj_positions < 0 
        proj_positions[select] = 0
        semantic_label = viewpoint_cam.semantic_seg[proj_positions[:,0], proj_positions[:,1]]
        self.nodes_semantic_label[viewpoint_cam.uid,:] = semantic_label

    def precompute_deformations(self, deform, init=True):
        skeleton_tree_path = os.path.join(self.dataset.model_path, 'skeleton_tree.npz')
        if init:
            if self.opt.num_gs_sample > 10:
                self.gaussians.sampling_and_prune(self.opt.num_gs_sample)
            sorted_train_cams = sorted(self.scene.getTrainCameras().copy(), key=lambda x: x.fid)   
            all_trans_nodes = []
            all_d_xyz = []
            all_d_rotation = []
            all_d_scaling = []
            if self.nodes_semantic_label is not None:
                self.nodes_semantic_label = self.nodes_semantic_label[:,:deform.deform.nodes.shape[0]]
            for viewpoint_cam in sorted_train_cams:
                time_input = deform.deform.expand_time(viewpoint_cam.fid)
                d_values = deform.step(self.gaussians.get_xyz.detach(), time_input, feature=self.gaussians.feature, motion_mask=self.gaussians.motion_mask)
                d_xyz, d_nodes = d_values['d_xyz'], d_values['d_nodes']
                all_trans_nodes.append(d_nodes)
                all_d_xyz.append(d_xyz)
                
                d_rotation, d_scaling = d_values['d_rotation'], d_values['d_scaling']
                all_d_rotation.append(d_rotation)
                all_d_scaling.append(d_scaling)
                
                proj_nodes = project_nodes_to_2d_elements(viewpoint_cam, d_nodes)
                self.set_semantic_label(proj_nodes, viewpoint_cam)
            
            all_d_xyz = torch.stack(all_d_xyz).detach()
            all_trans_nodes = torch.stack(all_trans_nodes).detach()
            all_d_rotation = torch.stack(all_d_rotation).detach()
            all_d_scaling = torch.stack(all_d_scaling).detach()

            template_idx = self.select_key_frame(all_trans_nodes)
            
            template_nodes = all_trans_nodes[template_idx]
            if self.nodes_semantic_label is not None:
                med_seg_label = torch.median(self.nodes_semantic_label, dim=0).values.int()
            else:
                med_seg_label = None 
 
            joints, parent_indices, joint_node_indices = obtain_skeleton_tree(template_nodes, all_trans_nodes, med_seg_label)

            np.savez(skeleton_tree_path, nodes=joints.cpu().numpy(), parents=parent_indices.cpu().numpy(), indices=joint_node_indices.cpu().numpy(), template_idx=int(template_idx))
        
            pretrain_deform_info = {'d_xyz': all_d_xyz.detach(),'d_nodes': all_trans_nodes.detach(), 'd_rotation': all_d_rotation.detach(), 'd_scaling': all_d_scaling.detach()}

            skeleton_tree_info = {'joints': joints, 'parent_indices': parent_indices, 'joint_node_indices': joint_node_indices, 'template_idx': int(template_idx)}
            
            from skeleton_utils.visualization import write_to_obj
            write_to_obj(joints, os.path.join(self.dataset.model_path, 'skeleton.obj'), parent_indices)

            return pretrain_deform_info, skeleton_tree_info 
        else:
            if os.path.exists(skeleton_tree_path):
                skeleton_tree = np.load(skeleton_tree_path)
                joints = torch.from_numpy(skeleton_tree['nodes']).float().to(self.device)
                parent_indices = torch.from_numpy(skeleton_tree['parents']).long().to(self.device)
                joint_node_indices = torch.from_numpy(skeleton_tree['indices']).long().to(self.device)
                template_idx = skeleton_tree['template_idx']
                skeleton_tree_info = {'joints': joints, 'parent_indices': parent_indices.long(), 'joint_node_indices': joint_node_indices, 'template_idx': template_idx}
            else:
                print('Error: NO saved skeleton tree!')
                exit(0)
            return None, skeleton_tree_info 
    
    def cal_template_offsets(self, deform, template_idx):
        sorted_train_cams = sorted(self.scene.getTrainCameras().copy(), key=lambda x: x.fid)   
        viewpoint_cam = sorted_train_cams[template_idx]
        time_input = deform.deform.expand_time(viewpoint_cam.fid)
        d_values = deform.step(self.gaussians.get_xyz.detach(), time_input, feature=self.gaussians.feature, motion_mask=self.gaussians.motion_mask)
        template_offsets = d_values['d_xyz'].detach()
        return template_offsets 

    def sampling_skeleton_points(self, joints, parents, num_sample=512):
        points_c = joints[1:]
        points_p = joints[parents[1:].long()]
        
        distance = (points_c - points_p).norm(dim=-1).detach()
        each_distance = distance.sum()/num_sample 
        max_distance = distance.max() 
        
        t = torch.linspace(0,1,int(max_distance/each_distance), device=points_c.device)[:,None,None]
        new_ps = t * joints[1:] + (1-t)*joints[parents[1:].long()]
        sampling_points = new_ps.reshape(-1,3)
        
        return sampling_points

    # no gui mode
    def train(self, iters=5000):
        if iters > 0:
            t0 = self.iteration 
            for i in tqdm.trange(t0, iters):
                if self.iteration < self.opt.skeleton_warm_up:
                    self.train_step(True)
                else:
                    self.train_step(False)
    

    def select_random_cam(self):
        # Pick a random Camera
        if not self.viewpoint_stack:
            if self.opt.progressive_train and self.iteration < int(self.opt.progressive_stage_steps / self.opt.progressive_stage_ratio):
                cameras_to_train_idx = int(min(((self.iteration) / self.opt.progressive_stage_steps + 1) * self.opt.progressive_stage_ratio, 1.) * len(self.scene.getTrainCameras()))
                cameras_to_train_idx = max(cameras_to_train_idx, 1)
                interval_len = int(len(self.scene.getTrainCameras()) * self.opt.progressive_stage_ratio)
                min_idx = max(0, cameras_to_train_idx - interval_len)
                sorted_train_cams = sorted(self.scene.getTrainCameras().copy(), key=lambda x: x.fid)
                viewpoint_stack = sorted_train_cams[min_idx: cameras_to_train_idx]
                out_domain_idx = np.arange(min_idx)
                if len(out_domain_idx) >= interval_len:
                    out_domain_idx = np.random.choice(out_domain_idx, [interval_len], replace=False)
                    out_domain_stack = [sorted_train_cams[idx] for idx in out_domain_idx]
                    viewpoint_stack = viewpoint_stack + out_domain_stack
            else:
                viewpoint_stack = self.scene.getTrainCameras().copy()
            self.viewpoint_stack = viewpoint_stack

    
    def cal_skeleton_loss(self, d_nodes, viewpoint_cam):
        sampling_points = self.sampling_skeleton_points(d_nodes, self.skeleton.deform.parents)
        proj_nodes = project_nodes_to_2d_elements(viewpoint_cam, sampling_points)
        gt_proj_nodes = viewpoint_cam.thinned
        chamfer_dists, _ = chamfer_distance(x=proj_nodes.unsqueeze(0), y=gt_proj_nodes.unsqueeze(0),norm=1)
        return chamfer_dists 


    def report_and_densification(self, loss, warmup_stage, viewpoint_cam, render_pkg_re, d_nodes):
        if self.dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')
        
        viewspace_point_tensor, visibility_filter, radii = render_pkg_re["viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]
            
        with torch.no_grad():
            # Progress bar
            self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
            if self.iteration % 10 == 0:
                self.progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{7}f}"})
                self.progress_bar.update(10)
            if self.iteration == self.opt.iterations:
                self.progress_bar.close()

            # Keep track of max radii in image-space for pruning
            if self.gaussians.max_radii2D.shape[0] == 0:
                self.gaussians.max_radii2D = torch.zeros_like(radii)
            self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

            # Log and save
            cur_psnr, cur_ssim, cur_lpips, cur_ms_ssim, cur_alex_lpips = skeleton_training_report(self.tb_writer, self.iteration, l1_loss, self.iter_start.elapsed_time(self.iter_end), self.testing_iterations, self.scene, render, (self.pipe, self.background), self.skeleton, self.dataset.load2gpu_on_the_fly, self.progress_bar)
            if self.iteration in self.testing_iterations:
                if cur_psnr.item() > self.best_psnr:
                    self.best_psnr = cur_psnr.item()
                    self.best_iteration = self.iteration
                    self.best_ssim = cur_ssim.item()
                    self.best_ms_ssim = cur_ms_ssim.item()
                    self.best_lpips = cur_lpips.item()
                    self.best_alex_lpips = cur_alex_lpips.item()

            if self.iteration in self.saving_iterations or self.iteration == self.best_iteration or self.iteration == self.opt.warm_up-1:
                print("\n[ITER {}] Saving Gaussians".format(self.iteration))
                self.scene.save(self.iteration)
                self.skeleton.save_weights(self.args.model_path, self.iteration)
                # self.skeleton.save_joints(self.args.model_path, self.iteration, d_nodes, viewpoint_cam.uid)


            # Densification
            if not warmup_stage and self.iteration < self.opt.densify_until_iter and self.iteration > self.opt.gs_densification_iterations:
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.iteration > self.opt.densify_from_iter and self.iteration % self.opt.densification_interval == 0:
                    size_threshold = 20 if self.iteration > self.opt.opacity_reset_interval else None
                    self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.005, self.scene.cameras_extent, size_threshold)

                if self.iteration % self.opt.opacity_reset_interval == 0 or (
                        self.dataset.white_background and self.iteration == self.opt.densify_from_iter):
                    self.gaussians.reset_opacity()

    
    def test_network_connect(self):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, self.pipe.do_shs_python, self.pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, self.dataset.source_path)
                if do_training and ((self.iteration < int(self.opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
        
        
    
    def deform_gaussians(self):
        total_frame = len(self.scene.getTrainCameras())
        time_interval = 1 / total_frame
        viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))
        if self.dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        time_input = self.skeleton.deform.expand_time(fid)
        N = time_input.shape[0]
        ast_noise = 0 if self.dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * self.smooth_term(self.iteration)   

        if self.iteration < self.opt.optimize_template_offsets_iters:
            self.skeleton.deform.use_template_offsets = False 
            self.skeleton.deform.use_skinning_weight_mlp = False 
        elif self.iteration == self.opt.optimize_template_offsets_iters:
            indices = farthest_point_sample(self.gaussians.get_xyz.detach()[None], 512)[0]
            control_nodes = self.gaussians.get_xyz.detach()[indices]
            self.skeleton.deform.update_control_nodes(control_nodes)
            self.skeleton.deform.use_template_offsets = self.dataset.use_template_offsets  
        else:
            self.skeleton.deform.use_template_offsets = self.dataset.use_template_offsets  
            self.skeleton.deform.use_skinning_weight_mlp = self.dataset.use_skinning_weight_mlp
        

        d_values = self.skeleton.step(self.gaussians.get_xyz.detach(), time_input + ast_noise, motion_mask=self.gaussians.motion_mask)
            
        
        return d_values, viewpoint_cam 
    
    def render_and_cal_loss(self, warmup_stage, d_values, viewpoint_cam):
        
        d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']
        
        d_nodes = d_values['d_nodes']
        d_scaling = torch.zeros_like(d_scaling).to(d_scaling.device) 
        if self.gaussians.use_isotropic_gs:
            d_rotation = torch.zeros_like(d_rotation).to(d_rotation.device)
        
        loss = 0
        
        # if self.opt.lambda_node_trans > 1e-8 and self.skeleton.deform.use_node_trans==True:
        #     node_trans = d_values['node_trans'] 
        #     node_trans_loss = l2_loss(node_trans, torch.zeros_like(node_trans).to(node_trans.device))
        #     loss = loss + self.opt.lambda_node_trans*node_trans_loss 
        #     self.tb_writer.add_scalar('train_skeleton/node_trans_loss', node_trans_loss.item(), self.iteration)

        # if self.opt.lambda_skinning_weight_offsets > 1e-8 and self.skeleton.deform.use_skinning_weight_mlp:
        #     skinning_weight_offsets = self.skeleton.deform.skinning_weight_offsets
        #     if skinning_weight_offsets is not None:
        #         d_xyz_norm = d_xyz.detach().norm(dim=-1)
        #         sigma = d_xyz_norm.median()/3.0
        #         weight_mask = torch.exp(-d_xyz_norm**2/sigma**2)

        #         diffs = weight_mask.unsqueeze(-1) * (skinning_weight_offsets - torch.ones_like(skinning_weight_offsets).to(self.device)/(self.joints.shape[0]-1)) 

        #         skinning_weight_offsets_loss = (diffs ** 2).mean()
        #         loss = loss + self.opt.lambda_skinning_weight_offsets * skinning_weight_offsets_loss 
        #         self.tb_writer.add_scalar('train_skeleton/skinning_weight_offsets_loss', skinning_weight_offsets_loss.item(), self.iteration)
        
        if self.skeleton.deform.use_template_offsets:
            template_offsets = self.skeleton.deform.template_offsets 
            template_offsets_loss = l2_loss(template_offsets, torch.zeros_like(template_offsets).to(template_offsets.device))
        
            lambda_template_offsets = self.opt.lambda_template_offsets
            if viewpoint_cam.uid == self.template_idx:
                lambda_template_offsets = lambda_template_offsets * 1e3
            
            loss = loss + lambda_template_offsets*template_offsets_loss 
            self.tb_writer.add_scalar('train_skeleton/template_offsets_loss', template_offsets_loss.item(), self.iteration)
            self.tb_writer.add_scalar('train_skeleton/lambda_template_offsets', lambda_template_offsets, self.iteration)
            
        
        if self.opt.lambda_deformed_node_prjection > 1e-8:
            deform_node_loss = self.cal_skeleton_loss(d_nodes, viewpoint_cam)
            
            self.all_nodes_projection_loss[viewpoint_cam.uid] = deform_node_loss.detach().item() 
            sigma = self.all_nodes_projection_loss.median()/2.0
            
            weight = torch.exp(-self.all_nodes_projection_loss[viewpoint_cam.uid]**2/(2.0*sigma**2))
            
            weight = self.opt.lambda_deformed_node_prjection*weight 
            
            
            loss = loss + weight*deform_node_loss 
            self.tb_writer.add_scalar('train_skeleton/deformed_node_prjection', deform_node_loss.item(), self.iteration)
            self.tb_writer.add_scalar('train_skeleton/lambda_deformed_node_prjection', weight, self.iteration)
        
        if self.opt.lambda_template_fixed > 1e-8:
            if viewpoint_cam.uid == self.template_idx:
                local_rotation = d_values['local_rotation']
                unit_rotation = torch.zeros_like(local_rotation).to(local_rotation.device)
                rot_bias = torch.tensor([1., 0, 0, 0]).float().to(local_rotation.device)
                unit_rotation = unit_rotation + rot_bias 
                
                template_fixed_loss = l2_loss(local_rotation, unit_rotation)
                loss = loss + self.opt.lambda_template_fixed*template_fixed_loss 
                self.tb_writer.add_scalar('train_skeleton/template_fixed_loss', template_fixed_loss.item(), self.iteration)
        
        # Render
        random_bg_color = (not self.dataset.white_background and self.opt.random_bg_color) and self.opt.gt_alpha_mask_as_scene_mask and viewpoint_cam.gt_alpha_mask is not None
        
        render_pkg_re = render(viewpoint_cam, self.gaussians, self.pipe, self.background, d_xyz, d_rotation, d_scaling, random_bg_color=random_bg_color, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=self.skeleton.d_rot_as_res)
            
        if warmup_stage:
            uid = viewpoint_cam.uid 
            d_xyz_loss = l2_loss(d_xyz, self.pretrain_deform_info['d_xyz'][uid])
            d_node_loss = l2_loss(d_nodes, self.pretrain_deform_info['d_joints'][uid])

            loss = loss + d_node_loss + d_xyz_loss  
            self.tb_writer.add_scalar('train_skeleton/d_xyz_loss', d_xyz_loss.item(), self.iteration)
            self.tb_writer.add_scalar('train_skeleton/d_node_loss', d_node_loss.item(), self.iteration)
        else:
            image = render_pkg_re["render"]
            gt_image = viewpoint_cam.original_image.cuda()
            if random_bg_color:
                gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
                gt_image = gt_alpha_mask * gt_image + (1 - gt_alpha_mask) * render_pkg_re['bg_color'][:, None, None]
            elif self.dataset.white_background and viewpoint_cam.gt_alpha_mask is not None and self.opt.gt_alpha_mask_as_scene_mask:
                gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
                gt_image = gt_alpha_mask * gt_image + (1 - gt_alpha_mask) * self.background[:, None, None]

            Ll1 = l1_loss(image, gt_image)
            loss_img = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss = loss + self.opt.lambda_rendering_image * loss_img
            self.tb_writer.add_scalar('train_skeleton/l1_loss', Ll1.item(), self.iteration)
            self.tb_writer.add_scalar('train_skeleton/loss_img', loss_img.item(), self.iteration)
        
        self.tb_writer.add_scalar('train_skeleton/total_loss', loss.item(), self.iteration)        
        return loss, render_pkg_re
    
    def optimizer_step(self, warmup_stage):
        
        lr = self.skeleton.optimizer.param_groups[0]['lr']
        self.tb_writer.add_scalar('train_skeleton/skeleton_lr', lr, self.iteration)
        
        if self.iteration < self.opt.iterations:
            
            if not warmup_stage:
                self.gaussians.optimizer.step()
                self.gaussians.update_learning_rate(self.iteration)
                self.gaussians.optimizer.zero_grad(set_to_none=True)
            
            self.skeleton.optimizer.step()
            self.skeleton.optimizer.zero_grad()
            self.skeleton.update_learning_rate(self.iteration-self.opt.skeleton_warm_up, warmup_stage)
                            
        self.skeleton.update(max(0, self.iteration - self.opt.skeleton_warm_up))
    
    def train_step(self, warmup_stage):
        self.test_network_connect()
        
        self.iter_start.record()
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.iteration % self.opt.oneupSHdegree_step == 0:
            self.gaussians.oneupSHdegree()
            
        self.select_random_cam()
        d_values, viewpoint_cam = self.deform_gaussians()
        
        loss, render_pkg_re = self.render_and_cal_loss(warmup_stage, d_values, viewpoint_cam)
        
        loss.backward()
        self.iter_end.record()
        
        self.report_and_densification(loss, warmup_stage, viewpoint_cam, render_pkg_re, d_values['d_nodes'])
        
        self.optimizer_step(warmup_stage)
        self.progress_description()
        
        
    def progress_description(self):
        
        self.progress_bar.set_description("Best PSNR={} in Iteration {}, SSIM={}, LPIPS={}, MS-SSIM={}, ALex-LPIPS={}".format('%.5f' % self.best_psnr, self.best_iteration, '%.5f' % self.best_ssim, '%.5f' % self.best_lpips, '%.5f' % self.best_ms_ssim, '%.5f' % self.best_alex_lpips))
        self.iteration += 1

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer
