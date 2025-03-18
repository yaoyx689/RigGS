from train_rig import TrainRig 
import datetime
import os
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import time
import torch
from gaussian_renderer import render 
import numpy as np
import dearpygui.dearpygui as dpg
import datetime
from PIL import Image
from train_gui_utils import DeformKeypoints
from minicam import MiniCam 
from cam_utils import OrbitCamera
from gaussian_renderer import quaternion_multiply
from skeleton_utils.interpolation_utils import run_interpolation 

class GUI(TrainRig):
    def __init__(self, args, dataset, opt, pipe, testing_iterations, saving_iterations) -> None:
        super().__init__(args, dataset, opt, pipe, testing_iterations, saving_iterations)

        print('gui_init')
        # For UI
        self.visualization_mode = 'RGB'
        self.gui = args.gui # enable gui
        self.W = args.W
        self.H = args.H
        self.cam = OrbitCamera(args.W, args.H, r=args.radius, fovy=args.fovy)
        self.vis_scale_const = None
        self.mode = "render"
        self.seed = "random"
        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.training = False
        self.video_speed = 1.    
        
        
        # For Animation
        self.animation_time = self.template_idx / (len(self.scene.getTrainCameras())-1)
        self.is_animation = False
        self.need_update_overlay = False
        self.buffer_overlay = None
        self.animation_trans_bias = None
        self.animation_deform_nodes = None 
        self.animation_rot_bias = None
        self.animation_scaling_bias = None
        self.animate_tool = None
        self.edited_skeleton_pose = None 
        self.cur_skeleton_pose = None 
        self.saved_skeleton_pose = None 
        self.is_training_animation_weight = False
        self.is_training_motion_analysis = False
        self.motion_genmodel = None
        self.motion_animation_d_values = None
        self.showing_overlay = True
        self.should_save_screenshot = False
        self.should_save_skeleton_pose = False 
        self.skeleton_load_path = f'./skeleton_pose/' + datetime.datetime.now().strftime('%Y-%m-%d')
        self.reference_skeleton_path = f'./select_examples/viz_skeleton/apnerf/.npz'
        self.reference_skeleton = None  # dict  {nodes: , parents,}
        self.reference_skeleton_edited_pos = None # tensor nodes 
        self.edit_reference_skeleton = False 
        self.edited_reference_skeleton_pose = None 
        self.saved_reference_skeleton_pose = None 
        self.should_save_reference_skeleton_pose = False 
        
        self.cur_deformed_nodes = self.skeleton.deform.nodes[:,:3]
        
        self.should_vis_trajectory = False
        self.screenshot_id = 0
        self.screenshot_sv_path = f'./screenshot/' + datetime.datetime.now().strftime('%Y-%m-%d')
        self.skeleton_pose_sv_path = f'./skeleton_pose/' + datetime.datetime.now().strftime('%Y-%m-%d')
        self.interpolate_pose_sv_path = f'./interpolate_pose/' + datetime.datetime.now().strftime('%Y-%m-%d')
        
        self.interpolation_poses = None 
        self.saved_key_poses = []
        self.render_interpolation_poses = False 
        
        self.traj_overlay = None
        self.vis_traj_realtime = False
        self.last_traj_overlay_type = None
        self.view_animation = True
        self.n_rings_N = 1
        # Use ARAP or Generative Model to Deform
        self.deform_mode = "arap_iterative"
        self.should_render_customized_trajectory = False
        self.should_render_customized_trajectory_spiral = False
        
        self.cur_cam = None 
        
        dpg.create_context()
        self.register_dpg()
        self.test_step()
        
        print('gui_init done')
    
    
    def update_control_point_overlay(self):
        from skimage.draw import line_aa
        import cv2 
        # should update overlay
        if self.need_update_overlay and len(self.deform_keypoints.get_kpt()) > 0 and self.animation_deform_nodes is not None:
            try:
                buffer_overlay = np.zeros_like(self.buffer_image)
                
                idx = self.deform_keypoints.get_kpt_idx()[-1]
                
                if self.edit_reference_skeleton:
                    keypoints = self.reference_skeleton_edited_pos[idx]
                else:
                    keypoints = self.animation_deform_nodes[idx]
                nodes_hom = torch.cat([keypoints, torch.ones_like(keypoints[..., :1])], dim=-1).detach().cpu()
                nodes_uv = nodes_hom @ self.cur_cam.full_proj_transform.cpu() # torch.tensor(mvp)
                nodes_uv = nodes_uv[..., :2] / nodes_uv[..., -1:]
                nodes_uv = (nodes_uv + 1) / 2 * torch.tensor([self.cur_cam.image_height, self.cur_cam.image_width])
                source_uv = nodes_uv.detach().cpu().numpy()
        
                color = [1,0,0]
                radius = int((self.H + self.W) / 2 * 0.005)
                left_top = source_uv-[radius,radius]
                right_bottom = source_uv+[radius,radius]
                cv2.rectangle(buffer_overlay, left_top.astype(int), right_bottom.astype(int), color=[float(color[0]), float(color[1]), float(color[2])], thickness=-1)
                self.buffer_overlay = buffer_overlay
            except:
                print('Async Fault in Overlay Drawing!')
                self.buffer_overlay = None

    def update_trajectory_overlay(self, gs_xyz, camera, samp_num=32, gs_num=512, thickness=1):
        if not hasattr(self, 'traj_coor') or self.traj_coor is None:
            from utils.time_utils import farthest_point_sample
            self.traj_coor = torch.zeros([0, gs_num, 4], dtype=torch.float32).cuda()
            opacity_mask = self.gaussians.get_opacity[..., 0] > .1 if self.gaussians.get_xyz.shape[0] == gs_xyz.shape[0] else torch.ones_like(gs_xyz[:, 0], dtype=torch.bool)
            masked_idx = torch.arange(0, opacity_mask.shape[0], device=opacity_mask.device)[opacity_mask]
            self.traj_idx = masked_idx[farthest_point_sample(gs_xyz[None, opacity_mask], gs_num)[0]]
            from matplotlib import cm
            self.traj_color_map = cm.get_cmap("jet")
        pts = gs_xyz[None, self.traj_idx]
        pts = torch.cat([pts, torch.ones_like(pts[..., :1])], dim=-1)
        self.traj_coor = torch.cat([self.traj_coor, pts], axis=0)
        if self.traj_coor.shape[0] > samp_num:
            self.traj_coor = self.traj_coor[-samp_num:]
        traj_uv = self.traj_coor @ camera.full_proj_transform
        traj_uv = traj_uv[..., :2] / traj_uv[..., -1:]
        traj_uv = (traj_uv + 1) / 2 * torch.tensor([camera.image_height, camera.image_width]).cuda()
        traj_uv = traj_uv.detach().cpu().numpy()

        import cv2
        colors = np.array([np.array(self.traj_color_map(i/max(1, float(gs_num - 1)))[:3]) * 255 for i in range(gs_num)], dtype=np.int32)
        alpha_img = np.zeros([camera.image_height, camera.image_width, 3], dtype=np.float32)
        traj_img = np.zeros([camera.image_height, camera.image_width, 3], dtype=np.float32)
        for i in range(gs_num):            
            alpha_img = cv2.polylines(img=alpha_img, pts=[traj_uv[:, i].astype(np.int32)], isClosed=False, color=[1, 1, 1], thickness=thickness)
            color = colors[i] / 255
            traj_img = cv2.polylines(img=traj_img, pts=[traj_uv[:, i].astype(np.int32)], isClosed=False, color=[float(color[0]), float(color[1]), float(color[2])], thickness=thickness)
        traj_img = np.concatenate([traj_img, alpha_img[..., :1]], axis=-1)
        self.traj_overlay = traj_img
       

    def update_skeleton_edges(self, camera, d_nodes, thickness=2):
        
        import cv2
        alpha_img = np.zeros([camera.image_height, camera.image_width, 3], dtype=np.float32)
        edge_img = np.zeros([camera.image_height, camera.image_width, 3], dtype=np.float32)
        parents = self.skeleton.deform.parents
        template_nodes = self.skeleton.deform.nodes[:,:3]
        nodes_hom = torch.cat([d_nodes, torch.ones_like(d_nodes[..., :1])], dim=-1)
        nodes_uv = nodes_hom @ camera.full_proj_transform
        nodes_uv = nodes_uv[..., :2] / nodes_uv[..., -1:]
        nodes_uv = (nodes_uv + 1) / 2 * torch.tensor([camera.image_height, camera.image_width]).cuda()
        cat_nodes = torch.cat([nodes_uv[1:],nodes_uv[parents[1:]]], dim=-1).reshape(-1, 2,2)
        cat_nodes = cat_nodes.detach().cpu().numpy() 
        
        from skeleton_utils.visualization import get_geometric_color
        color = [68/255, 114/255, 196/255]    
        for i in range(d_nodes.shape[0]-1): 
            alpha_img = cv2.polylines(img=alpha_img, pts=[cat_nodes[i].astype(np.int32)], isClosed=False, color=[1, 1, 1], thickness=thickness)
            
            edge_img = cv2.polylines(img=edge_img, pts=[cat_nodes[i].astype(np.int32)], isClosed=False, color=[float(color[0]), float(color[1]), float(color[2])], thickness=thickness)
        
        colors = get_geometric_color(template_nodes)
        nodes_uv = nodes_uv.detach().cpu().numpy()
        for i in range(d_nodes.shape[0]):
            alpha_img = cv2.circle(img=alpha_img, center=nodes_uv[i].astype(np.int32), radius=4, color=[1, 1, 1], thickness=-1)
            color = colors[i]
            edge_img = cv2.circle(img=edge_img, center=nodes_uv[i].astype(np.int32), radius=6, color=[float(color[0]), float(color[1]), float(color[2])], thickness=-1)
            
        edge_img = np.concatenate([edge_img, alpha_img[..., :1]], axis=-1)
        self.skeleton_edge_img = edge_img
    
    
    def update_reference_skeleton_edited_pos(self):
        parents = self.reference_skeleton['parents']
        edited_pose = None 
        edited_trans = None 
        if self.saved_reference_skeleton_pose is not None:
            edited_pose = self.saved_reference_skeleton_pose['local_rotation']
            edited_trans = self.saved_reference_skeleton_pose['global_trans']
        
        if self.edited_reference_skeleton_pose is not None:
            if edited_pose is not None:
                edited_pose = quaternion_multiply(self.edited_reference_skeleton_pose['local_rotation'], edited_pose)
                edited_trans = edited_trans + self.edited_reference_skeleton_pose['global_trans']
            else:
                edited_pose = self.edited_reference_skeleton_pose['local_rotation']
                edited_trans = self.edited_reference_skeleton_pose['global_trans']
            
        if edited_pose is None:
            d_nodes = self.reference_skeleton['nodes']
        else:
            from utils.reference_deform import chain_product_transform          
            nodes = self.reference_skeleton['nodes']
            d_nodes, _ = chain_product_transform(edited_pose, nodes, parents)
            d_nodes = d_nodes + edited_trans
        self.reference_skeleton_edited_pos = d_nodes 
        return d_nodes, parents 
    
    def update_reference_skeleton(self, camera, thickness=2):
        
        import cv2
        alpha_img = np.zeros([camera.image_height, camera.image_width, 3], dtype=np.float32)
        edge_img = np.zeros([camera.image_height, camera.image_width, 3], dtype=np.float32)
        
        d_nodes, parents = self.update_reference_skeleton_edited_pos() 
        
        
        nodes_hom = torch.cat([d_nodes, torch.ones_like(d_nodes[..., :1])], dim=-1)
        nodes_uv = nodes_hom @ camera.full_proj_transform
        nodes_uv = nodes_uv[..., :2] / nodes_uv[..., -1:]
        nodes_uv = (nodes_uv + 1) / 2 * torch.tensor([camera.image_height, camera.image_width]).cuda()
        cat_nodes = torch.cat([nodes_uv[1:],nodes_uv[parents[1:]]], dim=-1).reshape(-1, 2,2)
        cat_nodes = cat_nodes.detach().cpu().numpy() 
        
        from skeleton_utils.visualization import get_geometric_color
        colors = get_geometric_color(d_nodes)
        nodes_uv = nodes_uv.detach().cpu().numpy()
        for i in range(d_nodes.shape[0]):
            alpha_img = cv2.circle(img=alpha_img, center=nodes_uv[i].astype(np.int32), radius=4, color=[1, 1, 1], thickness=-1)
            color = colors[i]
            edge_img = cv2.circle(img=edge_img, center=nodes_uv[i].astype(np.int32), radius=4, color=[float(color[0]), float(color[1]), float(color[2])], thickness=-1)
             
        color = [237/255, 125/255, 49/255]  
        for i in range(d_nodes.shape[0]-1): 
            alpha_img = cv2.polylines(img=alpha_img, pts=[cat_nodes[i].astype(np.int32)], isClosed=False, color=[1, 1, 1], thickness=thickness)
            
            edge_img = cv2.polylines(img=edge_img, pts=[cat_nodes[i].astype(np.int32)], isClosed=False, color=[float(color[0]), float(color[1]), float(color[2])], thickness=thickness)
            
        edge_img = np.concatenate([edge_img, alpha_img[..., :1]], axis=-1)
        self.reference_skeleton_img = edge_img
    
    def render(self):
        assert self.gui
        while dpg.is_dearpygui_running():
            if self.should_vis_trajectory:
                self.draw_gs_trajectory()
                self.should_vis_trajectory = False
            if self.should_render_customized_trajectory:
                self.render_customized_trajectory(use_spiral=self.should_render_customized_trajectory_spiral)
            
            self.test_step()

            dpg.render_dearpygui_frame()
            
    
    def animation_initialize(self, use_traj=True):
        from lap_deform import LapDeform
        gaussians = self.skeleton.deform.as_gaussians
        fid = torch.tensor(self.animation_time).cuda().float()
        time_input = fid.unsqueeze(0).expand(gaussians.get_xyz.shape[0], -1)
        values = self.skeleton.deform.node_deform(t=time_input)
        trans = values['d_xyz']
        pcl = gaussians.get_xyz + trans

        if use_traj:
            print('Trajectory analysis!')
            t_samp_num = 16
            t_samp = torch.linspace(0, 1, t_samp_num).cuda().float()
            time_input = t_samp[None, :, None].expand(gaussians.get_xyz.shape[0], -1, 1)
            trajectory = self.skeleton.deform.node_deform(t=time_input)['d_xyz'] + gaussians.get_xyz[:, None]
        else:
            trajectory = None

        self.animate_init_values = values
        self.animate_tool = LapDeform(init_pcl=pcl, K=0, trajectory=trajectory, node_radius=self.skeleton.deform.node_radius.detach())
        self.keypoint_idxs = []
        self.keypoint_3ds = []
        self.keypoint_labels = []
        self.keypoint_3ds_delta = []
        self.keypoint_idxs_to_drag = []
        self.deform_keypoints = DeformKeypoints()
        self.animation_trans_bias = None
        self.animation_rot_bias = None
        self.buffer_overlay = None
        print('Initialize Animation Model with %d control nodes' % len(pcl))
    
    
    
    def update_skeleton_pose_by_rotation(self, axis, angle, translation=torch.zeros(1, 3)):
        from scipy.spatial.transform import Rotation
        axis_angle = (axis * angle).cpu().numpy()
        r = Rotation.from_rotvec(axis_angle)
        q = r.as_quat()
        
        if self.edit_reference_skeleton:
            if self.edited_reference_skeleton_pose is None:
                self.edited_reference_skeleton_pose = {}
                self.edited_reference_skeleton_pose['local_rotation'] = torch.zeros(self.reference_skeleton_edited_pos.shape[0], 4).to(self.device) 
                self.edited_reference_skeleton_pose['local_rotation'][:,0] = 1
                self.edited_reference_skeleton_pose['global_trans'] = torch.zeros(1, 3).to(self.device)
            node_idx = self.deform_keypoints.get_kpt_idx()[-1] 
            self.edited_reference_skeleton_pose['local_rotation'][node_idx] = torch.tensor([q[3],q[0],q[1],q[2]]).to(self.device)
            self.edited_reference_skeleton_pose['global_trans'] = translation.to(self.device)
                        
        else:
            if self.edited_skeleton_pose is None:
                self.edited_skeleton_pose = {}
                self.edited_skeleton_pose['local_rotation'] = torch.zeros(self.skeleton.deform.nodes.shape[0], 4).to(self.device) 
                self.edited_skeleton_pose['local_rotation'][:,0] = 1
                self.edited_skeleton_pose['global_trans'] = torch.zeros(1, 3).to(self.device)
            
            node_idx = self.deform_keypoints.get_kpt_idx()[-1] 
            self.edited_skeleton_pose['local_rotation'][node_idx] = torch.tensor([q[3],q[0],q[1],q[2]]).to(self.device)
            self.edited_skeleton_pose['global_trans'] = translation.to(self.device)
    
    
    def update_edited_skeleton_pose(self, edited_angle, translation=torch.zeros(1, 3)):
        fid = torch.tensor(self.animation_time).cuda().float()
        cur_cam = MiniCam(
            self.cam.pose,
            self.W,
            self.H,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
            fid = fid
        )
        
        dir = torch.tensor([0, 0, 1, 1]).cuda().float().view(1, 4)
        dir_3d = (dir @ torch.inverse(cur_cam.full_proj_transform))[0, :3]         
        dir_3d = dir_3d/dir_3d.norm()           

        
        axis = dir_3d
        angle = edited_angle 
        self.update_skeleton_pose_by_rotation(axis, angle, translation)

    
    @torch.no_grad()
    def test_step(self, specified_cam=None):

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        if not hasattr(self, 't0'):
            self.t0 = time.time()
            self.fps_of_fid = 10
        
        if self.is_animation:
            fid = torch.tensor(self.animation_time).cuda().float()
        else:
            fid = torch.remainder(torch.tensor((time.time()-self.t0) * self.fps_of_fid).float().cuda() / len(self.scene.getTrainCameras()) * self.video_speed, 1.)

        interpolate_pose_id = -1
        if self.render_interpolation_poses and self.interpolation_poses is not None:
            interpolate_pose_id = int(self.interpolation_poses['num']*torch.remainder(torch.tensor((time.time()-self.t0) * self.fps_of_fid).float().cuda() / len(self.scene.getTrainCameras()) * self.video_speed, 1.))

        if self.should_save_screenshot and os.path.exists(os.path.join(self.args.model_path, 'screenshot_camera.pickle')):
            print('Use fixed camera for screenshot: ', os.path.join(self.args.model_path, 'screenshot_camera.pickle'))
            from utils.pickle_utils import load_obj
            self.cur_cam = load_obj(os.path.join(self.args.model_path, 'screenshot_camera.pickle'))
        elif specified_cam is not None:
            self.cur_cam = specified_cam
        else:
            self.cur_cam = MiniCam(
                self.cam.pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
                fid = fid
            )
        fid = self.cur_cam.fid

        if self.skeleton.name == 'node':
            
            # revise to the deformation of the skeleton 
            if 'onlySkeleton' in self.visualization_mode:
                d_rotation_bias = None
                gaussians = self.skeleton.deform.as_gaussians
                
                time_input = fid.unsqueeze(0).expand(gaussians.get_xyz.shape[0], -1)
                node_attrs = self.skeleton.deform.get_pose_info(time_input)
                
                
                if self.edited_skeleton_pose is not None:
                    node_attrs['local_rotation'] = quaternion_multiply(self.edited_skeleton_pose['local_rotation'], node_attrs['local_rotation'])
                    node_attrs['global_trans'] = node_attrs['global_trans'] + self.edited_skeleton_pose['global_trans']
                
                
                if interpolate_pose_id >=0:
                    node_attrs['local_rotation'] = self.interpolation_poses['local_rotation'][interpolate_pose_id]
                    node_attrs['global_trans'] = self.interpolation_poses['global_trans'][interpolate_pose_id][None]
                
                
                self.cur_skeleton_pose = node_attrs 
                self.cur_skeleton_pose['fid'] = fid 
            
                d_values = self.skeleton.deform.node_deformation(x=gaussians.get_xyz.detach(), node_attrs=node_attrs)
                
                self.animation_deform_nodes = d_values['d_xyz'] + gaussians.get_xyz.detach() 
                if self.motion_animation_d_values is not None:
                    for key in self.motion_animation_d_values:
                        d_values[key] = self.motion_animation_d_values[key]                                
                
                d_xyz, d_opacity, d_color = d_values['d_xyz'] * gaussians.motion_mask, d_values['d_opacity'] * gaussians.motion_mask if d_values['d_opacity'] is not None else None, d_values['d_color'] * gaussians.motion_mask if d_values['d_color'] is not None else None
                d_rotation, d_scaling = 0., 0.
                if self.view_animation and self.animation_trans_bias is not None:
                    d_xyz = d_xyz + self.animation_trans_bias
                vis_scale_const = self.vis_scale_const
                
                self.update_skeleton_edges(camera=self.cur_cam, d_nodes=d_xyz + gaussians.get_xyz)
                self.cur_deformed_nodes = d_xyz + gaussians.get_xyz
            else:
                if self.view_animation:
                    node_trans_bias = self.animation_trans_bias
                else:
                    node_trans_bias = None
                
                time_input = fid.unsqueeze(0).expand(self.gaussians.get_xyz.shape[0], -1)
                node_attrs = self.skeleton.deform.get_pose_info(time_input)
                
                
                
                if self.saved_skeleton_pose is not None:
                    node_attrs['local_rotation'] = quaternion_multiply(self.saved_skeleton_pose['local_rotation'], node_attrs['local_rotation'])
                    node_attrs['global_trans'] = node_attrs['global_trans'] + self.saved_skeleton_pose['global_trans']
                
                if self.edited_skeleton_pose is not None:
                    node_attrs['local_rotation'] = quaternion_multiply(self.edited_skeleton_pose['local_rotation'], node_attrs['local_rotation'])
                    node_attrs['global_trans'] = node_attrs['global_trans'] + self.edited_skeleton_pose['global_trans']
                
                
                if interpolate_pose_id >=0:
                    node_attrs['local_rotation'] = self.interpolation_poses['local_rotation'][interpolate_pose_id]
                    node_attrs['global_trans'] = self.interpolation_poses['global_trans'][interpolate_pose_id][None]
                    
                self.cur_skeleton_pose = node_attrs 
                self.cur_skeleton_pose['fid'] = fid 
                
                d_values = self.skeleton.deform.deform_by_pose(self.gaussians.get_xyz.detach(), node_attrs, motion_mask=self.gaussians.motion_mask)
                
                self.animation_deform_nodes = d_values['d_nodes']
                
                gaussians = self.gaussians
                d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']
                vis_scale_const = None
                d_rotation_bias = d_values['d_rotation_bias'] if 'd_rotation_bias' in d_values.keys() else None
                d_nodes = d_values['d_nodes']
                
                self.update_skeleton_edges(camera=self.cur_cam, d_nodes=d_nodes)
                self.cur_deformed_nodes = d_nodes 
        else:
            vis_scale_const = None
            d_rotation_bias = None
            if self.iteration < self.opt.warm_up:
                d_xyz, d_rotation, d_scaling, d_opacity, d_color = 0.0, 0.0, 0.0, 0.0, 0.0
                gaussians = self.gaussians
            else:
                N = self.gaussians.get_xyz.shape[0]
                time_input = fid.unsqueeze(0).expand(N, -1)
                gaussians = self.gaussians
                d_values = self.skeleton.step(self.gaussians.get_xyz.detach(), time_input)
                
                d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']
        
        if self.vis_traj_realtime:
            if 'Node' in self.visualization_mode:
                if self.last_traj_overlay_type != 'node':
                    self.traj_coor = None
                self.update_trajectory_overlay(gs_xyz=gaussians.get_xyz+d_xyz, camera=self.cur_cam, gs_num=512)
                self.last_traj_overlay_type = 'node'
            else:
                if self.last_traj_overlay_type != 'gs':
                    self.traj_coor = None
                self.update_trajectory_overlay(gs_xyz=gaussians.get_xyz+d_xyz, camera=self.cur_cam)
                self.last_traj_overlay_type = 'gs'
        
        if self.visualization_mode == 'Dynamic' or self.visualization_mode == 'Static':
            d_opacity = torch.zeros_like(self.gaussians.motion_mask)
            if self.visualization_mode == 'Dynamic':
                d_opacity[self.gaussians.motion_mask < .9] = - 1e3
            else:
                d_opacity[self.gaussians.motion_mask > .1] = - 1e3
        
        if not self.showing_overlay:
            self.buffer_overlay = None
        else:
            self.update_control_point_overlay()
            
        render_motion = "Motion" in self.visualization_mode
        if render_motion:
            vis_scale_const = self.vis_scale_const
        if type(d_rotation) is not float and gaussians._rotation.shape[0] != d_rotation.shape[0]:
            d_xyz, d_rotation, d_scaling = 0, 0, 0
            print('Async in Gaussian Switching')
            
        
        if self.mode == 'skinning':
            
            from skeleton_utils.visualization import get_color_for_skinning_weights 
            vn_idx = d_values['nn_idx']
            vn_weight = d_values['nn_weight']
            override_color = get_color_for_skinning_weights(self.gaussians.get_xyz, vn_idx=vn_idx, vn_weight=vn_weight,control_points=self.skeleton.deform.nodes.detach()[:,:3])
        else:
            override_color = None 
            
        out = render(viewpoint_camera=self.cur_cam, pc=gaussians, pipe=self.pipe, bg_color=self.background, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling, render_motion=render_motion, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=self.skeleton.d_rot_as_res, scale_const=vis_scale_const, d_rotation_bias=d_rotation_bias, override_color=override_color)

        if self.mode == "normal_dep":
            from utils.other_utils import depth2normal
            normal = depth2normal(out["depth"])
            out["normal_dep"] = (normal + 1) / 2

        if self.mode=="skinning":
            out["skinning"] = out["render"]
            
        buffer_image = out[self.mode]  # [3, H, W]
        
        ## add render reference skeleton 

        if self.should_save_screenshot:
            alpha = out['alpha']
            sv_image = torch.cat([buffer_image, alpha], dim=0).clamp(0,1).permute(1,2,0).detach().cpu().numpy()
            def save_image(image, image_dir):
                os.makedirs(image_dir, exist_ok=True)
                idx = len(os.listdir(image_dir))
                print('>>> Saving image to %s' % os.path.join(image_dir, '%05d.png'%idx))
                Image.fromarray((image * 255).astype('uint8')).save(os.path.join(image_dir, '%05d.png'%idx))
                # Save the camera of screenshot
                from utils.pickle_utils import save_obj
                save_obj(os.path.join(image_dir, '%05d_cam.pickle'% idx), self.cur_cam)
            save_image(sv_image, self.screenshot_sv_path)
            self.should_save_screenshot = False
        
        if self.should_save_skeleton_pose:
            import glob
            os.makedirs(self.skeleton_pose_sv_path, exist_ok=True)
            npz_files = glob.glob(f"{self.skeleton_pose_sv_path}/*.npz")
            idx = len(npz_files)
            save_path = os.path.join(self.skeleton_pose_sv_path, '%05d.npz'%idx)
            skeleton_pose = self.cur_skeleton_pose['local_rotation'].cpu().numpy()
            skeleton_trans = self.cur_skeleton_pose['global_trans'].cpu().numpy()
            if self.saved_skeleton_pose is not None:
                edited_skeleton_pose = self.saved_skeleton_pose['local_rotation'].cpu().numpy()
                edited_skeleton_trans = self.saved_skeleton_pose['global_trans'].cpu().numpy()
            else:
                edited_skeleton_pose = None
                edited_skeleton_trans = None
            nodes = self.cur_deformed_nodes.cpu().numpy()
            parents = self.skeleton.deform.parents.cpu().numpy()
            np.savez(save_path, pose=skeleton_pose, trans=skeleton_trans, edited_pose=edited_skeleton_pose, edited_trans=edited_skeleton_trans, fid=self.cur_skeleton_pose['fid'].cpu().numpy(), nodes=nodes,parents=parents)
            print('>>> Saving pose to %s' % os.path.join(save_path))
            alpha = out['alpha']
            sv_image = torch.cat([buffer_image, alpha], dim=0).clamp(0,1).permute(1,2,0).detach().cpu().numpy()
            Image.fromarray((sv_image * 255).astype('uint8')).save(os.path.join(self.skeleton_pose_sv_path, '%05d.png'%idx))
            self.should_save_skeleton_pose = False 
        
        
        
        if self.should_save_reference_skeleton_pose:
            import glob
            os.makedirs(self.skeleton_pose_sv_path, exist_ok=True)
            npz_files = glob.glob(f"{self.skeleton_pose_sv_path}/*.npz")
            idx = len(npz_files)
            save_path = os.path.join(self.skeleton_pose_sv_path, '%05d_reference.npz'%idx)
            edited_pose = None 
            edited_trans = None 
            if self.edited_reference_skeleton_pose is not None:
                edited_pose = self.edited_reference_skeleton_pose['local_rotation']
                edited_trans = self.edited_reference_skeleton_pose['global_trans']

            if self.saved_reference_skeleton_pose is not None:
                if edited_pose is None:
                    edited_pose = self.saved_reference_skeleton_pose['local_rotation']
                    edited_trans = self.saved_reference_skeleton_pose['global_trans']
                else:
                    edited_pose = quaternion_multiply(edited_pose, self.saved_reference_skeleton_pose['local_rotation'])
                    edited_trans = edited_trans + self.saved_reference_skeleton_pose['global_trans']
                
            
            if edited_pose is not None:
                skeleton_pose = edited_pose.cpu().numpy() 
                skeleton_trans = edited_trans.cpu().numpy()  
                nodes = self.reference_skeleton_edited_pos.cpu().numpy()
                parents = self.reference_skeleton['parents'].cpu().numpy()
                canonical_nodes = self.reference_skeleton['nodes'].cpu().numpy()
                np.savez(save_path, pose=skeleton_pose, trans=skeleton_trans, canonical_nodes=canonical_nodes, nodes=nodes,parents=parents)
                print('>>> Saving pose to %s' % os.path.join(save_path))
                alpha = out['alpha']
                sv_image = torch.cat([buffer_image, alpha], dim=0).clamp(0,1).permute(1,2,0).detach().cpu().numpy()
                Image.fromarray((sv_image * 255).astype('uint8')).save(os.path.join(self.skeleton_pose_sv_path, '%05d_reference.png'%idx))
            else:
                print('>>> No new pose need to be saved!') 
            
            self.should_save_reference_skeleton_pose = False
                
        

        if self.mode in ['depth', 'alpha']:
            buffer_image = buffer_image.repeat(3, 1, 1)
            if self.mode == 'depth':
                buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)

        buffer_image = torch.nn.functional.interpolate(
            buffer_image.unsqueeze(0),
            size=(self.H, self.W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        self.buffer_image = (
            buffer_image.permute(1, 2, 0)
            .contiguous()
            .clamp(0, 1)
            .contiguous()
            .detach()
            .cpu()
            .numpy()
        )

        buffer_image = self.buffer_image             

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        

        if self.vis_traj_realtime:
            buffer_image = buffer_image * (1 - self.traj_overlay[..., 3:]) + self.traj_overlay[..., :3] * self.traj_overlay[..., 3:]
        
        if 'Skeleton' in self.visualization_mode or 'onlySkeleton' in self.visualization_mode:
            buffer_image = buffer_image * (1 - self.skeleton_edge_img[..., 3:]) + self.skeleton_edge_img[..., :3] * self.skeleton_edge_img[..., 3:]

        if self.reference_skeleton is not None:
            self.update_reference_skeleton(self.cur_cam)
            buffer_image = buffer_image * (1 - self.reference_skeleton_img[..., 3:]) + self.reference_skeleton_img[..., :3] * self.reference_skeleton_img[..., 3:]
        
        if self.is_animation and self.buffer_overlay is not None:
            overlay_mask = self.buffer_overlay.sum(axis=-1, keepdims=True) == 0
            try:
                buffer_image = buffer_image * overlay_mask + self.buffer_overlay
            except:
                buffer_image = buffer_image
                
            
        if self.gui:
            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS FID: {fid.item()})")
            dpg.set_value(
                "_texture", buffer_image
            )  # buffer must be contiguous, else seg fault!
        return buffer_image


    
    
    def register_dpg(self):
        ### register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window
        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=600,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_setattr(sender, app_data, user_data):
                setattr(self, user_data, app_data)

            # init stuff
            with dpg.collapsing_header(label="Initialize", default_open=True):

                # seed stuff
                def callback_set_seed(sender, app_data):
                    self.seed = app_data
                    self.seed_everything()

                dpg.add_input_text(
                    label="seed",
                    default_value=self.seed,
                    on_enter=True,
                    callback=callback_set_seed,
                )

                # input stuff
                def callback_select_input(sender, app_data):
                    # only one item
                    for k, v in app_data["selections"].items():
                        dpg.set_value("_log_input", k)
                        self.load_input(v)

                    self.need_update = True

                with dpg.file_dialog(
                    directory_selector=False,
                    show=False,
                    callback=callback_select_input,
                    file_count=1,
                    tag="file_dialog_tag",
                    width=700,
                    height=400,
                ):
                    dpg.add_file_extension("Images{.jpg,.jpeg,.png}")

                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="input",
                        callback=lambda: dpg.show_item("file_dialog_tag"),
                    )
                    dpg.add_text("", tag="_log_input")

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Visualization: ")

                    def callback_vismode(sender, app_data, user_data):
                        self.visualization_mode = user_data

                    dpg.add_button(
                        label="RGB",
                        tag="_button_vis_rgb",
                        callback=callback_vismode,
                        user_data='RGB',
                    )
                    dpg.bind_item_theme("_button_vis_rgb", theme_button)

                    def callback_vis_traj_realtime():
                        self.vis_traj_realtime = not self.vis_traj_realtime
                        if not self.vis_traj_realtime:
                            self.traj_coor = None
                        print('Visualize trajectory: ', self.vis_traj_realtime)
                    dpg.add_button(
                        label="Traj",
                        tag="_button_vis_traj",
                        callback=callback_vis_traj_realtime,
                    )
                    dpg.bind_item_theme("_button_vis_traj", theme_button)

                    dpg.add_button(
                        label="MotionMask",
                        tag="_button_vis_motion_mask",
                        callback=callback_vismode,
                        user_data='MotionMask',
                    )
                    dpg.bind_item_theme("_button_vis_motion_mask", theme_button)

                    dpg.add_button(
                        label="NodeMotion",
                        tag="_button_vis_node_motion",
                        callback=callback_vismode,
                        user_data='MotionMask_Node',
                    )
                    dpg.bind_item_theme("_button_vis_node_motion", theme_button)

                    dpg.add_button(
                        label="Skeleton",
                        tag="_button_vis_skeleton",
                        callback=callback_vismode,
                        user_data='Skeleton',
                    )
                    dpg.bind_item_theme("_button_vis_skeleton", theme_button)
                    
                    dpg.add_button(
                        label="onlySkeleton",
                        tag="_button_vis_onlyskeleton",
                        callback=callback_vismode,
                        user_data='onlySkeleton',
                    )
                    dpg.bind_item_theme("_button_vis_onlyskeleton", theme_button)

                    dpg.add_button(
                        label="Dynamic",
                        tag="_button_vis_Dynamic",
                        callback=callback_vismode,
                        user_data='Dynamic',
                    )
                    dpg.bind_item_theme("_button_vis_Dynamic", theme_button)

                    dpg.add_button(
                        label="Static",
                        tag="_button_vis_Static",
                        callback=callback_vismode,
                        user_data='Static',
                    )
                    dpg.bind_item_theme("_button_vis_Static", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("Scale Const: ")
                    def callback_vis_scale_const(sender):
                        self.vis_scale_const = 10 ** dpg.get_value(sender)
                        self.need_update = True
                    dpg.add_slider_float(
                        label="Log vis_scale_const (For debugging)",
                        default_value=-3,
                        max_value=-.5,
                        min_value=-5,
                        callback=callback_vis_scale_const,
                    )

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Temporal Speed: ")
                    self.video_speed = 1.
                    def callback_speed_control(sender):
                        self.video_speed = 10 ** dpg.get_value(sender)
                        self.need_update = True
                    dpg.add_slider_float(
                        label="Play speed",
                        default_value=0.,
                        max_value=3.,
                        min_value=-3.,
                        callback=callback_speed_control,
                    )
                
                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Save: ")

                    def callback_save(sender, app_data, user_data):
                        print("\n[ITER {}] Saving Gaussians".format(self.iteration))
                        self.scene.save(self.iteration)
                        self.deform.save_weights(self.args.model_path, self.iteration)
                    dpg.add_button(
                        label="model",
                        tag="_button_save_model",
                        callback=callback_save,
                        user_data='model',
                    )
                    dpg.bind_item_theme("_button_save_model", theme_button)

                    def callback_screenshot(sender, app_data):
                        self.should_save_screenshot = True
                    dpg.add_button(
                        label="screenshot", tag="_button_screenshot", callback=callback_screenshot
                    )
                    dpg.bind_item_theme("_button_screenshot", theme_button)
                    
                    
                    

                    def callback_render_traj(sender, app_data):
                        self.should_render_customized_trajectory = True
                    dpg.add_button(
                        label="render_traj", tag="_button_render_traj", callback=callback_render_traj
                    )
                    dpg.bind_item_theme("_button_render_traj", theme_button)

                    def callback_render_traj(sender, app_data):
                        self.should_render_customized_trajectory_spiral = not self.should_render_customized_trajectory_spiral
                        if self.should_render_customized_trajectory_spiral:
                            dpg.configure_item("_button_render_traj_spiral", label="camera")
                        else:
                            dpg.configure_item("_button_render_traj_spiral", label="spiral")
                    dpg.add_button(
                        label="spiral", tag="_button_render_traj_spiral", callback=callback_render_traj
                    )
                    dpg.bind_item_theme("_button_render_traj_spiral", theme_button)
                    
                    def callback_cache_nn(sender, app_data):
                        self.deform.deform.cached_nn_weight = not self.deform.deform.cached_nn_weight
                        print(f'Cached nn weight for higher rendering speed: {self.deform.deform.cached_nn_weight}')
                    dpg.add_button(
                        label="cache_nn", tag="_button_cache_nn", callback=callback_cache_nn
                    )
                    dpg.bind_item_theme("_button_cache_nn", theme_button)

            # training stuff
            with dpg.collapsing_header(label="Train", default_open=True):
                # lr and train button
                with dpg.group(horizontal=True):
                    dpg.add_text("Train: ")

                    def callback_train(sender, app_data):
                        if self.training:
                            self.training = False
                            dpg.configure_item("_button_train", label="start")
                        else:
                            # self.prepare_train()
                            self.training = True
                            dpg.configure_item("_button_train", label="stop")

                    dpg.add_button(
                        label="start", tag="_button_train", callback=callback_train
                    )
                    dpg.bind_item_theme("_button_train", theme_button)

                    def callback_save_deform_kpt(sender, app_data):
                        from utils.pickle_utils import save_obj
                        self.deform_keypoints.t = self.animation_time
                        save_obj(path=self.args.model_path+'/deform_kpt.pickle', obj=self.deform_keypoints)
                        print('Save kpt done!')
                    dpg.add_button(
                        label="save_deform_kpt", tag="_button_save_deform_kpt", callback=callback_save_deform_kpt
                    )
                    dpg.bind_item_theme("_button_save_deform_kpt", theme_button)

                    def callback_load_deform_kpt(sender, app_data):
                        from utils.pickle_utils import load_obj
                        self.deform_keypoints = load_obj(path=self.args.model_path+'/deform_kpt.pickle')
                        self.animation_time = self.deform_keypoints.t
                        with torch.no_grad():
                            animated_pcl, quat, ani_d_scaling = self.animate_tool.deform_arap(handle_idx=self.deform_keypoints.get_kpt_idx(), handle_pos=self.deform_keypoints.get_deformed_kpt_np(), return_R=True)
                            self.animation_trans_bias = animated_pcl - self.animate_tool.init_pcl
                            self.animation_rot_bias = quat
                            self.animation_scaling_bias = ani_d_scaling
                        self.need_update_overlay = True
                        print('Load kpt done!')
                    dpg.add_button(
                        label="load_deform_kpt", tag="_button_load_deform_kpt", callback=callback_load_deform_kpt
                    )
                    dpg.bind_item_theme("_button_load_deform_kpt", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_psnr")
                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_log")

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    ("render", "depth", "alpha", "normal_dep", "skinning"),
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoV (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=np.rad2deg(self.cam.fovy),
                    callback=callback_set_fovy,
                )
            
            # animation options
            with dpg.collapsing_header(label="Motion Editing", default_open=True):
                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Motion Editing: ")
                    def callback_animation_time(sender):
                        self.animation_time = dpg.get_value(sender)
                        self.is_animation = True
                        self.need_update = True
                        # self.animation_initialize()
                    dpg.add_slider_float(
                        label="",
                        default_value=0.,
                        max_value=1.,
                        min_value=0.,
                        callback=callback_animation_time,
                    )

                with dpg.group(horizontal=True):
                    def callback_animation_mode(sender, app_data):
                        with torch.no_grad():
                            self.is_animation = not self.is_animation
                            if self.is_animation:
                                if not hasattr(self, 'animate_tool') or self.animate_tool is None:
                                    self.animation_initialize()
                    dpg.add_button(
                        label="Play",
                        tag="_button_vis_animation",
                        callback=callback_animation_mode,
                        user_data='Animation',
                    )
                    dpg.bind_item_theme("_button_vis_animation", theme_button)

                    def callback_animation_initialize(sender, app_data):
                        with torch.no_grad():
                            self.is_animation = True
                            self.animation_initialize()
                    dpg.add_button(
                        label="Init Graph",
                        tag="_button_init_graph",
                        callback=callback_animation_initialize,
                    )
                    dpg.bind_item_theme("_button_init_graph", theme_button)

                    def callback_clear_animation(sender, app_data):
                        with torch.no_grad():
                            self.is_animation = True
                            self.animation_reset()
                    dpg.add_button(
                        label="Clear Graph",
                        tag="_button_clc_animation",
                        callback=callback_clear_animation,
                    )
                    dpg.bind_item_theme("_button_clc_animation", theme_button)
                    
                    
                    def callback_reset_animation(sender, app_data):
                        with torch.no_grad():
                            self.is_animation = True
                            if self.edited_skeleton_pose is not None:
                                self.edited_skeleton_pose = None
                            self.animation_reset()
                    dpg.add_button(
                        label="Reset",
                        tag="_button_reset_pose",
                        callback=callback_reset_animation,
                    )
                    dpg.bind_item_theme("_button_reset_pose", theme_button)
                    
                    def callback_save_skeleton(sender, app_data):
                        self.should_save_skeleton_pose = True
                    dpg.add_button(
                        label="Save Pose", tag="_button_save_skeleton", callback=callback_save_skeleton
                    )
                    dpg.bind_item_theme("_button_save_skeleton", theme_button)
                    
                    
                    
                with dpg.group(horizontal=True):
                    def callback_input_skeleton_path(sender, app_data):
                        self.skeleton_load_path = app_data 
                    dpg.add_input_text(label='path', default_value=self.skeleton_load_path, callback=callback_input_skeleton_path)
                    
                    def callback_load_skeleton(sender, app_data):
                        skeleton_pose = np.load(self.skeleton_load_path, allow_pickle=True)
                        if 'edited_pose' in skeleton_pose and skeleton_pose['edited_pose'] is not None:
                            self.saved_skeleton_pose = {}
                            self.saved_skeleton_pose['local_rotation'] = torch.tensor(skeleton_pose['edited_pose']).to(self.device).float()
                            self.saved_skeleton_pose['global_trans'] = torch.tensor(skeleton_pose['edited_trans']).to(self.device).float()
                        else:
                            self.saved_skeleton_pose = None
                        if 'fid' in skeleton_pose:
                            self.animation_time = skeleton_pose['fid']
                        self.edited_skeleton_pose = None                    
                    dpg.add_button(
                        label="load_skeleton", tag="_button_load_skeleton", callback=callback_load_skeleton
                    )
                    dpg.bind_item_theme("_button_load_skeleton", theme_button)
                
                
                with dpg.group(horizontal=True):
                    def callback_reference_skeleton_path(sender, app_data):
                        self.reference_skeleton_path = app_data 
                    dpg.add_input_text(label='path', default_value=self.reference_skeleton_path, callback=callback_reference_skeleton_path)

                with dpg.group(horizontal=True):
                    def callback_load_reference_skeleton(sender, app_data):
                        skeleton = np.load(self.reference_skeleton_path)
                        self.reference_skeleton = {}
                        self.reference_skeleton['nodes'] = torch.tensor(skeleton['nodes']).to(self.device).float()
                        self.reference_skeleton['parents'] = torch.tensor(skeleton['parents']).to(self.device).int()
                        self.reference_skeleton_edited_pos = torch.tensor(skeleton['nodes']).to(self.device).float()
                    
                    dpg.add_button(
                        label="Load Reference", tag="_button_load_reference_skeleton", callback=callback_load_reference_skeleton
                    )
                    dpg.bind_item_theme("_button_load_reference_skeleton", theme_button)
                
                
                    def callback_clear_reference_skeleton(sender, app_data):
                        self.reference_skeleton = None 
                    dpg.add_button(
                        label="Clear reference", tag="_button_clear_reference_skeleton", callback=callback_clear_reference_skeleton
                    )
                    dpg.bind_item_theme("_button_clear_reference_skeleton", theme_button)
                    
                    def callback_edit_reference_skeleton(sender, app_data):
                        self.edit_reference_skeleton = not self.edit_reference_skeleton 
                        print('Edit reference skeleton: ', self.edit_reference_skeleton)
                    dpg.add_button(
                        label="Edit reference", tag="_button_edit_reference_skeleton", callback=callback_edit_reference_skeleton
                    )
                    dpg.bind_item_theme("_button_edit_reference_skeleton", theme_button)
                    
                    def callback_save_reference_skeleton(sender, app_data):
                        self.should_save_reference_skeleton_pose = True
                    dpg.add_button(
                        label="Save Pose", tag="_button_save_reference_skeleton", callback=callback_save_reference_skeleton
                    )
                    dpg.bind_item_theme("_button_save_reference_skeleton", theme_button)
                
                with dpg.group(horizontal=True):
                    dpg.add_text("X-axis: ")
                    def callback_edited_pose_x_axis(sender):
                        edited_angle = dpg.get_value(sender)
                        self.need_update = True
                        self.is_animation = True
                        self.update_skeleton_pose_by_rotation(torch.tensor([1,0,0]),edited_angle)
                    dpg.add_slider_float(
                        label="",
                        default_value=0.,
                        max_value=np.pi,
                        min_value=-np.pi,
                        callback=callback_edited_pose_x_axis,
                    )
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Y-axis: ")
                    def callback_edited_pose_y_axis(sender):
                        edited_angle = dpg.get_value(sender)
                        self.need_update = True
                        self.is_animation = True
                        self.update_skeleton_pose_by_rotation(torch.tensor([0,1,0]),edited_angle)
                    dpg.add_slider_float(
                        label="",
                        default_value=0.,
                        max_value=np.pi,
                        min_value=-np.pi,
                        callback=callback_edited_pose_y_axis,
                    )
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Z-axis: ")
                    def callback_edited_pose_z_axis(sender):
                        edited_angle = dpg.get_value(sender)
                        self.need_update = True
                        self.is_animation = True
                        self.update_skeleton_pose_by_rotation(torch.tensor([0,0,1]),edited_angle)
                    dpg.add_slider_float(
                        label="",
                        default_value=0.,
                        max_value=np.pi,
                        min_value=-np.pi,
                        callback=callback_edited_pose_z_axis,
                    )
                
                def callback_apply_pose(sender, app_data):
                    with torch.no_grad():
                        self.is_animation = True
                        if self.edit_reference_skeleton:
                            if self.edited_reference_skeleton_pose is not None:
                                if self.saved_reference_skeleton_pose is not None:
                                    self.saved_reference_skeleton_pose['local_rotation'] = quaternion_multiply(self.edited_reference_skeleton_pose['local_rotation'], self.saved_reference_skeleton_pose['local_rotation'])
                                    self.saved_reference_skeleton_pose['global_trans'] = self.saved_reference_skeleton_pose['global_trans'] + self.edited_reference_skeleton_pose['global_trans']
                                else:
                                    self.saved_reference_skeleton_pose = self.edited_reference_skeleton_pose 
                                self.edited_reference_skeleton_pose = None
                                    
                        else:
                            if self.edited_skeleton_pose is not None:
                                if self.saved_skeleton_pose is not None:
                                    self.saved_skeleton_pose['local_rotation'] = quaternion_multiply(self.edited_skeleton_pose['local_rotation'], self.saved_skeleton_pose['local_rotation'])
                                    self.saved_skeleton_pose['global_trans'] = self.saved_skeleton_pose['global_trans'] + self.edited_skeleton_pose['global_trans']
                                else:
                                    self.saved_skeleton_pose = self.edited_skeleton_pose 
                                self.edited_skeleton_pose = None 
                        # self.animation_reset()
                dpg.add_button(
                    label="Apply",
                    tag="_button_apply_angle",
                    callback=callback_apply_pose,
                )
                dpg.bind_item_theme("_button_apply_angle", theme_button)
                    

            with dpg.collapsing_header(label="Motion Interpolation", default_open=True):
                with dpg.group(horizontal=True):
                    def callback_add_key_poses(sender, app_data):
                        self.saved_key_poses.append(self.cur_skeleton_pose)
                        print('add key pose, #keypose = ', len(self.saved_key_poses))
                    dpg.add_button(
                        label="Add key pose", tag="_button_add_key_poses", callback=callback_add_key_poses
                    )
                    dpg.bind_item_theme("_button_add_key_poses", theme_button)
                    
                    def callback_clear_key_poses(sender, app_data):
                        self.saved_key_poses = [] 
                        self.render_interpolation_poses = False 
                        print('clear key pose, #keypose = ', len(self.saved_key_poses))
                    dpg.add_button(
                        label="Clear key pose", tag="_button_clear_key_poses", callback=callback_clear_key_poses
                    )
                    dpg.bind_item_theme("_button_clear_key_poses", theme_button)
                    
                    def callback_save_poses(sender, app_data):                        
                        if len(self.saved_key_poses)==0 or self.interpolation_poses is None:
                            print('No key pose or interpolated pose to save!')
                            return 
                        
                        os.makedirs(self.interpolate_pose_sv_path, exist_ok=True)
                        idx = len(os.listdir(self.interpolate_pose_sv_path))
                        save_path = os.path.join(self.interpolate_pose_sv_path, str(idx).zfill(5) + '.npz')
                        interpolate_poses = self.interpolation_poses['local_rotation'].cpu().numpy()
                        interpolate_trans = self.interpolation_poses['global_trans'].cpu().numpy()
                        np.savez(save_path, poses=interpolate_poses, global_trans=interpolate_trans)
                        print('save interpolate poses into ', save_path)
            
                    dpg.add_button(
                        label="Save poses", tag="_button_save_poses", callback=callback_save_poses
                    )
                    dpg.bind_item_theme("_button_save_poses", theme_button)
                    
                    def callback_interpolate_key_poses(sender, app_data):
                        self.interpolation_poses = run_interpolation(self.saved_key_poses, device=self.device)
                        self.render_interpolation_poses = True 
                        print('perform interpolation')
                    dpg.add_button(
                        label="Interpolation", tag="_button_interpolate_key_poses", callback=callback_interpolate_key_poses
                    )
                    dpg.bind_item_theme("_button_interpolate_key_poses", theme_button)
                    
                    
                    
                
        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            self.mouse_loc = np.array(app_data)

        def callback_keypoint_drag(sender, app_data):
            if not self.is_animation:
                print("Please switch to animation mode!")
                return
            if not dpg.is_item_focused("_primary_window"):
                return
            if len(self.deform_keypoints.get_kpt()) == 0:
                return
            
            fid = torch.tensor(self.animation_time).cuda().float()
            cur_cam = MiniCam(
                self.cam.pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
                fid = fid
            )  
                
            time_input = self.skeleton.deform.expand_time(fid)
            # Pick the closest node as the keypoint
            node_attrs = self.skeleton.deform.get_pose_info(time_input)

            node_trans = self.skeleton.deform.node_deformation(self.skeleton.deform.nodes[:,:3], node_attrs)['d_xyz']
            nodes = self.skeleton.deform.nodes[..., :3] + node_trans
            
            idx = self.deform_keypoints.keypoints_idx_list[-1]
            parent_idx = self.skeleton.deform.parents[idx]
            translation = torch.tensor([0,0,0])
            angle = 0 
            moving_point_uv = np.array([int(self.mouse_loc[0]), int(self.mouse_loc[1])])
            if parent_idx < 0:                
                dx = app_data[1]
                dy = app_data[2]
                delta = 0.001 * self.cam.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, 0])
                translation = torch.tensor(delta).float()
            else:
                parent_point = nodes[parent_idx]
                children_point = nodes[self.deform_keypoints.keypoints_idx_list[-1]]
                
                parent_point = torch.tensor([parent_point[0], parent_point[1], parent_point[2], 1]).to(self.device).view(1,4)
                children_point = torch.tensor([children_point[0], children_point[1], children_point[2], 1]).to(self.device).view(1,4)
                
                points = torch.cat([parent_point, children_point], dim=0)
                point_uv = (points @ cur_cam.full_proj_transform)
                point_uv = point_uv[..., :2] / point_uv[..., -1:]
                point_uv = (point_uv + 1) / 2 * torch.tensor([cur_cam.image_height, cur_cam.image_width]).cuda()
                point_uv = point_uv.detach().cpu().numpy()
                vec_a = point_uv[1] - point_uv[0]  
                vec_b = moving_point_uv - point_uv[0]
                cos_angle = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a)*np.linalg.norm(vec_b)+1e-16)
                angle = np.arccos(cos_angle)
                
                v_cross = np.cross(vec_a, vec_b)
                if v_cross < 0:
                    angle = - angle 

            self.update_edited_skeleton_pose(angle, translation)
            self.keypoint_3ds = nodes[idx]
            self.need_update_overlay = True

        def callback_keypoint_add(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            ##### select keypoints by shift + click
            if dpg.is_key_down(dpg.mvKey_S) or dpg.is_key_down(dpg.mvKey_D) or dpg.is_key_down(dpg.mvKey_F) or dpg.is_key_down(dpg.mvKey_A) or dpg.is_key_down(dpg.mvKey_Q):
                if not self.is_animation:
                    print("Please switch to animation mode!")
                    return
                # Rendering the image with node gaussians to select nodes as keypoints
                fid = torch.tensor(self.animation_time).cuda().float()
                cur_cam = MiniCam(
                    self.cam.pose,
                    self.W,
                    self.H,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                    fid = fid
                )
                with torch.no_grad():
                    if self.edit_reference_skeleton:
                        d_nodes = self.reference_skeleton_edited_pos
                        parents = self.reference_skeleton['parents']
                    else:
                        d_nodes = self.cur_deformed_nodes
                        parents = self.skeleton.deform.parents
                    nodes_hom = torch.cat([d_nodes, torch.ones_like(d_nodes[..., :1])], dim=-1)
                    nodes_uv = nodes_hom @ cur_cam.full_proj_transform
                    nodes_uv = nodes_uv[..., :2] / nodes_uv[..., -1:]
                    nodes_uv = (nodes_uv + 1) / 2 * torch.tensor([cur_cam.image_height, cur_cam.image_width]).cuda()

                    # Project mouse_loc to points_3d
                    pw, ph = int(self.mouse_loc[0]), int(self.mouse_loc[1])

                    mouse_point = torch.tensor([pw, ph]).cuda().unsqueeze(0)
                    print('nodes_uv.shape = ', nodes_uv.shape, 'mouse_point.shape = ', mouse_point.shape)
                    keypoint_idxs = torch.tensor((nodes_uv - mouse_point).norm(dim=-1).argmin()).cuda()

                if dpg.is_key_down(dpg.mvKey_A):
                    if True:
                        keypoint_idxs = self.animate_tool.add_n_ring_nbs(keypoint_idxs, n=self.n_rings_N)
                    
                    self.animation_reset()
                    keypoint_3ds = d_nodes[keypoint_idxs]
                    self.deform_keypoints.add_kpts(keypoint_3ds, keypoint_idxs, parents, np.array([pw, ph]))
                    self.keypoint_3ds = keypoint_3ds
                    print(f'Add kpt: {self.deform_keypoints.selective_keypoints_idx_list}', keypoint_idxs)
                    
                self.need_update_overlay = True

        self.callback_keypoint_add = callback_keypoint_add
        self.callback_keypoint_drag = callback_keypoint_drag

        ### register camera handler

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True
                
        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

            dpg.add_mouse_move_handler(callback=callback_set_mouse_loc)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Right, callback=callback_keypoint_drag)
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=callback_keypoint_add)

        dpg.create_viewport(
            title="Gaussian3D",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        dpg.show_viewport()
    
    def animation_reset(self):
        self.animate_tool.reset()
        self.keypoint_idxs = []
        self.keypoint_3ds = []
        self.keypoint_labels = []
        self.keypoint_3ds_delta = []
        self.keypoint_idxs_to_drag = []
        self.deform_keypoints = DeformKeypoints()
        self.animation_trans_bias = None
        self.animation_rot_bias = None
        self.buffer_overlay = None
        self.motion_animation_d_values = None
        print('Reset Animation Model ...')
        
    
    