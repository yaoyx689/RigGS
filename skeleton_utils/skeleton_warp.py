# from typing import Any, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.time_utils import ControlNodeWarp, quaternion_to_matrix, matrix_to_quaternion 

from skeleton_utils.network_utils import DeformMLP, WeightMLP, PoseMLP

class SkeletonWarp(ControlNodeWarp):
    def __init__(self, is_blender, joints, parent_indices, init_pcl=None, K=3, use_hash=False, hash_time=False, enable_densify_prune=False, pred_opacity=False, pred_color=False, with_arap_loss=False, with_node_weight=False, local_frame=False, d_rot_as_res=True, skinning=False, hyper_dim=2, progressive_brand_time=False, max_d_scale=-1, is_scene_static=False,  use_skinning_weight_mlp=True, use_template_offsets=True, **kwargs):
        super().__init__(is_blender, init_pcl=init_pcl, node_num=joints.shape[0], K=K, use_hash=use_hash, hash_time=hash_time, enable_densify_prune=enable_densify_prune, pred_opacity=pred_opacity, pred_color=pred_color, with_arap_loss=with_arap_loss, with_node_weight=with_node_weight, local_frame=local_frame, d_rot_as_res=d_rot_as_res, skinning=skinning, hyper_dim=hyper_dim, progressive_brand_time=progressive_brand_time, max_d_scale=max_d_scale, is_scene_static=is_scene_static,  **kwargs)

        self.nodes.data[:,:3] = joints 
        self.parents = parent_indices.long() 
        self.nodes.requires_grad = False 
        
        self.use_skinning_weight_mlp = use_skinning_weight_mlp 
        self.use_template_offsets = use_template_offsets 

        print('use_template_offsets = ', use_template_offsets)
        print('use_skinning_weight_mlp = ', self.use_skinning_weight_mlp)

        self.skinning_weight_offsets = None  
        if self.use_skinning_weight_mlp:
            d_in = 3  
            d_out = joints.shape[0]-1
            self.skinning_weight_mlp = WeightMLP(input_ch=d_in,output_ch=d_out).to(self.nodes.device)


        self.control_nodes = nn.Parameter(torch.zeros(512, 3))
        self.detail_net = DeformMLP(xyz_input_ch=3, time_input_ch=joints.shape[0]*4, t_multires=-1)

        self.template_offsets = None   
        
        self.pose_net = PoseMLP(1,(joints.shape[0])*4).cuda()
    
    def update_control_nodes(self, nodes):
        self.control_nodes.data = nodes 
    
    def cal_nn_weight_skeleton(self, x:torch.Tensor, K=None, nodes=None, gs_kernel=True, temperature=1.):
        K = self.K if K is None else K
        # Weights of control nodes
        nodes = self.nodes[..., :3].detach() if nodes is None else nodes[..., :3]
        
        if K > 0:
            dists = self.compute_vertices_to_bones_weights(x, nodes)
            nn_dist2, nn_idxs = torch.topk(dists, K, largest=False, dim=1)
            nn_idxs = nn_idxs + 1
        else:
            nn_dist2 = self.compute_vertices_to_bones_weights(x, nodes)
            nn_idxs = torch.arange(1, nodes.shape[0], dtype=torch.long).cuda()
            nn_idxs = nn_idxs.unsqueeze(0).expand([nn_dist2.shape[0],-1])   
                
        
        if self.use_skinning_weight_mlp:
            # x: (N,3)  nn_dist_offset: (N, K)
            nn_offsets = self.skinning_weight_mlp(x[...,:3])
            if K != -1:
                nn_offsets = nn_offsets[torch.arange(nn_offsets.size(0)).unsqueeze(1), nn_idxs] 
            self.skinning_weight_offsets = nn_offsets 
            
        if gs_kernel:
                        
            nn_radius = self.node_radius[nn_idxs]  # N, K
            nn_weight = torch.exp(- nn_dist2 / (2 * nn_radius ** 2))
            
            if self.use_skinning_weight_mlp:
                nn_weight = nn_weight * self.skinning_weight_offsets

            nn_weight = nn_weight + 1e-7
            nn_weight = nn_weight / nn_weight.sum(dim=-1, keepdim=True)  # N, K
            return nn_weight, nn_dist2, nn_idxs
        else:
            nn_weight = torch.softmax(- nn_dist2 / temperature, dim=-1)
            return nn_weight, nn_dist2, nn_idxs
    
    def expand_xtime(self, t, x):
        N = x.shape[0]
        t = t.unsqueeze(0).expand(N, -1)
        return t

    def control_node_deform(self, t, detach_node=True, **kwargs):
        tshape = t.shape
        node_num = self.control_nodes.shape[0]
        if t.dim() == 3:
            assert t.shape[0] == node_num, f'Shape of t {t.shape} does not match the shape of nodes {self.control_nodes.shape}'
            nodes = self.control_nodes[:, None, ..., :3].expand(node_num, t.shape[1], 3).reshape(-1, 3)
            t = t.reshape(-1, 1)
        else:
            nodes = self.control_nodes[..., :3]
        if detach_node:
            nodes = nodes.detach()
        values = self.query_network(x=nodes, t=t, **kwargs)
        values = {key: values[key].view(*tshape[:-1], values[key].shape[-1]) if values[key] is not None else None for key in values}
        return values
    
    def update_template_offsets(self, x, rotation, K=10):
        if self.use_template_offsets:
            translation = self.detail_net(x, rotation)
            self.template_offsets = translation 
        else:
            self.template_offsets = torch.zeros_like(x).to(x.device)
        return 
    

    def expand_ctime(self, t):
        N = self.control_nodes.shape[0]
        t = t.unsqueeze(0).expand(N, -1)
        return t
    
    def forward(self, x, t, motion_mask, **kwargs):
        if t.dim() == 0:
            t = self.expand_time(t)

        node_attrs = {}
        
        
        rot_bias = torch.tensor([1., 0, 0, 0]).float().to(x.device)
        motion_attrs = self.pose_net(t[0])
        node_attrs['local_rotation'], node_attrs['global_trans'] = motion_attrs['rotation'].reshape(-1,4) + rot_bias, motion_attrs['translation']
        
        node_attrs['t'] = t[0]
        
        return_dict = self.deform_by_pose(x, node_attrs, motion_mask)
        return return_dict 
    

    
    def deform_by_pose(self, x, node_attrs, motion_mask):
        x = x.detach()
        local_rot = node_attrs['local_rotation'] 
        global_trans = node_attrs['global_trans']
            
        local_rot_matrix = quaternion_to_matrix(local_rot)
        
        # Calculate nn weights: [N, K]
        nn_weight, _, nn_idx = self.cal_nn_weight_skeleton(x=x, nodes=self.nodes)
    
        J_transformed, global_transforms = self.chain_product_transform(local_rot_matrix, self.nodes[:,:3])

        global_rotation = global_transforms[:,:3,:3]
        
        node_rot = matrix_to_quaternion(global_rotation.detach())  
        
        deformed_nodes = J_transformed + global_trans
        global_translation = global_transforms[:,:3,3] 
        
        Ax = torch.einsum('nkab,nkb->nka', global_rotation[nn_idx], (x)[:, None]) + global_translation[nn_idx] 
        Ax_avg = (Ax * nn_weight[..., None]).sum(dim=1)
        
        pose = local_rot.detach().reshape(-1)[None].expand(x.shape[0], -1)
        self.update_template_offsets(x, pose)
        
        if self.template_offsets is not None:
            Ax_avg = Ax_avg + global_trans + self.template_offsets
        else:
            Ax_avg = Ax_avg + global_trans
            
        translate = Ax_avg - x             
        translate = translate * motion_mask

        rotation = (node_rot[nn_idx] * nn_weight[..., None]).sum(dim=1)
        rotation = rotation * motion_mask
        scale = torch.zeros(x.shape[0],3).to(x.device)
        return_dict = {'d_xyz': translate, 'd_rotation': rotation, 'd_scaling': scale, 'd_nodes': deformed_nodes, 'nn_idx': nn_idx, 'nn_weight': nn_weight, 'local_rotation': node_attrs['local_rotation'], 'global_trans': global_trans}

      
        return_dict['d_opacity'] = None
        return_dict['d_color'] = None
        
        return return_dict

    def get_pose_info(self, t):
        if t.dim() == 0:
            t = self.expand_time(t)

        node_attrs = {}        
        motion_attrs = self.pose_net(t[0])
        rot_bias = torch.tensor([1., 0, 0, 0]).float().to(self.nodes.device)
        node_attrs['local_rotation'], node_attrs['global_trans'] = motion_attrs['rotation'].reshape(-1,4) + rot_bias, motion_attrs['translation']
        
        node_attrs['t'] = t[0]
        return node_attrs

    def node_deformation(self, x, node_attrs):
        
        x = x.detach()
            
        local_rot = node_attrs['local_rotation'] 
        global_trans = node_attrs['global_trans']
            
        local_rot_matrix = quaternion_to_matrix(local_rot)
        adjusted_nodes = self.nodes + 0.  
    
        J_transformed, _ = self.chain_product_transform(local_rot_matrix, adjusted_nodes[:,:3])
        
        deformed_nodes = J_transformed + global_trans
        
        return_dict = {}
        return_dict['d_xyz'] = deformed_nodes - x 
        return_dict['d_opacity'] = None
        return_dict['d_color'] = None
        return_dict['local_rotation'] = node_attrs['local_rotation']
        return return_dict 
    
    def compute_vertices_to_bones_weights(self, x:torch.Tensor, nodes:torch.Tensor):
        joints_b = nodes[1:,:3]
        joints_a = nodes[self.parents[1:].long(), :3]

        dist_to_bones = self.line_segment_distance(joints_a, joints_b, x[:,:3]).squeeze()
        vertices_to_bones = dist_to_bones 
        return vertices_to_bones
    
    def line_segment_distance(self, a, b, points, sqrt=False):
        """
        compute the distance between a point and a line segment defined by a and b
        a, b: ... x D
        points: ... x D
        """
        def sumprod(x, y, keepdim=True):
            return torch.sum(x * y, dim=-1, keepdim=keepdim)

        points = points[..., None, :]
        
        t_min = sumprod(points - a, b - a) / torch.max(sumprod(b - a, b - a), torch.tensor(1e-6, device=a.device))
        
        t_line = torch.clamp(t_min, 0.0, 1.0)

        # closest points on the line to every point
        s = a + t_line * (b - a)

        distance = sumprod(s - points, s - points, keepdim=False)
        
        if sqrt:
            distance = torch.sqrt(distance + 1e-6)

        return distance
    
    # rot_mats: (K-1, 3, 3), for children 
    # adjusted_nodes: (K, 3) [root, children]
    def chain_product_transform(self, rot_mats, adjusted_nodes):
        # (K, 3, 1)
        joints = torch.unsqueeze(adjusted_nodes, dim=-1)
        
        virtual_parents = torch.zeros_like(self.parents).to(joints.device).long() 
        virtual_parents[1:] = self.parents[1:]
        
        RJ = torch.matmul(rot_mats, joints[virtual_parents])
        # (K, 3, 1)
        local_trans = joints[virtual_parents] - RJ 
        
        # (K, 4, 4)
        transforms_mat = self.transform_mat(
        rot_mats, local_trans).reshape(joints.shape[0], 4, 4)
        
        transform_chain = [transforms_mat[0]]
        for i in range(1, self.parents.shape[0]):
            # Subtract the joint location at the rest pose
            # No need for rotation, since it's identity when at rest
            curr_res = torch.matmul(transform_chain[self.parents[i]],
                                    transforms_mat[i])
            transform_chain.append(curr_res)
        
        # (K, 4, 4)
        transforms = torch.stack(transform_chain, dim=0)
        # (K, 4, 1)
        joints_homogen = F.pad(joints, [0, 0, 0, 1], value=1)
        
        # (K, 3)
        posed_joints = torch.matmul(transforms, joints_homogen).squeeze()[:,:3]
    
        return posed_joints, transforms
    
    
    def trainable_parameters(self):
        params = []

        params += [{'params': [self._node_radius], 'name': 'nodes'}]
        params += [{'params': list(self.pose_net.parameters()), 'name': 'pose'}]
        if self.use_skinning_weight_mlp:
            params += [{'params': list(self.skinning_weight_mlp.parameters()), 'name': 'skinning_mlp'}]
        
        if self.use_template_offsets:
            params += [{'params': list(self.detail_net.parameters()), 'name': 'detail_net'}]
            
        return params 


    def transform_mat(self, R, t):
        ''' Creates a batch of transformation matrices
            Args:
                - R: Bx3x3 array of a batch of rotation matrices
                - t: Bx3x1 array of a batch of translation vectors
            Returns:
                - T: Bx4x4 Transformation matrix
        '''
        # No padding left or right, only add an extra row
        return torch.cat([F.pad(R, [0, 0, 0, 1]),
                        F.pad(t, [0, 0, 0, 1], value=1)], dim=2)