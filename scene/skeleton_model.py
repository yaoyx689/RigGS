import torch
import torch.nn as nn
import torch.nn.functional as F
from skeleton_utils.skeleton_warp import SkeletonWarp
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func

class SkeletonModel:
    def __init__(self, is_blender=False, d_rot_as_res=True, **kwargs):
        self.deform = SkeletonWarp(is_blender=is_blender, d_rot_as_res=d_rot_as_res, **kwargs).cuda()
        self.name = self.deform.name
        self.optimizer = None
        self.spatial_lr_scale = 1
        self.d_rot_as_res = d_rot_as_res

    @property
    def reg_loss(self):
        return self.deform.reg_loss
    
    def step(self, xyz, time_emb, **kwargs):
        return self.deform(xyz, time_emb, **kwargs)

    def train_setting(self, training_args):

        ########################
        l = []
        for group in self.deform.trainable_parameters():
            lr = training_args.deform_mlp_lr_init
            
            param_dict = {'params': group['params'], 'lr': lr, 'name': group['name']}
            l.append(param_dict)

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.deform_mlp_lr_init, lr_final=training_args.deform_mlp_lr_final, lr_delay_mult=training_args.deform_mlp_lr_delay_mult, max_steps=training_args.deform_mlp_lr_max_steps)

        if self.name == 'node':
            self.deform.as_gaussians.training_setup(training_args)

        

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "skeleton/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'skeleton.pth'))
    
    
    def save_joints(self, model_path, iteration, d_nodes, idx):
        from skeleton_utils.visualization import write_to_obj 
        out_path = os.path.join(model_path, "skeleton/iteration_{}".format(iteration))
        os.makedirs(out_path, exist_ok=True)
        write_to_obj(self.deform.nodes[:,:3].detach().cpu(), out_path + '/template_nodes.obj', self.deform.parents.cpu())
        write_to_obj(d_nodes.detach().cpu(), out_path + '/t' + str(idx).zfill(3) + '_d_nodes.obj', self.deform.parents.cpu())

        write_to_obj(self.deform.control_nodes.cpu(), out_path + '/t' + str(idx).zfill(3) + '_control_nodes.obj', self.deform.parents.cpu())

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "skeleton"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "skeleton/iteration_{}/skeleton.pth".format(loaded_iter))
        
        if os.path.exists(weights_path):
            self.deform.load_state_dict(torch.load(weights_path))
            
            self.deform.parents = self.deform.parents.int()
            
            return True
        else:
            return False

    def update_learning_rate(self, iteration, warmup_stage):
        
        for param_group in self.optimizer.param_groups:
            
            if warmup_stage:
                lr = 5e-4
            else:
                lr = self.deform_scheduler_args(iteration)
            param_group['lr'] = lr 
        
    def update(self, iteration):
        self.deform.update(iteration)
