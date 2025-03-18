import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.time_utils import get_embedder 

class DeformMLP(nn.Module):
    def __init__(self, D=8, W=256, xyz_input_ch=3, time_input_ch=1, output_ch=3, t_multires=-1, multires=4): 
        super(DeformMLP, self).__init__()
        self.D = D
        self.W = W
        self.output_ch = output_ch
        self.t_multires = t_multires
        self.skips = [D // 2]

        if t_multires > 0:
            self.embed_time_fn, time_input_ch = get_embedder(self.t_multires, time_input_ch)
        else:
            self.embed_time_fn = None
        
        if multires > 0:
            self.embed_fn, xyz_input_ch = get_embedder(multires, xyz_input_ch)
        else:
            self.embed_fn = None 
            
        self.input_ch = xyz_input_ch + time_input_ch

        
        self.linear = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                for i in range(D - 1)]
        )

        self.gaussian_warp = nn.Linear(W, output_ch)
        
        for layer in self.linear:
            nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(layer.bias)

        nn.init.normal_(self.gaussian_warp.weight, mean=0, std=1e-5)
        nn.init.zeros_(self.gaussian_warp.bias)
    
    def trainable_parameters(self):
        return [{'params': list(self.parameters()), 'name': 'offset_mlp'}]

    def forward(self, x, t, **kwargs):
        if self.embed_time_fn is not None:
            t_emb = self.embed_time_fn(t)
        else:
            t_emb = t 
        
        if self.embed_fn is not None:
            x_emb = self.embed_fn(x)
        else:
            x_emb = x 
            
        h = torch.cat([x_emb, t_emb], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, t_emb, h], -1)

        d_xyz = self.gaussian_warp(h)
        return d_xyz 
    
    def update(self, iteration, *args, **kwargs):
        if self.progressive_brand_time:
            self.embed_time_fn.update_step(iteration)
        return


class WeightMLP(nn.Module):
    def __init__(self, input_ch, output_ch, D=8, W=256, multires=10): 
        super(WeightMLP, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.skips = [D // 2]

        self.embed_fn, xyz_input_ch = get_embedder(multires, 3)
        self.input_ch = xyz_input_ch 

        self.linear = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                for i in range(D - 1)]
        )

        self.weight_predict = nn.Linear(W, output_ch)
        
    
    def trainable_parameters(self):
        return [{'params': list(self.parameters()), 'name': 'weight_mlp'}]

    def forward(self, x, **kwargs):
        x_emb = self.embed_fn(x)
        h=x_emb 
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, h], -1)

        weight = self.weight_predict(h)
        return F.sigmoid(weight)
    
    def update(self, iteration, *args, **kwargs):
        if self.progressive_brand_time:
            self.embed_time_fn.update_step(iteration)
        return


class PoseMLP(nn.Module): 
    def __init__(self, input_ch, output_ch, depth=8, hidden_dimensions=256, multires=8):
        super(PoseMLP, self).__init__()

        self.skips = [depth // 2]
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_dims = get_embedder(multires, input_ch)
            self.embed_fn = embed_fn
            input_ch = input_dims
        
        self.net = nn.ModuleList(
            [nn.Linear(input_ch, hidden_dimensions)] + [nn.Linear(hidden_dimensions, hidden_dimensions) if i not in self.skips else nn.Linear(hidden_dimensions + input_ch, hidden_dimensions) for i in range(depth-1)])

        self.rotation_predictor = nn.Linear(hidden_dimensions, output_ch)
        self.translation_predictor = nn.Linear(hidden_dimensions, 3)
        
        
    
    def forward(self, t):
        if self.embed_fn is not None:
            t_emb = self.embed_fn(t) 
        else:
            t_emb = t 
        h= t_emb + 0.
        
        for i, l in enumerate(self.net):
            h = self.net[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([t_emb, h], -1)

        rotation = self.rotation_predictor(h)
        translation = self.translation_predictor(h)
        
        return {'rotation': rotation, 'translation': translation} 