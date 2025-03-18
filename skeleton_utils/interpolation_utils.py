import  torch 
import os 

def slerp_batch(q0, q1, t):
    """
    Perform spherical linear interpolation (SLERP) between two batches of quaternions.
    
    Args:
    q0: torch.Tensor of shape (n, 4), the starting quaternions.
    q1: torch.Tensor of shape (n, 4), the ending quaternions.
    t: torch.Tensor of shape (m,), the interpolation parameters, 0 <= t <= 1.
    
    Returns:
    torch.Tensor of shape (n, m, 4), the interpolated quaternions.
    """
    n = q0.shape[0]
    m = t.shape[0]
    
    # Normalize the quaternions to ensure they are unit quaternions
    q0 = q0 / q0.norm(dim=1, keepdim=True)
    q1 = q1 / q1.norm(dim=1, keepdim=True)
    
    # Compute the dot product (cosine of the angle between the quaternions)
    dot_product = (q0 * q1).sum(dim=1)
    
    # If the dot product is negative, invert one quaternion to take the shorter path
    q1 = torch.where(dot_product.unsqueeze(1) < 0.0, -q1, q1)
    dot_product = torch.abs(dot_product)
    
    # Clamp the dot product to avoid numerical issues with acos
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    
    # Compute the angle between the quaternions
    theta_0 = torch.acos(dot_product)
    sin_theta_0 = torch.sin(theta_0)
    
    # Expand dimensions to match the shape (n, m)
    theta_0 = theta_0.unsqueeze(1).expand(n, m)
    sin_theta_0 = sin_theta_0.unsqueeze(1).expand(n, m)
    t = t.unsqueeze(0).expand(n, m)
    
    # Compute the weights for the interpolation
    w0 = torch.sin((1.0 - t) * theta_0) / sin_theta_0
    w1 = torch.sin(t * theta_0) / sin_theta_0
    
    # Handle the case when sin_theta_0 is very small (use linear interpolation)
    w0 = torch.where(sin_theta_0 > 1e-6, w0, 1.0 - t)
    w1 = torch.where(sin_theta_0 > 1e-6, w1, t)
    
    # Compute the interpolated quaternions
    q0 = q0.unsqueeze(1).expand(n, m, 4)
    q1 = q1.unsqueeze(1).expand(n, m, 4)
    q_interpolated = w0.unsqueeze(2) * q0 + w1.unsqueeze(2) * q1
    return q_interpolated / q_interpolated.norm(dim=2, keepdim=True)  # Normalize the result



def run_interpolation(key_poses, device, num_frames=20):
    if len(key_poses) <= 1:
        print('Not enough key poses: #poses=', len(key_poses))
        return None 
    t = torch.linspace(0, 1, steps=num_frames+1)[:-1].to(device)
    
    new_poses = []
    new_trans = []
    for idx in range(len(key_poses)-1):
        pose1 = key_poses[idx]['local_rotation2']
        pose2 = key_poses[idx+1]['local_rotation2']
        
        trans1 = key_poses[idx]['global_trans']
        trans2 = key_poses[idx+1]['global_trans']
        
        # (m,n,4)
        interpolate_poses = slerp_batch(pose1, pose2, t).transpose(0,1)
        t_expand = t[:,None]
        interpolate_trans = (1-t_expand)* trans1 + t_expand*trans2 
        
        new_poses.append(interpolate_poses)
        new_trans.append(interpolate_trans)
        
    new_poses = torch.cat(new_poses, dim=0)
    new_trans = torch.cat(new_trans, dim=0)
    print(new_poses.shape, new_trans.shape)
    
    return_dict = {'local_rotation2':new_poses, 'global_trans':new_trans, 'num': new_poses.shape[0]}
    return return_dict
