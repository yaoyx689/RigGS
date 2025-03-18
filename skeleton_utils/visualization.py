import torch 
import openmesh as om 
import numpy as np
from matplotlib import cm

def write_to_obj(points, out_path, parents=None):
    fid = open(out_path, 'w')
    for i in range(points.shape[0]):
        print('v %f %f %f'%(points[i,0], points[i,1], points[i,2]), file=fid)
    
    if parents is not None:
        for j in range(1, len(parents)):
            print('l %d %d'%(j+1, parents[j]+1), file=fid)       
    
    fid.close()
    
def get_jet():
    colormap_int = np.zeros((256, 3), np.uint8)
    for i in range(0, 256, 1):
        colormap_int[i, 2] = np.int_(np.round(cm.jet(i)[0] * 255.0))
        colormap_int[i, 1] = np.int_(np.round(cm.jet(i)[1] * 255.0))
        colormap_int[i, 0] = np.int_(np.round(cm.jet(i)[2] * 255.0))
    return colormap_int

def get_autumn():
    colormap_int = np.zeros((256, 3), np.uint8)
    for i in range(0, 256, 1):
        colormap_int[i, 0] = np.int_(np.round(cm.autumn(i)[0] * 255.0))
        colormap_int[i, 1] = np.int_(np.round(cm.autumn(i)[1] * 255.0))
        colormap_int[i, 2] = np.int_(np.round(cm.autumn(i)[2] * 255.0))
    return colormap_int


def get_cool():
    colormap_int = np.zeros((256, 3), np.uint8)
    for i in range(0, 256, 1):
        colormap_int[i, 0] = np.int_(np.round(cm.cool(i)[0] * 255.0))
        colormap_int[i, 1] = np.int_(np.round(cm.cool(i)[1] * 255.0))
        colormap_int[i, 2] = np.int_(np.round(cm.cool(i)[2] * 255.0))
    return colormap_int


def get_error_colors(errs, max_value=None, colorbar='jet'):
    if colorbar == 'jet':
        colors = get_jet()
    elif colorbar == 'cool':
        colors = get_cool()
    elif colorbar == 'autumn':    
        colors = get_autumn()
        
    
    colors = torch.from_numpy(colors).to(errs.device)
    if max_value is None:
        max_value = max(errs)
    err_colors = []
    id = (256 * errs/max_value).int() - 1
    select = id > 255
    id[select] = 255
    select = id < 0
    id[select] = 0
    id = 255 - id 
    
    err_colors = torch.index_select(colors, 0, id)/255.0
    return err_colors

def write_to_mesh(vertices, faces, outf, bounding_box_len=None, center_trans=None, colors=None):
    if center_trans is not None:
        vertices_n = vertices * bounding_box_len 
    else:
        vertices_n = vertices + 0.0
        
    if bounding_box_len is not None:
        vertices_n = vertices_n + center_trans 
    
    mesh = om.TriMesh()
    for i in range(vertices_n.shape[0]):
        v = [float(vertices_n[i,0]), float(vertices_n[i,1]), float(vertices_n[i,2])]
        vh = mesh.add_vertex(v)
        if colors is not None:
            c = [float(colors[i,0]), float(colors[i,1]), float(colors[i,2]), 1]
            mesh.set_color(vh, c)
    if faces is not None:
        for i in range(faces.shape[0]):
            f = [int(faces[i,0]), int(faces[i,1]), int(faces[i,2])]
            mesh.add_face([mesh.vertex_handle(f[0]), mesh.vertex_handle(f[1]), mesh.vertex_handle(f[2])])
    
    if colors is not None:                  
        om.write_mesh(outf, mesh, vertex_color=True, color_alpha=True)
    else:
        om.write_mesh(outf, mesh)
        
def get_geometric_color(points):
    max_p = points.max(0).values
    min_p = points.min(0).values
    scale = (max_p - min_p)
    new_points = (points - min_p)/scale 
    point_colors = (new_points * 255).int()/255.0
    select = point_colors >= 1
    point_colors[select] = 0.99
    select = point_colors < 0
    point_colors[select] = 0
    return point_colors 

def vis_blending_weight_all(filename, points, faces, vn_idx, vn_weight, control_points, bounding_box_len, center_trans):
    
    node_colors = get_geometric_color(control_points)
    vn_colors = torch.index_select(node_colors, 0, vn_idx.reshape(-1)).reshape(points.shape[0], vn_weight.shape[1], 3)
    point_colors = torch.sum(vn_weight.unsqueeze(-1)*vn_colors, dim=1)

    write_to_mesh(points.detach(), faces, filename + '_skinning_weight.ply', bounding_box_len, center_trans, point_colors.detach().cpu().numpy())
    print('write_to ', filename+ '_skinning_weight.ply', 'done')
    
 
    write_to_mesh(control_points.detach(), None, filename + '_node.ply', bounding_box_len, center_trans, node_colors.detach().cpu().numpy())
    print('write_to ', filename+ '_node.ply', 'done')
 
    
    max_w, max_index = torch.max(vn_weight, dim=1)
    max_weight_node_indices = torch.gather(vn_idx, 1, max_index.view(-1,1))
    weight_color = torch.index_select(node_colors, 0, max_weight_node_indices.squeeze())
    write_to_mesh(points.detach(), faces, filename + '_segmentation.ply', bounding_box_len, center_trans, weight_color.detach())
    print('write_to ', filename+ '_segmentation.ply', 'done')


def get_color_for_skinning_weights(points, vn_idx, vn_weight, control_points):
    node_colors = get_geometric_color(control_points)
    vn_colors = torch.index_select(node_colors, 0, vn_idx.reshape(-1)).reshape(points.shape[0], vn_weight.shape[1], 3)
    point_colors = torch.sum(vn_weight.unsqueeze(-1)*vn_colors, dim=1)
    return point_colors 