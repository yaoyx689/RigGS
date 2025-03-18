import numpy as np
from collections import deque 
from skeleton_utils.mst_utils import gene_tree
import torch 
from utils.time_utils import farthest_point_sample

def bfs_path_len(idx, nodes, ends, neighbor_points):
        
    search_queue = deque()
    search_queue.append(idx)
    visited = np.zeros(nodes.shape[0])
    visited[idx] = 1
    path_len = 0 
    while search_queue:
        node = search_queue.popleft()
            
        if ends[node]:
            return path_len 

        if len(neighbor_points[node])==0:
            continue 
        
        for ni in neighbor_points[node]:
            if visited[ni]==0:
                search_queue.append(ni)
                path_len = path_len+1
                visited[ni] = 1 
            
    return -1


def bfs_get_parents_and_sort(idx, nodes, neighbor_points, select_indices, parents):
        search_queue = deque()
        search_queue.append(idx)
        visited = np.zeros((nodes.shape[0]))
        visited[idx] = 1
        path_len = 0 
        new_nodes = []
        new_parents = [-1]
        new_indices = []
        while search_queue:
            node = search_queue.popleft()
            if len(neighbor_points[node])==0:
                continue 
            
            new_nodes.append(nodes[node])
            new_indices.append(select_indices[node])
            for ni in neighbor_points[node]:
                if visited[ni]==0:
                    search_queue.append(int(ni))
                    
                    path_len = path_len+1
                    visited[ni] = 1 
                    new_parents.append(len(new_nodes)-1)
            
        
        return new_nodes, new_parents, new_indices

def adjust_arrow_dir(nodes, parents, select_indices):
    neighbor_points = [[] for _ in range(nodes.shape[0])]
    for i in range(nodes.shape[0]):
        pi = parents[i]
        if pi >= 0:
            neighbor_points[i].append(pi)
            neighbor_points[pi].append(i) 
    
    neighbor_nums = np.stack([len(neighbor_points[i]) for i in range(nodes.shape[0])])
    
    junctions = neighbor_nums>=3 
    ends = neighbor_nums==1
    
    root_score = []
    test_nn = []
    candidate_root = []
    for i in range(nodes.shape[0]):
        if junctions[i]:
            path_len = bfs_path_len(i, nodes, ends, neighbor_points)
            root_score.append(path_len) 
            test_nn.append(neighbor_nums[i])
            candidate_root.append(i)
    
    root = candidate_root[np.argmax(np.array(root_score))]
    new_points, new_parents, new_indices = bfs_get_parents_and_sort(root, nodes, neighbor_points, select_indices, parents)
    
    return new_points, new_parents, new_indices 


def update_children(parents):
    children = [[] for _ in range(len(parents))]
    
    for i in range(len(parents)):
        pi = parents[i]
        if pi >= 0:
            children[pi].append(i)
    
    children_num = np.array([len(children[i]) for i in range(len(parents))])
    return children, children_num 

def line_segment_distance(a, b, points, sqrt=True):
        """
        compute the distance between a point and a line segment defined by a and b
        a, b: ... x D
        points: ... x D
        """
        def sumprod(x, y, keepdim=True):
            return torch.sum(x * y, dim=-1, keepdim=keepdim)
                
        t_min = sumprod(points - a, b - a) / torch.max(sumprod(b - a, b - a), torch.tensor(1e-6, device=a.device))
        
        t_line = torch.clamp(t_min, 0.0, 1.0)

        # closest points on the line to every point
        s = a + t_line * (b - a)

        distance = sumprod(s - points, s - points, keepdim=False)
        
        if sqrt:
            distance = torch.sqrt(distance + 1e-6)

        return distance
    
def compute_insert_points(path, all_points, dist_thres, num_thres):
    edges = []
    edges_idxs = []
    line_pairs = deque()
    line_pairs.append([0, len(path)-1])
    while line_pairs:
        line_pair = line_pairs.popleft()
        a, b = line_pair
        
        if b-a < 2:
            edges.append([path[a],path[b]])
            edges_idxs.append([a,b])
            continue 
        
        points_a = all_points[:,path[a:a+1]]
        points_b = all_points[:,path[b:b+1]]
        points_ab = all_points[:,path[a+1:b]]

        #dists: (n_frames, [a+1,b))
        dists_to_ab = line_segment_distance(points_a, points_b, points_ab).mean(0)
        
        dists_to_a = (points_ab - points_a).norm(-1).mean(0)
        dists_to_b = (points_ab - points_b).norm(-1).mean(0)
        
        select = dists_to_a > dists_to_b 
        dists_to_a[select] = dists_to_b[select]        

        dist_score = dists_to_ab - 0.1*dists_to_a 
        if torch.max(dists_to_ab) < dist_thres:
            edges.append([path[a],path[b]])
            edges_idxs.append([a,b])
            continue 

        if len(edges_idxs) > num_thres:
            continue
        min_idx = int(torch.argmax(dist_score) + a + 1)
        line_pairs.append([a, min_idx])
        line_pairs.append([min_idx, b])

    return edges, edges_idxs


# all_points (n_frames, n_points, 3)
def compute_single_source_path_distance(all_points, path):
        
    points_a = all_points[:, path[:-1]]
    points_b = all_points[:, path[1:]]
    diff = (points_a - points_b).norm(dim=-1).mean(dim=0)
    distances = np.zeros(len(path))
    for i in range(len(path)-1):
        distances[i+1] = diff[:i+1].sum()
    return distances


# (paths: all_paths, (n,); edge_idxs: key points for all_paths, (n,))
def apply_symmetry(paths, edge_idxs, all_points, semantic_label0, length_thres=0.7, semantic_thres=0.6):
    semantic_label = semantic_label0.cpu().numpy()
    semantics = []
    for path in paths:
        semantics.append(semantic_label[path])
    
    # symmetric pair
    symmetric_pairs = [] 
    visited = np.zeros(len(paths)).astype(int)
    for path_i in range(len(paths)):
        if visited[path_i]:
            continue 
            
        best_score = 0 
        best_idx = -1
        for path_j in range(path_i+1, len(paths)):
            len_i = len(paths[path_i])
            len_j = len(paths[path_j])
            if len(edge_idxs[path_i])==1 and len(edge_idxs[path_j])==1:
                continue 
            
            length_ratio = 1.0 - abs(len_i - len_j)/(max(len_i, len_j)+1e-10)
            if length_ratio > length_thres:
                si = np.unique(semantics[path_i])
                sj = np.unique(semantics[path_j])
                
                intersection = np.intersect1d(si, sj)
                semantic_score = len(intersection)/(max(len(si), len(sj))+1e-10)
                if semantic_score > semantic_thres:                    
                    
                    score = length_ratio + semantic_score 
                    if score > best_score:
                        best_score = score 
                        best_idx = path_j
        
        if best_idx >=0:
            symmetric_pairs.append([path_i, best_idx])
            visited[best_idx] = 1
    
    
    
    # select good parts 
    # merge symmetric parts 
    for pair in symmetric_pairs:
        
        if abs(len(edge_idxs[pair[0]])-2) < abs(len(edge_idxs[pair[1]])-2):
            select_id = pair[0]
            other_id = pair[1]
        else:
            select_id = pair[1]
            other_id = pair[0]
        
        
        sort_edge_idxs = sorted(edge_idxs[select_id], key=lambda x: x[0])
        dist_select_id = compute_single_source_path_distance(all_points, paths[select_id])
        dist_other_id = compute_single_source_path_distance(all_points, paths[other_id])
        
        dist_select_id = dist_select_id/dist_select_id[-1]
        dist_other_id = dist_other_id/dist_other_id[-1]
        
        new_idxs = [] 
        for i in range(len(edge_idxs[select_id])):
            
            if i==0:
                new_start_id = 0
            else:
                start_id = sort_edge_idxs[i][0]
                new_start_id = np.argmin(abs(dist_select_id[start_id]-dist_other_id))
            
            if i==len(paths[select_id])-1:
                new_end_id = paths[other_id][-1]
            else:
                end_id = sort_edge_idxs[i][1]
                new_end_id = np.argmin(abs(dist_select_id[end_id]-dist_other_id))
            
            new_idxs.append([new_start_id, new_end_id])
            
        edge_idxs[other_id] = new_idxs 
    return edge_idxs 

def simplify_tree(all_points, parents, semantic_label, dist_thres=1.0):
    
    children, children_num = update_children(parents)
    junctions = children_num > 1 
    
    _, average_edge_len = compute_average_edge_length(all_points, parents)
    
    key_points = junctions
    
    paths = [] 
    for idx in range(len(parents)):
        pi = parents[idx]
        if pi < 0:
            continue 
        ci = children[idx]
        if len(ci)==0 or key_points[idx]:
            path = [idx]
            while 1:
                path.append(pi)
                if pi < 0:
                    break 
                if key_points[pi]:
                    break
                pi = parents[pi] 
            paths.append(path)
    
    new_parents = -2*torch.ones(len(parents)).int().to(all_points.device)
    edges = [] 
    edge_idxs = []
    
    idx = 0 
    for path in paths:
        edge, edge_idx = compute_insert_points(path, all_points, dist_thres*average_edge_len, 3)
        edges += edge 
        edge_idxs.append(edge_idx)
        idx = idx + 1
    
    if semantic_label is not None:
        edge_idxs = apply_symmetry(paths, edge_idxs, all_points, semantic_label)
    
    for i in range(len(edge_idxs)):
        for edge in edge_idxs[i]:
            new_parents[paths[i][edge[0]]] = paths[i][edge[1]]
    new_parents[0] = -1
    return new_parents




# all_points: (n_frames, n_points, 3)
def compute_average_edge_length(all_points, parents):
    parents = np.array(parents)
    select = parents>=0
    points_parents = all_points[:,parents[select]]
    points_children = all_points[:,select]
    edge_length = (points_parents - points_children).norm(dim=-1).mean(dim=0).squeeze()
    average_edge_length = edge_length.mean()
    all_edge_length = torch.zeros(len(parents)).to(all_points.device)
    all_edge_length[select] = edge_length 
    return all_edge_length, average_edge_length 


def prune_tree(nodes, all_points, parents, init_average_edge_length=None, thres=1000):
    new_parents = np.array(parents) + 0 
    children = [[] for _ in range(len(parents))]
    for i in range(len(parents)):
        pi = parents[i]
        if pi >= 0:
            children[pi].append(i)
    
    edge_length, average_edge_length = compute_average_edge_length(all_points, parents)
    if init_average_edge_length is None:
        init_average_edge_length = average_edge_length
    else:
        average_edge_length = init_average_edge_length
    
    for idx in range(len(parents)):
        # Redundant junctions with endpoints
        if len(children[idx])==0:
            pi = parents[idx]
            
            dist = 0
            prune_label = False 
            ci = idx 
            passing_nodes = []
            while pi >= -1 and len(passing_nodes) < 4:
                dist = dist + edge_length[ci]
                if len(children[pi])>1:
                    prune_label = True 
                    break 
                else:
                    # n_passing_nodes = n_passing_nodes + 1
                    passing_nodes.append(pi)
                    ci = pi
                    pi = parents[ci]
            
            if prune_label:
                new_parents[idx] = -2 
                if idx in children[parents[idx]]:
                    children[parents[idx]].remove(idx)
                for pass_i in passing_nodes:
                    new_parents[pass_i] = -2 
                    if pass_i in children[parents[pass_i]]:
                        children[parents[pass_i]].remove(pass_i)    

    
    visited = np.zeros(len(parents))
    for idx in range(len(parents)):  
        ci = len(parents) - 1 - idx 
        pi = new_parents[ci]
        if visited[ci] > 0 or visited[pi] > 0:
            continue 
        
        if pi < 0:
            continue             
        
        if len(children[ci]) <= 1:
            continue 
    
        passing_nodes = [] 
        end_junction = -2
        while len(passing_nodes) < 3:
            if pi < 0:
                break 
            if len(children[pi]) == 1:
                passing_nodes.append(pi)
                pi = new_parents[pi]
            elif len(children[pi])>1:
                end_junction = pi 
                break 
            else:
                break 
        
        if end_junction > -1:
            new_positions = nodes[ci] + nodes[end_junction]
            
            if len(passing_nodes) > 0:
                for pass_i in passing_nodes:
                    new_positions = new_positions + nodes[pass_i] 
            
            new_positions = new_positions / (2+len(passing_nodes))
            
            # end_junction
            nodes[end_junction] = new_positions 
            
            visited[end_junction] = 1 
            visited[ci] = 1
            for cci in children[ci]:
                if cci not in children[end_junction]:
                    children[end_junction].append(cci)
                    new_parents[cci] = end_junction
                
            # ci 
            new_parents[ci] = -2 
            children[ci] = []
            
            # passing nodes
            for pass_i in passing_nodes:
                p_pass_i = new_parents[pass_i]
                if pass_i in children[p_pass_i]:
                    children[p_pass_i].remove(pass_i) 
                visited[pass_i] = 1 
                    
                new_parents[pass_i] = -2 
                children[pass_i] = []
    
    return new_parents, init_average_edge_length 


def obtain_skeleton_tree(nodes, all_deformed_nodes, seg_labels):
    
    """
    compute the sparse skeleton tree 
    nodes: n x 3
    all_deformed_nodes: #frame x n x 3
    seg_labels: n x 1
    """
    
    device = nodes.device
    indices = torch.arange(0, nodes.shape[0]).to(device).int() 
    
    if nodes.shape[0] > 200: 
        sample_indices = farthest_point_sample(nodes.unsqueeze(0), 200).squeeze()
    else:
        sample_indices = torch.arange(0, nodes.shape[0]).to(device).int()
    
    select_nodes = nodes[sample_indices]
    
    select_deformed_nodes = all_deformed_nodes[:,sample_indices]
    select_deformed_nodes = select_deformed_nodes.unsqueeze(-2)
    distances = torch.norm(select_deformed_nodes - select_deformed_nodes.transpose(1,2), dim=-1)
    mean_distances = torch.mean(distances, dim=0)    
    
    parents = gene_tree(select_nodes.cpu().numpy(), mean_distances.cpu().numpy())
    
    select_indices = indices[sample_indices]
    reorder_nodes, reorder_parents, reorder_indices = adjust_arrow_dir(select_nodes, parents, select_indices)
    
    prune_parents, _ = prune_tree(reorder_nodes,all_deformed_nodes[:,reorder_indices], reorder_parents, thres=1000)

    
    if seg_labels is not None:
        simplified_parents = simplify_tree(all_deformed_nodes[:,reorder_indices], prune_parents, seg_labels[reorder_indices])
    else:
        simplified_parents = simplify_tree(all_deformed_nodes[:,reorder_indices], prune_parents, None)
    
    reorder_nodes = torch.stack(reorder_nodes, dim=0)
    new_nodes, new_parents, new_indices = adjust_arrow_dir(reorder_nodes, simplified_parents, reorder_indices)
    

    new_nodes = torch.stack(new_nodes, dim=0)
    
    new_parents = torch.tensor(new_parents).to(new_nodes.device)
    new_indices = torch.tensor(new_indices).to(new_nodes.device)
   
    return new_nodes, new_parents, new_indices 