import torch
from torch.utils.data import DataLoader
import numpy as np

def medoids_from_loader(loader):
    # Gather all points and labels from the loader
    all_points = []
    all_labels = []
    for _, p, l in loader:
        all_points.append(p)
        all_labels.append(l)
    all_points = torch.cat(all_points)
    all_labels = torch.cat(all_labels)
    
    medoids = {}
    unique_labels = torch.unique(all_labels)
    
    # For each label, compute medoid
    for label in unique_labels:
        label_mask = (all_labels == label)
        points_of_label = all_points[label_mask]
        
        # Compute pairwise distance matrix (Euclidean)
        dist_matrix = torch.cdist(points_of_label, points_of_label, p=2)
        
        # Sum distances for each point
        sum_distances = dist_matrix.sum(dim=1)
        
        # Get index of the point with minimal sum distance
        medoid_idx = torch.argmin(sum_distances)
        
        # Store medoid point (coordinates) and its index within label subset
        medoids[int(label.item())] = points_of_label[medoid_idx].numpy()
    
    return medoids
