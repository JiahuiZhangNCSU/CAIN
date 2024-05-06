import numpy as np
from scipy.spatial.transform import Rotation
from typing import Tuple
from CAIN.utils import AlphaCavity

def IRWD(P:np.ndarray, Q:np.ndarray, r_scale:float, mask:np.ndarray):
    """
    Inverse Radius weighted distance.
    Calculate the score of the alignment of two sets of aligned points P and Q.
    r_scale: the importance of the radius.
    mask: the mask of the points in P. The points with mask value 0 will be ignored.
    The number of points in P and Q should be the same.
    """
    P_crd, P_radii = P[:,:3], P[:,3]
    Q_crd, Q_radii = Q[:,:3], Q[:,3]
    # get the distance matrix.
    distances = np.linalg.norm(P_crd[:, np.newaxis, :] - Q_crd, axis=2)
    # get the shortes distance between each point in P to any point in Q.
    shortest_distances = np.min(distances, axis=0)
    nearest_idx = np.argmin(distances, axis=0) # shape: (num_P,)
    # get the difference of the inverse radii of each point of P to its nearest point of Q.
    # if the idx is not masked, inverse_r_diff = abs(1/P_radii - 1/Q_radii[nearest_idx])
    # if the idx is masked, inverse_r_diff = 1/Q_radii[nearest_idx]
    inverse_r_diff = np.zeros(P_radii.shape)
    inverse_r_diff[mask == 1] = np.abs(1/P_radii[mask == 1] - 1/Q_radii[nearest_idx[mask == 1]])
    inverse_r_diff[mask == 0] = 1/Q_radii[nearest_idx[mask == 0]]
    # use 1/r kernel.
    r_factor = np.exp(r_scale * inverse_r_diff) # shape: (num_P,)
    distances = shortest_distances # shape: (num_P,)
    root_mean_square = np.sqrt(np.mean(r_factor * distances**2)) # shape: (1,)
    return root_mean_square

def centering(P:np.ndarray, Q:np.ndarray)->np.ndarray:
    """
    Center the two sets of points P and Q.
    return the centered P according to the center of Q.
    """
    Q_crd, P_crd, P_radii = Q[:,:3], P[:,:3], P[:,3]
    center_Q = np.mean(Q_crd, axis=0)
    center_P = np.mean(P_crd, axis=0)
    P_centered_crd = P_crd - center_P + center_Q
    P_centered = np.hstack([P_centered_crd, P_radii.reshape(-1, 1)])
    return P_centered
    
def random_align(P:np.ndarray, Q:np.ndarray, mask:np.ndarray, r_scale:float=1.0, num_samples=100000):
    """
    Use exact ratation search to align the two sets of points P and Q.
    P and Q are recentered.
    The number of points in P and Q should be the same.
    """
    min_score  = IRWD(P, Q, r_scale, mask)
    best_rotation = None
    # do random search.
    for rotation in Rotation.random(num_samples):
        rotation_matrix = rotation.as_matrix()
        P_crd = np.dot(P[:,:3], rotation_matrix)
        P = np.hstack([P_crd, P[:,3].reshape(-1, 1)])
        score = IRWD(P, Q, r_scale, mask)
        if score < min_score:
            min_score = score
            best_rotation = rotation

    return min_score, best_rotation

def expand_points_to_match(P:np.ndarray, Q:np.ndarray)->Tuple[np.ndarray, np.ndarray]:
    """
    Expand P to match the number of points in Q.
    The number of points in P should be less than or equal to the number of points in Q.
    """
    num_P, num_Q = P.shape[0], Q.shape[0]
    assert num_Q >= num_P
    P_crd_center = np.mean(P[:,:3], axis=0)
    P_expanded_vector = np.append(P_crd_center, 0)
    expanded_P = np.vstack([P, np.tile(P_expanded_vector, (num_Q - num_P, 1))])
    mask = np.ones(expanded_P.shape[0])
    mask[num_P:] = 0
    return expanded_P, mask

def do_align(P:np.ndarray, Q:np.ndarray, r_scale:float)->Tuple[float, np.ndarray]:
    """
    P should be shorter than Q.
    return the best IRWD and the best rotation.
    """
    # first do the recenter.
    P = centering(P, Q)
    # then expand P to match the number of points in Q.
    P, mask = expand_points_to_match(P, Q)
    min_score, best_rotation = random_align(P, Q,  mask, r_scale)
    return min_score, best_rotation
        
def compare_two_pockets(P:str, Q:str, eps:float=1.0)->float:
    # P and Q are two csv files, convert them into numpy array.
    P = np.loadtxt(P, delimiter=',', skiprows=1, usecols=(0, 1, 2, 3))
    Q = np.loadtxt(Q, delimiter=',', skiprows=1, usecols=(0, 1, 2, 3))
    if len(P) > len(Q):
        P, Q = Q, P
        print('The number of points in P should be less than or equal to the number of points in Q. Swap P and Q.')
    
    min_score, best_rotation = do_align(P, Q, eps)
    return min_score, best_rotation