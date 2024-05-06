import numpy as np
from scipy.spatial import Voronoi, Delaunay
from sklearn.cluster import KMeans
import heapq
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class AlphaSphere:
    crd: np.ndarray
    radius: float
    atom_ids: List[int]

def gen_random_points(min_x, max_x, min_y, max_y, min_z, max_z, n) -> np.ndarray:
    """
    Generate n random points in the box.
    """
    points = np.random.rand(n, 3)
    points[:, 0] = points[:, 0] * (max_x - min_x) + min_x
    points[:, 1] = points[:, 1] * (max_y - min_y) + min_y
    points[:, 2] = points[:, 2] * (max_z - min_z) + min_z
    return points

def PocketVolume(alpha_spheres: List[AlphaSphere], MC_points:int=100000) -> float:
    """
    Calculate the volume of a pocket.
    """
    # Use Mento Carlo method to calculate the volume of a pocket.
    # first give the bounding box of the pocket.
    min_x = min([x.crd[0] - x.radius for x in alpha_spheres])
    max_x = max([x.crd[0] + x.radius for x in alpha_spheres])
    min_y = min([x.crd[1] - x.radius for x in alpha_spheres])
    max_y = max([x.crd[1] + x.radius for x in alpha_spheres])
    min_z = min([x.crd[2] - x.radius for x in alpha_spheres])
    max_z = max([x.crd[2] + x.radius for x in alpha_spheres])
    # calculate the volume of the box.
    box_volume = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
    # generate the random points in the box, with the constraint that the point is inside the box.
    count = 0
    for i in range(MC_points):
        point = gen_random_points(min_x, max_x, min_y, max_y, min_z, max_z, 1)
        for alpha_sphere in alpha_spheres:
            if np.linalg.norm(point - alpha_sphere.crd) <= alpha_sphere.radius:
                count += 1
                break
    return count / MC_points * box_volume

class Pocket:
    AlphaSpheres: List[AlphaSphere]
    name: str = None
    res_name: str = None
    center: np.ndarray = None
    volume: float = None
  
    def __post_init__(self, name:str=None):
        if self.center is None:
            self.center = np.mean([x.crd for x in self.AlphaSpheres], axis=0)
        if name:
            # take the first three letters of name and convert to uppercase.
            res_name = name[:3].upper()
            self.name = name
            self.res_name = res_name
    
    def volume(self):
        if self.volume is None:
            self.volume = PocketVolume(self.AlphaSpheres)
        return self.volume

    def save_pdb(self):
        """
        save the pocket to a pdb file.
        The b-factor of each pseudo atom is the radius of the alpha sphere.
        The residue name is the name of the pocket.
        """
        if self.name is None:
            res_name = 'CAV'
            name ='cavity'
        else:
            res_name = self.res_name
        with open(name + '.pdb', 'w') as f:
            for i, apha_sphere in enumerate(self.AlphaSpheres):
                xyz = apha_sphere.crd
                radius = apha_sphere.radius
                f.write(f'HETATM{i+1:5}  O   {res_name} A   0    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  {radius:5.2f}           O\n')
            f.write('END')

    def save_csv(self):
        """
        save the pocket information to a csv file.
        The columns are x, y, z, radius, first_atom_id, second_atom_id, third_atom_id, last_atom_id.
        """
        if self.name is None:
            name = 'cavity'
        data = []
        for alpha_sphere in self.AlphaSpheres:
            data.append([*alpha_sphere.crd, alpha_sphere.radius, *alpha_sphere.atom_ids])
        df = pd.DataFrame(data, columns=['x', 'y', 'z', 'radius', 'first_atom_id', 'second_atom_id', 'third_atom_id', 'last_atom_id'])
        df.to_csv(name + '.csv', index=True)

def TunnelLength(alpha_spheres:List[AlphaSphere], pocket_center:np.ndarray) -> float:
    """
    Calculate the length of a tunnel.
    """
    # Calculate the length of a tunnel.
    # From the center of the pocket to the farthest alpha sphere. The path must along the nearest alpha spheres.
    # Use Dijkstra's algorithm.
    distances = np.linalg.norm(pocket_center - np.array([x.crd for x in alpha_spheres]), axis=1)
    n = len(alpha_spheres)
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(alpha_spheres[i].crd - alpha_spheres[j].crd)
            if dist < alpha_spheres[i].radius + alpha_spheres[j].radius:
                adj[i, j] = adj[j, i] = dist
    visited = [False] * n
    distances = [np.inf] * n
    distances[0] = 0
    queue = [(0, 0)]
    while queue:
        d, node = heapq.heappop(queue)
        visited[node] = True
        for i in range(n):
            if not visited[i] and adj[node, i] > 0:
                if distances[i] > d + adj[node, i]:
                    distances[i] = d + adj[node, i]
                    heapq.heappush(queue, (distances[i], i))
    return max(distances)

def openTunnelLength(alpha_spheres:List[AlphaSphere]) -> float:
    """
    Length for an open tunnel.
    """
    n = len(alpha_spheres)
    adj = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(alpha_spheres[i].crd - alpha_spheres[j].crd)
            if dist < alpha_spheres[i].radius + alpha_spheres[j].radius:
                adj[i, j] = dist 
                adj[j, i] = dist

    visited = [False] * n
    distances = [0] * n
    queue = [(0, 0)]

    while queue:
        d, node = heapq.heappop(queue)
        visited[node] = True
        for i in range(n):
            if not visited[i] and adj[node, i]>0:
                if distances[i] < d + adj[node, i]:
                    distances[i] = d + adj[node, i]
                    heapq.heappush(queue, (distances[i], i))
    
    length = np.max(distances)
    return length

class Tunnel:
    AlphaSpheres: List[AlphaSphere]
    pocket_center: np.ndarray
    is_open: bool = False
    name: str = None
    res_name: str = None
    length: float = None
    
    def __post_init__(self, name:str=None):
        if name:
            # take the first three letters of name and convert to uppercase.
            res_name = name[:3].upper()
            self.name = name
            self.res_name = res_name

    def length(self):
        if self.length is None:
            if not self.is_open:
                self.length = TunnelLength(self.AlphaSpheres, self.pocket_center)
            else:
                self.length = openTunnelLength(self.AlphaSpheres)
        return self.length

    def save_pdb(self):
        """
        save the tunnel to a pdb file.
        The b-factor of each pseudo atom is the radius of the alpha sphere.
        The residue name is the name of the tunnel.
        """
        if self.name is None:
            res_name = 'TUN'
            name ='tunnel'
        else:
            res_name = self.res_name
            name = self.name
        with open(name + '.pdb', 'w') as f:
            for i, apha_sphere in enumerate(self.AlphaSpheres):
                xyz = apha_sphere.crd
                radius = apha_sphere.radius
                f.write(f'HETATM{i+1:5}  O   {res_name} A   0    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  {radius:5.2f}           O\n')
            f.write('END')

    def save_csv(self):
        """
        save the tunnel information to a csv file.
        The columns are x, y, z, radius, first_atom_id, second_atom_id, third_atom_id, last_atom_id.
        """
        if self.name is None:
            name = 'tunnel'
        data = []
        for alpha_sphere in self.AlphaSpheres:
            data.append([*alpha_sphere.crd, alpha_sphere.radius, *alpha_sphere.atom_ids])
        df = pd.DataFrame(data, columns=['x', 'y', 'z', 'radius', 'first_atom_id', 'second_atom_id', 'third_atom_id', 'last_atom_id'])
        df.to_csv(name + '.csv', index=True)

def read_ligand_pdb(pdb_file:str) -> np.ndarray:
    """
    Read the ligand pdb file.
    return the coordinates of the ligand atoms.
    """
    ligand_atom_crds = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('HETATM') or line.startswith('ATOM'):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                ligand_atom_crds.append([x, y, z])
    return np.array(ligand_atom_crds)

def read_ligand_sdf(sdf_file:str) -> np.ndarray:
    """
    Read the ligand sdf file.
    return the coordinates of the ligand atoms.
    """
    ligand_atom_crds = []
    BioElements = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Na', 'Mg', 'Ca', 'Fe', 'Zn', 'Cu', 'Mn', 'Co', 'Ni', 'Se', 'As', 'Hg', 'Cd', 'K', 'Li', 'Ag', 'Au', 'H', 'B', 'Si', 'Sn', 'Pb', 'Sr', 'Ba', 'Al', 'Cr', 'V', 'Ti', 'Zr', 'Hf', 'Mo', 'W', 'U', 'Pt', 'Pd', 'Ru', 'Rh', 'Ir', 'Os', 'Re', 'Tc', 'Tl', 'Bi', 'Be', 'Sc', 'Y', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Ac', 'Th', 'Pa', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
    with open(sdf_file, 'r') as f:
        for line in f:
            if line[3].strip() in BioElements:
                x, y, z = line.split()[:3]
                ligand_atom_crds.append([x, y, z])
    return np.array(ligand_atom_crds)

def read_protein_pdb(pdb_file:str) -> List[Tuple[np.ndarray, int]] :
    """
    Read a protein pdb file, only focus on the CA atoms.
    return the coordinates of the CA atoms and its atom ids.
    """
    with open(pdb_file,'r') as f:
        lines = f.readlines()
        atoms = []
        for line in lines:
            if line.startswith('ATOM') or line.startswith('HETATM') and line[12:16].strip() == 'CA':
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                ca_coord = np.ndarray([x, y, z])
                atom_id  = int(line[6:11])
                atoms.append((ca_coord, atom_id))
    return atoms

def read_protein_pdb_aa(pdb_file:str) -> List[Tuple[np.ndarray, int]] :
    """
    Read a protein pdb file, all atoms version.
    return the coordinates of all atoms and its atom ids.
    """
    with open(pdb_file,'r') as f:
        lines = f.readlines()
        atoms = []
        for line in lines:
            # Don't include the hydrogen atoms.
            if line.startswith('ATOM') or line.startswith('HETATM') and line[12:16].strip() != 'H':
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                ca_coord = np.ndarray([x, y, z])
                atom_id  = int(line[6:11])
                atoms.append((ca_coord, atom_id))
    return atoms

def get_alpha_spheres(atoms:List[Tuple[np.ndarray, int]], min_radius, max_radius) -> List[AlphaSphere]:
    """
    Given the coordinates of the atoms, return the alpha spheres.
    atoms: a list of tuples, each tuple contains the coordinates of an atom and its atom id.
    min_radius: the minimum radius of the alpha sphere to be considered.
    max_radius: the maximum radius of the alpha sphere to be considered.
    """
    ca_coords = np.array([x[0] for x in atoms])
    raw_alpha_lining_idx = Delaunay(ca_coords).simplices
    raw_alpha_lining_xyz = np.take(ca_coords, raw_alpha_lining_idx[:,0].flatten(), axis=0)  
    raw_alpha_xyz = Voronoi(ca_coords).vertices
    raw_alpha_sphere_radii = np.linalg.norm(raw_alpha_lining_xyz - raw_alpha_xyz, axis=1)
    filtered_alpha_idx = np.where(np.logical_and(min_radius <= raw_alpha_sphere_radii,
                                                     raw_alpha_sphere_radii <= max_radius))[0]
    alpha_radii = np.take(raw_alpha_sphere_radii, filtered_alpha_idx)
    alpha_lining = np.take(raw_alpha_lining_idx, filtered_alpha_idx, axis=0)
    alpha_xyz = np.take(raw_alpha_xyz, filtered_alpha_idx, axis=0)
    alpha_spheres = []
    for i in range(len(alpha_radii)):
        alpha_spheres.append(AlphaSphere(alpha_xyz[i], alpha_radii[i], alpha_lining[i]))
    return alpha_spheres

def gen_ligand_pocket(alpha_spheres:List[AlphaSphere], ligand_crds:np.ndarray) -> Pocket:
    """
    Given the alpha spheres and the ligand coordinates, return the pocket that contains the ligand.
    """
    pocket_alpha_spheres = []
    for alpha_sphere in alpha_spheres:
        dist = np.linalg.norm(alpha_sphere.crd - ligand_crds)
        if dist < alpha_sphere.radius:
            pocket_alpha_spheres.append(alpha_sphere)
    pocket = Pocket(pocket_alpha_spheres)
    return pocket

def gen_ligand_tunnel(alpha_spheres:List[AlphaSphere], pocket:Pocket, d:float, is_oppen:bool) -> Tunnel:
    """
    Given the alpha spheres and the ligand coordinates, return the tunnel that contains the ligand.
    d: the maximum distance between two alpha spheres to be considered as a tunnel.
    """
    tunnel_alpha_spheres = pocket.AlphaSpheres
    flag = True
    while flag:
        for i, alpha_sphere in enumerate(alpha_spheres):
            # calculate the min distance between the alpha sphere and the pocket.
            min_distance = np.min(np.linalg.norm(alpha_sphere.crd - np.array([x.crd for x in pocket.AlphaSpheres]), axis=1))
            if min_distance <= d:
                tunnel_alpha_spheres.append(alpha_sphere)
                alpha_spheres.pop(i)
            else:
                flag = False
    tunnel = Tunnel(tunnel_alpha_spheres, pocket.center, is_oppen)
    return tunnel

def clustering_alpha_spheres(alpha_spheres:List[AlphaSphere])->List[Pocket]:
    """
    Use the t-SNE algorithm to cluster the alpha spheres.
    Need to judge wich k is the best.
    """
    best_score = -1
    best_k = 0
    # Maximum number of clusters is 20.
    for k in range(2, 21):
        kmeans = KMeans(n_clusters=k, random_state=0).fit([x.crd for x in alpha_spheres])
        score = kmeans.score([x.crd for x in alpha_spheres])
        if score > best_score:
            best_score = score
            best_k = k
            result_kmeans = kmeans
    pockets = []
    for i in range(best_k):
        pocket_alpha_spheres = [alpha_spheres[j] for j in range(len(alpha_spheres)) if result_kmeans.labels_[j] == i]
        # Only consider the pocket that has more than 2 alpha spheres.
        if len(pocket_alpha_spheres) > 5:
            pocket = Pocket(pocket_alpha_spheres, name=f'cavity_{i}')
            pockets.append(pocket)
    return pockets

class AlphaCavity:
    """
    All the alpha cavities in a protein.
    """
    AlphaSpheres: List[AlphaSphere] = None
    ligand_crds: np.ndarray = None
    pockets: List[Pocket] = None
    ligand_pocket: Pocket = None
    ligand_tunnel: Tunnel = None
    max_radius: float = 6.0
    min_radius: float = 3.0
    tunnel_d: float = 2.8
    name: str = None
    aa: bool = False # all atoms version
    is_open: bool = False

    def get_AlphaShperes(self, pdb_file:str):
        if self.aa:
            atoms = read_protein_pdb_aa(pdb_file)
        else:
            atoms = read_protein_pdb(pdb_file)
        self.AlphaSpheres = get_alpha_spheres(atoms, self.min_radius, self.max_radius)

    def get_ligand_crds(self, ligand_file:str):
        if ligand_file.endswith('.pdb'):
            self.ligand_crds = read_ligand_pdb(ligand_file)
        elif ligand_file.endswith('.sdf'):
            self.ligand_crds = read_ligand_sdf(ligand_file)
    
    def get_pockets_as_whole(self):
        self.pockets = [Pocket(self.AlphaSpheres)]

    def get_pockets(self):
        self.pockets = clustering_alpha_spheres(self.AlphaSpheres)

    def get_ligand_pocket(self):
        self.ligand_pocket = gen_ligand_pocket(self.AlphaSpheres, self.ligand_crds)
    
    def get_ligand_tunnel(self):
        self.ligand_tunnel = gen_ligand_tunnel(self.AlphaSpheres, self.ligand_pocket, self.tunnel_d, self.is_open)
    
    def save_pockets_csv(self):
        for pocket in self.pockets:
            pocket.save_csv()

    def save_pockets_pdb(self):
        """
        save all pockets in one pdb file, each pocket is a residue.
        The first residue id is 1.
        """
        if self.name is None:
            name = 'XXX'
        with open(name + '_pockets.pdb', 'w') as f:
            for i, pocket in enumerate(self.pockets):
                for j, alpha_sphere in enumerate(pocket.AlphaSpheres):
                    xyz = alpha_sphere.crd
                    radius = alpha_sphere.radius
                    # keep the residue id to be aligned
                    if i < 9:
                        f.write(f'HETATM{j+1:5}  O   {pocket.res_name} A   {i+1}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  {radius:5.2f}           O\n') # 3 spaces between A and residue id
                    else:
                        f.write(f'HETATM{j+1:5}  O   {pocket.res_name} A  {i+1}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  {radius:5.2f}           O\n') # 2 spaces between A and residue id
            f.write('END')

    def save_ligand_pocket_csv(self):
        self.ligand_pocket.save_csv()
    
    def save_ligand_pocket_pdb(self):
        self.ligand_pocket.save_pdb()

    def pockets_volume(self):
        """
        A list of the volumes of all pockets.
        """
        return [pocket.volume() for pocket in self.pockets]
    
    def ligand_pocket_volume(self):
        """
        The volume of the ligand pocket.
        """
        return self.ligand_pocket.volume()
    
    def ligand_tunnel_length(self):
        """
        The length of the ligand tunnel.
        """
        return self.ligand_tunnel.length()
    


    
    




    



