
import math
import numbers
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


epsilon = 1e-10   # a very small number as a value comparison threshold


class Vector3d:
    """ Abstraction for a 3D vector. 
    
    Notes
      - Rotation: visualize a pole through the center of crop (or origin).
      - Flipping: visualize a plane that mirrors involving the 2 axis that
                   aren't in the dim we are flipping about. 
    """
    
    def __init__(self, *args):
        """
        Args:
            no input: initializes a 0-vector
            one input: initializes a positional vec with endpoints @ input
                Note: input must have ndim=3 (a sequence type)
            two input: initializes a vec with a startpoint and endpoing
                Note: each input must have ndim=3 (a sequence type)
        """
        assert len(args) in (0, 1, 2)
        if len(args) == 0:
            start_point = end_point = [0, 0, 0]
        elif len(args) == 1:
            assert len(args[0]) == 3, 'Given sequence must be of length 3.'
            start_point = [0, 0, 0]
            end_point = args[0]
        else:
            assert len(args[0]) == 3, 'First sequence must be of length 3.'
            assert len(args[1]) == 3, 'Second sequence must be of length 3.'
            start_point = args[0]
            end_point = args[1]
        
        assert len(start_point) == 3
        assert len(end_point) == 3
        self._start_point = np.array(start_point).astype('float')
        self._end_point = np.array(end_point).astype('float')
        
    @property
    def spherical(self):
        """ Returns: r, θ, phi (angles in radians). """
        return self.magnitude, self.theta, self.phi
    
    @property
    def cylindrical(self):
        """ Returns: ρ, φ, z (angles in radians). """
        return self.rho, self.phi, self.end[2] - self.start[2]
    
    @property
    def magnitude(self):
        """ 'r' or 'ρ' in sperhical coordinates; the L2 euclidean norm. """
        l2 = np.sqrt(((self._end_point - self._start_point) ** 2).sum())
        return l2.tolist()
    
    @property
    def theta(self):
        """ 'θ' or polar angle in spherical coordinates. 
        Returns: θ in radians.
        """
        if self.magnitude < epsilon:
            return 0
        return math.acos((self.end[2] - self.start[2]) / self.magnitude)
    
    @property
    def phi(self):
        """ 'φ' or azimuthal angle in spherical coordinates. """
        if self.magnitude < epsilon:
            return 0
        return math.atan2((self.end[1] - self.start[1]), 
                          (self.end[0] - self.start[0]))
        
    @property
    def rho(self):
        """ 'ρ' """
        return math.sqrt((self.end[0] - self.start[0]) **2 + 
                         (self.end[1] - self.start[1]) ** 2)
    
    @property
    def start(self):
        return self._start_point
    
    @property
    def end(self):
        return self._end_point
    
    @property
    def position_vector(self):
        return Vector3d([0, 0, 0], self.end - self.start)
    
    @property
    def position_array(self):
        return Vector3d([0, 0, 0], self.end - self.start).end
    
    def rotate_dim1(self, angle, pivot_coords=None, is_degrees=True):
        if is_degrees:
            angle = math.pi * angle / 180
        
        if pivot_coords is None:
            pivot_coords = self.start
        
        # 1. Subtract pivot point
        xs = self.start[0] - pivot_coords[0]
        xe = self.end[0] - pivot_coords[0]
        ys = self.start[1] - pivot_coords[1]
        ye = self.end[1] - pivot_coords[1]
        zs = self.start[2] - pivot_coords[2]
        ze = self.end[2] - pivot_coords[2]
        
        # 2. Rotate
        new_start_point = np.array([
            xs, 
            math.cos(angle) * ys - math.sin(angle) * zs,
            math.sin(angle) * ys + math.cos(angle) * zs
        ])# .astype('int')
        new_end_point = np.array([
            xe, 
            math.cos(angle) * ye - math.sin(angle) * ze,
            math.sin(angle) * ye + math.cos(angle) * ze
        ])# .astype('int')
        
        # 3. Add
        new_start_point = [
            new_start_point[0] + pivot_coords[0],
            new_start_point[1] + pivot_coords[1],
            new_start_point[2] + pivot_coords[2]
        ]
        new_end_point = [
            new_end_point[0] + pivot_coords[0],
            new_end_point[1] + pivot_coords[1],
            new_end_point[2] + pivot_coords[2]
        ]
        return Vector3d(new_start_point, new_end_point)
    
    def rotate_dim2(self, angle, pivot_coords=None, is_degrees=True):
        if is_degrees:
            angle = math.pi * angle / 180
        
        if pivot_coords is None:
            pivot_coords = self.start
        
        # 1. Subtract pivot point
        xs = self.start[0] - pivot_coords[0]
        xe = self.end[0] - pivot_coords[0]
        ys = self.start[1] - pivot_coords[1]
        ye = self.end[1] - pivot_coords[1]
        zs = self.start[2] - pivot_coords[2]
        ze = self.end[2] - pivot_coords[2]
        
        # 2. Rotate
        new_start_point = np.array([
            math.cos(angle) * xs + math.sin(angle) * zs, 
            ys,
            - math.sin(angle) * xs + math.cos(angle) * zs
        ])# .astype('int')
        new_end_point = np.array([
            math.cos(angle) * xe + math.sin(angle) * ze, 
            ye,
            - math.sin(angle) * xe + math.cos(angle) * ze
        ])# .astype('int')
        
        # 3. Add
        new_start_point = [
            new_start_point[0] + pivot_coords[0],
            new_start_point[1] + pivot_coords[1],
            new_start_point[2] + pivot_coords[2]
        ]
        new_end_point = [
            new_end_point[0] + pivot_coords[0],
            new_end_point[1] + pivot_coords[1],
            new_end_point[2] + pivot_coords[2]
        ]
        return Vector3d(new_start_point, new_end_point)
    
    def rotate_dim3(self, angle, pivot_coords=None, is_degrees=True):
        if is_degrees:
            angle = math.pi * angle / 180
        
        if pivot_coords is None:
            pivot_coords = self.start
        
        # 1. Subtract pivot point
        xs = self.start[0] - pivot_coords[0]
        xe = self.end[0] - pivot_coords[0]
        ys = self.start[1] - pivot_coords[1]
        ye = self.end[1] - pivot_coords[1]
        zs = self.start[2] - pivot_coords[2]
        ze = self.end[2] - pivot_coords[2]
        
        # 2. Rotate
        new_start_point = np.array([
            math.cos(angle) * xs - math.sin(angle) * ys, 
            math.sin(angle) * xs + math.cos(angle) * ys,
            zs
        ])# .astype('int')
        new_end_point = np.array([
            math.cos(angle) * xe - math.sin(angle) * ye, 
            math.sin(angle) * xe + math.cos(angle) * ye,
            ze
        ])# .astype('int')
        
        # 3. Add
        new_start_point = [
            new_start_point[0] + pivot_coords[0],
            new_start_point[1] + pivot_coords[1],
            new_start_point[2] + pivot_coords[2]
        ]
        new_end_point = [
            new_end_point[0] + pivot_coords[0],
            new_end_point[1] + pivot_coords[1],
            new_end_point[2] + pivot_coords[2]
        ]
        return Vector3d(new_start_point, new_end_point)
    
    def mirror_dim1(self, mirror_coord):
        assert isinstance(mirror_coord, numbers.Number)
        new_start_point = [self.start[0] + 2 * (mirror_coord - self.start[0]), 
                           self.start[1], 
                           self.start[2]]
        new_end_point = [self.end[0] + 2 * (mirror_coord - self.end[0]), 
                         self.end[1], 
                         self.end[2]]
        return Vector3d(new_start_point, new_end_point)
    
    def mirror_dim2(self, mirror_coord):
        assert isinstance(mirror_coord, numbers.Number)
        new_start_point = [self.start[0], 
                           self.start[1] + 2 * (mirror_coord - self.start[1]),  
                           self.start[2]]
        new_end_point = [self.end[0], 
                         self.end[1] + 2 * (mirror_coord - self.end[1]), 
                         self.end[2]]
        return Vector3d(new_start_point, new_end_point)
    
    def mirror_dim3(self, mirror_coord):
        assert isinstance(mirror_coord, numbers.Number)
        new_start_point = [self.start[0], 
                           self.start[1], 
                           self.start[2] + 2 * (mirror_coord - self.start[2])]
        new_end_point = [self.end[0], 
                         self.end[1],
                         self.end[2] + 2 * (mirror_coord - self.end[2])]
        return Vector3d(new_start_point, new_end_point)
    
    def visualize(self, other_vectors=[]):
        vecs_to_plot = [self] + other_vectors
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i, vec in enumerate(vecs_to_plot):
            vlength=np.linalg.norm(vec.position_array)
            ax.quiver(vec.start[0], vec.start[1], vec.start[2],
                      vec.end[0], vec.end[1], vec.end[2], 
                      color='red', arrow_length_ratio=0.3/vlength)
        ax.set_xlim([min([v.start[0] for v in vecs_to_plot]) * 0.5, 
                     max([v.end[0] for v in vecs_to_plot]) * 1.5])
        ax.set_ylim([min([v.start[1] for v in vecs_to_plot]) * 0.5, 
                     max([v.end[1] for v in vecs_to_plot]) * 1.5])
        ax.set_zlim([min([v.start[2] for v in vecs_to_plot]) * 0.5, 
                     max([v.end[2] for v in vecs_to_plot]) * 1.5])
        plt.show()
        
    def __eq__(self, other):
        return np.allclose(self.start, other.start) and \
               np.allclose(self.end, other.end)
    
    def __repr__(self):
        string = f'Vector3d(S:{self.start}, E:{self.end})' 
        return string
    

if __name__ == '__main__':
    import IPython; IPython.embed(); 
        