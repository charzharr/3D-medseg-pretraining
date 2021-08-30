

from data.transforms.crops.utils import get_grid_locations_3d


import torch

from lib.utils.parse import parse_nonnegative_int
from data.transforms.crops.utils import get_grid_locations_3d


class ChopBatchAggregate3d:
    """
    
    Example Usage:
    ```
        for volume, mask in test_set:
            CA = ChopAndAggregate(volume, (96, 96, 96), (20, 20, 20))
            
            for batch in CA:
                X = batch.cuda()
                Y = model(X)
                CA.add_batch(Y)
            
            preds = CA.aggregate()
            dice = get_dice(preds, mask)
    ```
    """
    
    def __init__(self, volume_tensor, patch_size, patch_overlap, batch_size):
        """ Does the chopping where grid locations are calculated.
        Args:
            volume_tensor: 
            patch_size:
            patch_overlap:
            batch_size
        """
        self.tensor = volume_tensor
        self.tensor_shape = volume_tensor.shape
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.batch_size = parse_nonnegative_int(batch_size, 'batch_size')
        
        # N x 6 array where each row is a crop: (d1,h1,x1,d2,h2,x2)
        self.grid_locations = get_grid_locations_3d(
            self.tensor_shape, self.patch_size, self.patch_overlap)
        
        self.accum_tensor = torch.zeros(self.tensor, dtype=self.tensor.dtype)
        self.average_mask = torch.zeros(self.tensor, dtype=self.tensor.dtype)
    
    
    def __iter__(self):
        """ Initializes batch iterator. """
    
        
    def __next__(self):
        """ Gets a batch of crops. """
    
    
    def aggregate(self):
        return torch.div(self.accum_tensor, self.average_mask + 1e-7)
        
    