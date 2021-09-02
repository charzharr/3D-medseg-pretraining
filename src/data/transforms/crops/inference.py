

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
    
    def __init__(self, volume_tensor, patch_size, patch_overlap, batch_size,
                 num_classes):
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
        self.num_patches = self.grid_locations.shape[0]

        self.num_classes = parse_nonnegative_int(num_classes, 'num_classes')
        mask_shape = [num_classes] + list(self.tensor.shape)
        self.accum_tensor = torch.zeros(mask_shape, dtype=torch.float32)
        self.average_mask = torch.ones(mask_shape, dtype=torch.float32)
    

    def __len__(self):
        """ Returns the total number of batches. """
        N_patches = self.grid_locations[0]
        additional_batch = int( (N_patches % self.batch_size) > 0 )
        return N_patches // batch_size + additional_batch


    def __iter__(self):
        """ Initializes batch iterator. """
        self.batch_counter = 0
        self.patch_counter = 0
        self.num_aggregated_batches = 0

        
    def __next__(self):
        """ Gets a batch of crops. """
        if self.batch_counter > len(self):
            raise StopIteration
        idx_start = self.batch_size * self.batch_counter
        idx_exl_end = min(idx_start + self.batch_size, 
                                self.num_patches)
        batch_patches = []
        for n in range(idx_start, idx_exl_end):
            lower = self.grid_locations[n,:3]
            upper = self.grid_locations[n,3:]
            patch = self.tensor[lower[0]: upper[0],
                                lower[1]: upper[1],
                                lower[2]: upper[2]]  # no clone, no Δ to data
            batch_patches.append(patch)

        self.batch_counter += 1
        batch_tensor = torch.stack(batch_patches, dim=0).unsqueeze(1)
        batch_locations = self.grid_locations[idx_start:idx_exl_end]
        return batch_tensor, batch_locations


    def add_batch_predictions(self, batch, locations, act='none'):
        """
        Args:
            batch: BxCxHxWxD prediction tensor (C=#classes)
            act: activation for model predictions
                'none' means that batch tensor are logits
                'softmax' means apply softmax to the class dimension
                'sigmoid' means apply a sigmoid to entire tensor
        """
        N, C = batch.shape[:2]
        assert batch.ndim == 4
        assert C == self.num_classes
        assert N == locations.shape[0]
        assert locations.shape[1] == 6

        if 'softmax' in act:
            batch = batch.softmax(1)
        elif 'sigmoid' in act:
            batch = batch.sigmoid()

        dev = self.tensor.device
        for n in range(N):
            lower = locations[n, :3]
            upper = locations[n, 3:]

            self.accum_tensor[:, lower[0]:upper[0],
                                 lower[1]:upper[1],
                                 lower[2]:upper[2]] += batch[n].to(dev)
            self.average_mask[:, lower[0]:upper[0],
                                 lower[1]:upper[1],
                                 lower[2]:upper[2]] += 1
    

    def aggregate(self, act='none', ret='none', cpu=True, numpy=True):
        """
        Args:
            act: 'none', 'softmax', 'sigmoid'
            ret: 'none', 'one_hot'/'1_h'/'1_hot', 'id'/'id_map'
        """
        agg_pred = torch.div(self.accum_tensor, self.average_mask)
        
        if 'softmax' in act:
            agg_pred = agg_pred.softmax(1)
        elif 'sigmoid' in act:
            agg_pred = agg_pred.sigmoid()

        if 'one' in ret or '1' in ret:  # one hot
            zeros = torch.zeros(agg_pred.shape)
            agg_pred = agg_pred.argmax(0).unsqueeze(0)
            agg_pred = zeros.scatter_(0, agg_pred, 1).to(torch.int32)
        elif 'id' in ret:
            agg_pred = agg_pred.argmax(0).to(torch.int32)

        if cpu:
            agg_pred = agg_pred.detach().cpu()
        if numpy:
            agg_pred = agg_pred.numpy()
        return agg_pred
        
    