

class ImageTransform:
    """ Base class for an image transform. """
    
    def __call__(self, image, *args, **kwargs):
        self.transform(image, *args, **kwargs)
    
    def transform(self, image, *args, **kwargs):
        raise NotImplementedError()
    
    def untransform(self, image, receipt, *args, **kwargs):
        raise NotImplementedError


class 


class ImageTransformReceipt:
    def __init__(self):
        self.start_size = size
        self.end_size = None
