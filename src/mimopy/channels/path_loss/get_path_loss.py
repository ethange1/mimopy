def get_path_loss(name):
    if name == 'NoLoss' or 'no_loss':
        from .path_loss import NoLoss
        return NoLoss()
    if name == 'FreeSpaceLoss' or 'free_space_loss':
        from .free_space import FreeSpaceLoss
        return FreeSpaceLoss()