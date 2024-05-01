def get_path_loss(name: str | float | int):
    if name == 'no_loss':
        from .path_loss import NoLoss
        return NoLoss()
    if name == 'free_space_loss' or name == 'free_space':
        from .free_space import FreeSpaceLoss
        return FreeSpaceLoss()
    if isinstance(name, (float, int)):
        from .path_loss import ConstantLoss
        return ConstantLoss(name)
    raise ValueError(f"Unknown path loss model: {name}")