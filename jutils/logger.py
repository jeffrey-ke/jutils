from torch.utils.tensorboard import SummaryWriter
# Singleton writer instance
_writer_instance = None
_iter_dict = {}
def get_id_cur_step(id):
    _iter_dict[id] = _iter_dict.setdefault(id, 0) + 1
    return _iter_dict[id]

def get_writer(log_dir="runs", with_id=None):
    """Get a singleton instance of the TensorBoard SummaryWriter."""
    global _writer_instance
    if id is not None:
        _iter_dict[id] = 0
    if _writer_instance is None:
        _writer_instance = SummaryWriter(log_dir)
    return _writer_instance

def close_writer():
    """Close the SummaryWriter (use this when training ends)."""
    global _writer_instance
    if _writer_instance is not None:
        _writer_instance.close()
        _writer_instance = None