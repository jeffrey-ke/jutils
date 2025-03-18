from torch.utils.tensorboard import SummaryWriter

# Singleton writer instance
_writer_instance = None

def get_writer(log_dir="runs"):
    """Get a singleton instance of the TensorBoard SummaryWriter."""
    global _writer_instance
    if _writer_instance is None:
        _writer_instance = SummaryWriter(log_dir)
    return _writer_instance

def close_writer():
    """Close the SummaryWriter (use this when training ends)."""
    global _writer_instance
    if _writer_instance is not None:
        _writer_instance.close()
        _writer_instance = None