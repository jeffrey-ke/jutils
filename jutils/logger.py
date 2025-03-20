from torch.utils.tensorboard import SummaryWriter
import torch
# Singleton writer instance
_writer_instance = None
_iter_dict = {}

def get_id_cur_step(id):
    _iter_dict[id] = _iter_dict.setdefault(id, 0) + 1
    return _iter_dict[id]
# aliases:
get_step = get_id_cur_step


def get_writer(log_dir="runs", with_id=None):
    """Get a singleton instance of the TensorBoard SummaryWriter."""
    global _writer_instance
    if id is not None:
        _iter_dict.setdefault(id, 0)
    if _writer_instance is None:
        _writer_instance = SummaryWriter(log_dir)
    return _writer_instance

def close_writer():
    """Close the SummaryWriter (use this when training ends)."""
    global _writer_instance
    if _writer_instance is not None:
        _writer_instance.close()
        _writer_instance = None

def grad_hook(tag="default"):
    def hook(grad):
        if grad is None:
            return
        step = get_id_cur_step(tag)
        writer = get_writer()
        writer.add_histogram(tag, grad.detach().cpu(), step)
    return hook


def log_gradients(model, step, logdir="runs", tag="Model", only_name=None):
    writer = get_writer(logdir)
     
    for name, param in model.named_parameters():
        if param.grad is not None:
            if only_name is not None and only_name not in name:
                continue
            grad = param.grad.detach().cpu()

            # Log histograms for gradient distribution
            writer.add_histogram(f"{tag}/grad/{name}", grad, step)

            # Log mean, std, max, min of gradients
#            writer.add_scalar(f"{tag}/grad/{name}_mean", grad.mean().item(), step)
#            writer.add_scalar(f"{tag}/grad/{name}_std", grad.std().item(), step)
#            writer.add_scalar(f"{tag}/grad/{name}_max", grad.max().item(), step)
#            writer.add_scalar(f"{tag}/grad/{name}_min", grad.min().item(), step)
#
#            # Log gradient norm (magnitude)
#            grad_norm = torch.norm(grad)
#            writer.add_scalar(f"{tag}/grad_norm/{name}", grad_norm.item(), step)
