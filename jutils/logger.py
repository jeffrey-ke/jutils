from torch.utils.tensorboard import SummaryWriter
import open3d as o3d
from jutils.utils import pdb
from matplotlib import pyplot as plt
from pathlib import Path
import torch
# Singleton writer instance
_writer_instance = None
_writer_name = None
_iter_dict = {}

def get_id_cur_step(id):
    _iter_dict[id] = _iter_dict.setdefault(id, 0) + 1
    return _iter_dict[id]
# aliases:
get_step = get_id_cur_step

"""
log_dir is deprecated
"""
def get_writer(log_dir="runs", with_id=None, experiment=None):
    """Get a singleton instance of the TensorBoard SummaryWriter."""
    global _writer_instance, _writer_name
    if id is not None:
        _iter_dict.setdefault(id, 0)
    if _writer_instance is None:
        if experiment is None:
            _writer_name = input("Name for log: (will be placed in runs/) ")
        else:
            _writer_name = experiment
        _writer_instance = SummaryWriter(f"runs/{_writer_name}")
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

def _plot_3d(axes, points, label, color):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    axes.scatter(x, y, z, c=color, label=label)
    
# to be able to plot multiple point clouds in one 3d graph:
# have the point cloud's name
def plot_3d(points_dict, name):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for label, (pc, color) in points_dict.items():
        _plot_3d(ax, pc, label,  color)
    ax.legend()
    path = Path.cwd() / "graphs"
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / f"{name}.png")
