import torch


def adam(params, lr):
    return torch.optim.Adam(params, lr)
def adamW(params, lr, weight_decay=1e-2):
    return torch.optim.AdamW(params, lr, weight_decay=weight_decay)


def cosine_anneal_schedule(optim, T_max):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max)


"""

singleton optimizer?
optimizers = {}
def get_optimizer("adam", "no schedule", lr=4e-4)
    if optimizer is None:
        optimizer = torch.optim.adam(params, lr)
    return optimizer,
    if want schedule:
        if schedule is None:
            scheduler = lr_scheduler(optimizer)
        return optimizer, scheduler
not sure how smart this is, or how much work this would save. You're pretty used
to using optimizers now. You can keep the adamw stuff and the cosine_anneal_schedule
but having a singleton paradigm of organizing this software isn't obviously useful.

it's cute. It kind of means that only one module at time can use this singleton:
get_optimizer().step()
get_scheduler().step()
but this doesn't save any lines of code. But it's cute.
but can't I just pass in the name of this pid?

which would get used like:
optim = get_optimizer("adam", "no schedule", lr)
"""