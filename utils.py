import torch
import config

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    """
    Saves the current state of the model and optimizer to a file.

    Parameters:
    - model (torch.nn.Module): The model to be saved.
    - optimizer (torch.optim.Optimizer): The optimizer to be saved.
    - filename (str, optional): The path to the file where the checkpoint will be saved. Defaults to "my_checkpoint.pth.tar".
    """
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    """
    Loads the model and optimizer states from a checkpoint file.

    Parameters:
    - checkpoint_file (str): The path to the checkpoint file.
    - model (torch.nn.Module): The model for which the state will be loaded.
    """
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
