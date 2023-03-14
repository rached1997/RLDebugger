import wandb


class WandbLogger:
    """A Wandb logger.
    """

    def __init__(
            self,
            project=None,
            name=None,
            dir=None,
            mode=None,
            id=None,
            resume=None,
            start_method=None):
        settings = None
        if start_method is not None:
            settings = wandb.Settings(start_method=start_method)

        wandb.init(
            project=project,
            name=name,
            dir=dir,
            mode=mode,
            id=id,
            resume=resume,
            settings=settings
        )

    def custom_wandb_logger(self, project, name, dir=None, mode=None, id=None, resume=None, start_method=None,
                            **kwargs):
        settings = None
        if start_method is not None:
            settings = wandb.Settings(start_method=start_method)
        wandb.init(project=project, name=name, dir=dir, mode=mode, id=id, resume=resume, settings=settings, **kwargs)

    def plot(self, metrics, prefix="debugger"):
        metrics = {f"{prefix}/{name}": value for (name, value) in metrics.items()}
        wandb.log(metrics)
