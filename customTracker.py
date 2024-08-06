import os
from typing import Union
from torch.utils.tensorboard import SummaryWriter
from accelerate.tracking import GeneralTracker, on_main_process


class TensorboardTracker(GeneralTracker):
    """
    Custom 'Tracker' class that supports 'tensorboard'
    Args:
        run_name (str): The name of the experiment run.
        logging_dir (Union[str, os.PathLike]): Location for TensorBoard logs to be stored.
        kwargs: Additional keyword arguments passed along to the 'tensorboard.SummaryWriter.__init__' method.
    """

    name = "tensorboard"
    requires_logging_directory = True

    @on_main_process  # Ensure this decorator is from the correct module.
    def __init__(self, run_name: str, logging_dir: Union[str, os.PathLike], **kwargs) -> None:
        super().__init__()
        self.run_name = run_name
        self.logging_dir = logging_dir
        self.writer = SummaryWriter(self.logging_dir, **kwargs)

    @property
    def tracker(self):
        return self.writer

    @on_main_process
    def add_scalar(self, tag: str, scalar_value: float, global_step: int, **kwargs) -> None:
        self.writer.add_scalar(tag=tag, scalar_value=scalar_value, global_step=global_step, **kwargs)

    @on_main_process
    def add_image(self, tag: str, img_tensor, global_step: int, **kwargs) -> None:
        self.writer.add_image(tag=tag, img_tensor=img_tensor, global_step=global_step, **kwargs)

    # Add more methods as needed.


if __name__ == "__main__":
    import time
    import torch
    from accelerate import Accelerator

    accelerator = Accelerator()
    # assuming that the 'accelerator' object is already initialized.

    # Example usage:
    tb_writer = TensorboardTracker(run_name="test", logging_dir="logs/")
    accelerator.log_with = tb_writer

    # training loop
    for i in range(10):
        # dummy metrics
        tb_writer.add_scalar("loss", 10 / (i + 1), i + 1)
        # dummy image
        tb_writer.add_image("image", torch.rand(3, 256, 256), i + 1)
        time.sleep(0.1)  # writing too many images too quickly may damage the logs?

    print("To check the logs, run 'tensorboard --logdir=logs/' in the terminal.")
