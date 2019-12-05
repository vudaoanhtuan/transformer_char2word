from tensorboardX import SummaryWriter
import numpy as np
import scipy.misc 



class Logger(object):
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def update_step(self, values, step):
        """Log a scalar variable."""
        self.writer.add_scalars("step_loss", values, step)
        self.writer.flush()

    def update_epoch(self, values, step):
        self.writer.add_scalars("epoch_loss", values, step)
        self.writer.flush()
