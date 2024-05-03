import os
import sys
import logging



log_dir = "MLOPs_workflow/logs"
log_file = os.path.join(log_dir, "running_logs.log")

if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

logging_format = "[%(asctime)s : %(levelname)s : %(module)s : %(message)s]"

logging.basicConfig(level= logging.INFO,
                    format = logging_format,
                    handlers = [
                        logging.FileHandler(log_file),
                        logging.StreamHandler(sys.stdout)
                    ])

logger = logging.getLogger("E2EMLOps")
