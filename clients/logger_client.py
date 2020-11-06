import logging
import logging.handlers
import os
from logging.handlers import RotatingFileHandler

script_name = os.path.basename(__file__)
script_dir = os.path.dirname(os.path.abspath(__file__))

class LoggerClient:
    """
    Generic logger client
    """
    @staticmethod
    def set_logger(log_file):
        logger = logging.getLogger(__name__)
        if "ENV" in os.environ and os.environ["ENV"] != "production":
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        # LOG_FILE = '/var/log/msd_recommendation_engine.log'
        handler = RotatingFileHandler(log_file, maxBytes=10485760, backupCount=30)
        formatter = logging.Formatter('%(process)d - %(processName)s - %(asctime)s - %(levelname)s - %(filename)s - %(lineno)s - %(message)s')

        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
