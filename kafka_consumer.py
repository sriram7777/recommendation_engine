import os
import pandas as pd
import time
from datetime import timedelta

script_name = os.path.basename(__file__)
script_dir = os.path.dirname(os.path.abspath(__file__))

from clients.kafka_client import KafkaClient
from clients.logger_client import LoggerClient
from utils import constants

logger = LoggerClient.set_logger("kafka_consumer.log")
conf = {
    # kafka configurations
}

kafka_consumer = KafkaClient.create_kafka_consumer(conf)

if __name__ == '__main__':
    for i, row in enumerate(kafka_consumer):
        # incremental training code to be kept here
        pass
