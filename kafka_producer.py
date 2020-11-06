import os
import pandas as pd
import time
from datetime import timedelta

script_name = os.path.basename(__file__)
script_dir = os.path.dirname(os.path.abspath(__file__))

from clients.kafka_client import KafkaClient
from clients.logger_client import LoggerClient
from utils import constants

logger = LoggerClient.set_logger("kafka_producer.log")
conf = {
    # kafka configurations
}

kafka_producer = KafkaClient.create_kafka_producer(conf)

if __name__ == '__main__':
    data_files = os.listdir(os.path.join(script_dir, 'data'))
    for file in data_files:
        stream_df = pd.read_csv(os.path.join(script_dir, 'data', file), sep='|')
        stream_df = stream_df.sort_values('clicked_epoch').reset_index(drop=True)
        stream_df.loc[:, 'clicked_time'] = pd.to_datetime(stream_df.clicked_epoch*10**9)
        start_time = stream_df.clicked_time.iloc[0]
        logger.info("current time: {}".format(start_time))
        start = time.time()
        for i, row in stream_df.iterrows():
            if start_time + timedelta(seconds=time.time() - start) > row.clicked_time:
                message = row.to_dict()
                message['clicked_time'] = str(message['clicked_time'])
                kafka_producer.send(topic=constants.KAFKA_TOPIC, value=message)
            if i%500 == 0:
                logger.info("kafka message: {}".format(message))
