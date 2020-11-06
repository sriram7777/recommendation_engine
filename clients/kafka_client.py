from kafka import KafkaConsumer
from kafka import KafkaProducer
import json


class KafkaClient:
    """
    Generic Kafka Client
    """
    @staticmethod
    def create_kafka_producer(conf, v_serializer=False):
        KAFKA_BROKERS = conf['brokers']
        if v_serializer:
            kafka_producer = KafkaProducer(bootstrap_servers= KAFKA_BROKERS,
                                value_serializer=lambda m: json.dumps(m).encode('utf-8'))
        else:
            kafka_producer = KafkaProducer(bootstrap_servers = KAFKA_BROKERS)
        return kafka_producer

    @staticmethod
    def create_kafka_consumer(conf):
        KAFKA_BROKERS = conf['brokers']
        KAFKA_TOPIC = conf['input_topic']
        KAFKA_GROUP_ID = conf['group_id']
        KAFKA_OFFSET = conf["consumer_offset"]
        kafka_consumer = KafkaConsumer(
                        bootstrap_servers= KAFKA_BROKERS,
                        group_id=KAFKA_GROUP_ID,
                        auto_offset_reset=KAFKA_OFFSET,
                        consumer_timeout_ms=300000)
        kafka_consumer.subscribe([KAFKA_TOPIC])
        return kafka_consumer
