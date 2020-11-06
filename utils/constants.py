# initial training constants
INITIAL_TRAINING_EPOCHS = 1
INITIAL_BATCH_SIZE = 1000
EMBEDDING_LENGTH = 40

# incremental training constants
INCREMENTAL_BATCH_SIZE = 10
INCREMENTAL_SAMPLE_SIZE = 500
INCREMENTAL_TRAINING_EPOCHS = 2
MODEL_DUMP_FREQUENCY_IN_SECONDS = 1800

# features
ITEM_ATTRIBUTES = ['ontology', 'price']

# Redis
REDIS_HOST = 'localhost'
REDIS_PORT = '6379'

# Mongo
MONGO_USER = None
MONGO_PORT = 27017
MONGO_HOST = 'localhost'
MONGO_PASSWORD = None
MONGO_DATABASE = 'msd_recommendation_engine'

# faiss app
FAISS_APP_HOST = 'localhost'
FAISS_APP_PORT = '5000'

# Kafka
KAFKA_TOPIC = "user_events"
KAFKA_BROKER = None  # to be set
KAFKA_CONSUMER_OFFSET = "latest"
KAFKA_GROUP_ID = "training"

# statsd
STATSD_HOST = None # to be set
STATSD_PORT = None # to be set
