# Real-time Recommendation Engine

# System Design Elements:

# Kafka:
This Apache Framework is the most preferred for streaming data, where the producer posts the data into a topic, and any number of consumers can subscribe and consume the same.
Capable of handling high velocity, high volume data. Low Latency. Supports partitioning for a distributed framework. 
Extremely fault-tolerant.

# PyTorch:
Though Tensorflow is kind-of the go-to place for DL models, PyTorch is preferred when you need more flexibility over models.
One of the main compelling reasons is that it supports dynamic computational graphs, which is exactly what we need, since we want to dynamically change the embedding layers as we encounter more users or items.
Another reason is the ability to selectively freeze some layers.

# Graphite:
Graphite is the data monitoring server setup that can collect data from different sources using simple plugins. Its fire-and-forget setup is preferred for its non-hindrance with the usual flow. 
Graphite can be easily integrated with any front end setup like Grafana.
This setup is not covered under the scope of this project but mentioned to give a holistic picture on how the whole setup would look like. I have included the stats-d module and commented lines representing how such stats can be sent over.

# Grafana:
Grafana is the open-source dashboarding tool that goes hand-in-hand with Graphite backend. Any kind of stat can be visualized over a chosen timeframe. It also has some monitoring and alerting features, which can come in handy.
This is not included in the project, but can be easily setup.

# Faiss:
Faiss is an open-source library developed by Facebook, for fast similarity search on high dimensional embedding vectors. It contains various techniques of indexes that we can use based on the use-case.
It has been used to index the learned item embeddings for faster similarity search during serving. It has latency time as low as 1~4 ms in a normal ec2 instance. For demo purposes, I have hosted this index with a falcon web app. One can also host it with a grpc framework for slightly higher throughput.
One can also look at Elastic Search (ES) offered by AWS for serving learned embeddings but would also increase the cost of serving naturally.

# MongoDB:
A Database is inevitable in any setup to store the data in an ordered way for later querying. MongoDB is a simple, schema-less, scalable NoSQL database that can serve our purpose of storing the user data (including embeddings). 
Embedding information can also be cached for relatively faster retrievals while serving but would need significantly more RAM when the user base scales to hundreds of millions. 

# Redis:
Redis is an in-memory data structure store for quick caching and retrieval of data. Here, Redis server is used to cache the item metadata, for faster retrievals from API framework.
This is relatively cheaper as we are storing only a few item attributes.

# Falcon:
Falcon is a lightweight fast RESTful API framework. No real reason behind it other than its ease of building quick API’s. 
Falcon framework is used to build the client-facing recommender app and also the internal faiss-app to serve faiss index.



# Algorithm:

Collaborative filtering is a well-known recommendation system to recommend based on learning implicit similarity existing between users. Apart from collaborative filtering methods, there are content-based recommender systems. In most of the real-world applications, these two methods are combined to spew out recommendations. Sticking with a simple collaborative filtering approach for the demo

One of the simplest methods is Matrix Factorization (MF), where the user and item are embedded into a latent space of N dimensions. The rating is the result of these two latent vectors. Matrix Factorization involves finding the components of these vectors. Traditional methods of MF include SVD, SVD++ etc. 

Using Neural Networks, one can approach MF through the Stochastic Gradient Descent (SGD) method. One such approach is explained in Neural Collaborative Filtering paper (https://arxiv.org/pdf/1708.05031.pdf)

# GMF:
They have explained many architectures and Generalized Matrix Factorization (GMF) is the one chosen for the project. This is a near real time approach where there will be some inherent delay for the trained embeddings to get served. But considering the reasonable complexity of the system design, this method would be ideal to showcase the depth in product and engineering aspects. 

This approach has been combined with an incremental set up explained in this paper https://repositorio.inesctec.pt/bitstream/123456789/3528/1/P-00A-1W7.pdf. As explained in that paper, I have used positive-only feedback which is more suitable for incremental learning. I’m reasonably certain that there are trade-offs in terms of performance, but incremental learning setup with near real time recommendations far outweigh the shortcomings.

# Model Intricacies:
Whenever the number of users or items overflows (as we see new users or items), 10% more embedding rows are added. Optimizer is reset to include the additional weights.
L2 Regulatizer (as in Fast Incremental Matrix Factorization) is given as weight decay in the optimizer.
The Linear layer before the output is not updated during incremental learning as it contains a single neuron which is already trained. During incremental learning, only the embedding weights are adjusted.
Inference loop in incremental learning runs on the faiss app to get the recommendations as it would be fast.

# Other Approaches:
There are some other interesting real time approaches like Rolling Feature Matrix explained in this blog https://towardsdatascience.com/real-time-recommendation-system-rolling-feature-matrix-f5ca701439df
Surprise - Python scikit for recommender systems
Pyspark.mllib.recommendation


# Some Compromises:
The code base has three separate modules (Incremental Training, Faiss App, Recommender App), crammed in. Ideal way is to separate them out and run as individual docker containers. Dockerfiles for dockerizing these services not included.
The same faiss app is used for offline and online inferencing, which might have to be separated out for better control over traffic.
Since the focus was more on the whole system design, I haven’t invested much time on model performance metrics, though I'm measuring the same (recall@k, precision@k) during incremental learning.
The popular items key in redis is right now populated only during initial training based on the frequency. Ideally, this has to be done from the streaming data using some efficient data structures. (popular items are used when neither user_id not item_id is known during online inferencing)
Graphite and grafana servers setup are not included, owing to time constraints. Statsd events are commented out in the codebase
Falcon framework is used for demonstration purposes. It can be replaced with any API framework of choice.


activate conda environment:
conda activate msd_recommendation_engine

install requirements:
pip install -r requirements.txt

run initial training:
python initial_training.py

launch faiss index app:
cd faiss_server,
gunicorn -b 0.0.0.0:5000 faiss_app:api -w 1 --reload --reload-extra-file $(pwd)/../models/item_embeddings.npy

launch incremental training:
python incremental_training.py

launch recommender app:
cd recommender_server, 
gunicorn -b 0.0.0.0:6000 recommender_app:api -w 1 --preload


