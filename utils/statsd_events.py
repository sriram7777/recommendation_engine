import statsd
import os
import sys

script_name = os.path.basename(__file__)
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.join(script_dir, "..") not in sys.path:
    sys.path.insert(0, os.path.join(script_dir, ".."))

from utils import constants

statsd_client = statsd.StatsClient(host=constants.STATSD_HOST, port=constants.STATSD_PORT)


def send_value(tag, value):
    statsd_client.gauge(tag, value)
