import ray
import numpy as np
import pandas as pd
import logging
import time
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level='INFO')



def add(a, b):
    time.sleep(1)
    return a + b

@ray.remote
def add_ray(a, b):
    time.sleep(1)
    return a + b


if __name__ == "__main__":
    ray.init()
    cpu_count = os.cpu_count()
    logger.info(f"Number of CPUs: {cpu_count}")
    a,b = 1,2
    start = time.time()
    result = [add(a,b) for _ in range(cpu_count - 2)]
    end = time.time()
    logger.info(f"Result: {result}, Time taken without Ray: {end-start}")
    start = time.time()
    res = ray.get([add_ray.remote(a,b) for _ in range(cpu_count - 2)])
    end = time.time()
    logger.info(f"Result: {res}, Time taken with Ray: {end-start}")
    ray.shutdown()