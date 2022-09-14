from collections import defaultdict
from datetime import timedelta
from typing import Iterable
import json
import numpy as np
import math
import time


def json_loadf(file_path: str):
    """ Parses a JSON file into a dictionary

    Args:
        file_path (string): path to the JSON file

    Returns:
        dict: the parsed JSON object
    """
    with open(file_path, "r") as file:
        return json.loads(file.read())


def json_savef(object: dict, file_path: str):
    """ Saves a dictionary as a JSON file

    Args:
        object (dict): the object to save as JSON
        file_path (str): the path to the JSON file to save in
    """
    with open(file_path, "w") as file:
        file.write(json.dumps(object))


def unzip(iterable: Iterable):
    """ The opposite of zip.
    Example:
        >>> time_temperature = [['7/22', 54], ['7/23', 55], ['7/24', 43]]
        >>> time, temp = unzip(time_temperature)
        >>> time
        ['7/22', '7/23', '7/24']
        >>> temp
        [54, 55, 43]

    Args:
        iterable (Iterable): the object to unzip

    Returns:
        List: the unzipped list
    """
    return list(zip(*iterable))


def wrap(x: float, high: float):
    """ Clips x on the domain [-high, high], repeating itself 
    See this graph to understand better
    https://www.desmos.com/calculator/qruhpbx45y

    Args:
        x (float): a number to wrap
        high (float): the maximum value of x
    """
    return ((x - high) % (2 * high)) - high


def discretize(x: float, lo: float, hi: float, n: float):
    """ Returns index of x between lo and hi divided by n
    """
    index = math.floor((x - lo) / (hi - lo) * n)
    return np.clip(index, 0, n - 1)


def staircase(x, y):
    """ Given a list of (x, y) coordinates, this function will return
    a list xy coordinates that, when plotted, show a staircase of xy

    Args:
        x (List[float]): list of x coordinates
        y (List[float]): list of y coordinates
    """
    assert len(x) == len(y), "Expected x and y to have same length"
    if len(x) <= 1:
        return x, y

    xy = sorted(zip(x, y), key=lambda t: t[0])
    x_res = [xy[0][0]]
    y_res = [xy[0][1]]
    for i in range(1, len(xy)):
        x_prev, y_prev = xy[i - 1]
        x_curr, y_curr = xy[i]

        x_res.append((x_prev + x_curr) / 2)
        y_res.append(y_prev)

        x_res.append((x_prev + x_curr) / 2)
        y_res.append(y_curr)

        x_res.append(x_curr)
        y_res.append(y_curr)

    return x_res, y_res    


def flatten(iterable):
    """ Flattens an iterable

    Args:
        iterable (Iterable): the iterable to flatten

    Returns:
        List[any]: the flatten list of items
    """
    res = []
    try:
        iterator = iter(iterable)
    except TypeError:
        return [iterable]
    else:
        for i in iterator:
            res += flatten(i)
    return res


class Timer:
    DEFAULT_NAME = 'Timer_Default_Name'
    START_KEY = 0
    END_KEY = 1
    RUNNING_KEY = 2

    def __init__(self):
        self.timer_map = defaultdict(lambda: [0, 0, False])
        self.running = False

    def start(self, name=DEFAULT_NAME):
        self.timer_map[name][Timer.START_KEY] = time.time()
        self.timer_map[name][Timer.RUNNING_KEY] = True

    def stop(self, name=DEFAULT_NAME):
        self.timer_map[name][Timer.END_KEY] = time.time()
        self.timer_map[name][Timer.RUNNING_KEY] = False

    def str(self, name=DEFAULT_NAME):
        hours, remainder = divmod(self.timer_map[name][Timer.END_KEY] - self.timer_map[name][Timer.START_KEY], 3600)
        minutes, seconds = divmod(remainder, 60)
        return '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))