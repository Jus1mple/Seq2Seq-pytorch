# -*- coding:utf-8 -*-
# Timer 
# author: Matthew

import time
import datetime
import numpy as np


class Timer:
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()
    
    def today(self):
        """get current date. like: 20220627"""
        today = datetime.date.today()
        return "".join(today.strftime("%Y-%m-%d").split("-")) # return format like `20220627` 

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()
    