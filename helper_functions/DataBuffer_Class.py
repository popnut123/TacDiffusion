import logging
import threading
logger = logging.getLogger("ml_service")

class DataBuffer:
    def __init__(self, capacity=1024, horizon_prev=7,  sensor_size=18):
        self.buffer = []
        self.capacity = capacity
        self.horizon_prev = horizon_prev # horizon for history
        self.sensor_size = sensor_size
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        print("DataBuffer for visulization is initialized!")
    def add_data(self, data):
        with self.lock:

            if len(self.buffer) >= self.capacity:  # determine the buffer size
                self.buffer.pop(0)  # pop out oldest buffer data
            self.buffer.append(data)  # add newest data

            # notify waiting threads that new data is available
            self.condition.notify_all()
            logger.debug("DataBuffer::add_data, current size: "+str(len(self.buffer)))
    def get_data(self, timeout=None):
        with self.lock:
            result = []
            # wait until there is data in the buffer
            while len(self.buffer) < self.horizon_prev:  # wait for buffer to save at least horizon_prev data points
                self.condition.wait(timeout=timeout)

            # get the data from the buffer
            try:
                # get the latest data and the data from the previous moment
                latest_data = self.buffer[-1][:self.sensor_size]
                previous_data = self.buffer[-self.horizon_prev][:self.sensor_size]
                result =  [latest_data + previous_data]

            except IndexError:
                pass
            return result
    def empty_buffer(self):
        with self.lock:
            self.buffer = []
    def get_size(self):
        return len(self.buffer)
