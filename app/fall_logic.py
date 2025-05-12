import time

class FallTimer:
    def __init__(self, threshold=10):
        self.start_time = None
        self.threshold = threshold

    def update(self, is_falling):
        current_time = time.time()
        if is_falling:
            if self.start_time is None:
                self.start_time = current_time
            elif current_time - self.start_time >= self.threshold:
                return True
        else:
            self.start_time = None
        return False