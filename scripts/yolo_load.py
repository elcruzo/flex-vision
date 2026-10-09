"""Bounded admission for staggered synthetic cameras, independent of CUDA."""
from collections import deque

FIELDS = ['frame','lane','variant','arrival_s','status','submission_s','completion_s',
          'arrival_to_completion_ms','gpu_pipeline_ms','correct']


class Arrivals:
    def __init__(self, cameras, fps, seconds, depth, rotation=0):
        if min(cameras,fps,seconds,depth) <= 0:
            raise ValueError('Arrival settings must be positive')
        self.cameras,self.fps,self.seconds,self.depth = cameras,fps,seconds,depth
        self.rotation = rotation
        self.total = cameras*fps*seconds
        self.next = 0
        self.queues = [deque() for _ in range(cameras)]
        self.rows = []
        self.max_pending = 0

    def advance(self, elapsed):
        while self.next < self.total and self.next/(self.cameras*self.fps) <= elapsed:
            index = self.next
            lane = (index+self.rotation)%self.cameras
            row = {'frame':index,'lane':lane,'variant':(index//self.cameras)%2,
                   'arrival_s':index/(self.cameras*self.fps),'status':'pending'}
            self.next += 1
            if len(self.queues[lane]) >= self.depth:
                row['status'] = 'dropped_queue_full'
            else:
                self.queues[lane].append(row)
            self.rows.append(row)
            self.max_pending = max(self.max_pending,sum(map(len,self.queues)))

    def pop(self):
        pending = [q for q in self.queues if q]
        return min(pending,key=lambda q:q[0]['frame']).popleft() if pending else None

    @property
    def finished(self):
        return self.next == self.total and not any(self.queues)
