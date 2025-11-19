"""
Vehicle Module
Vehicle module: defines vehicle class with computation, movement, and task processing
"""

import numpy as np
from .task import TaskStatus


class Vehicle:
    """
    Vehicle Class
    Represents a vehicle with computing capability that can process tasks locally
    """
    
    def __init__(self, position=0, cpu=1.5, speed=20, task_queue=None, 
                 cache_size=10*1024*1024, time_slot=0.1, use_idm=False):
        """
        Initialize vehicle object
        
        Args:
            position (float): Initial position in meters
            cpu (float): CPU computing capability in GHz
            speed (float): Vehicle speed in m/s
            task_queue (list): Task queue
            cache_size (int): Cache size in bytes
            time_slot (float): Time slot in seconds
            use_idm (bool): Whether to use IDM intelligent driving model
        """
        self.position = position
        self.cpu = cpu  # GHz
        self.speed = speed  # m/s
        self.task_queue = task_queue if task_queue is not None else []
        self.current_task = []
        self.time_slot = time_slot
        self.cache_available = cache_size
        self.cache_size = cache_size
        self.max_queue_length = 5  # Maximum task queue length
        
        # IDM model parameters
        self.use_idm = use_idm  # Whether to use IDM model
        self.desired_speed = speed  # Desired speed (m/s)
        self.current_speed = 10  # Current speed (m/s)
        self.max_acceleration = 8.0  # Maximum acceleration (m/s^2)
        self.comfortable_deceleration = 4.0  # Comfortable deceleration (m/s^2)
        self.min_speed = max(5, speed - 5)  # Minimum speed (m/s)
        self.max_speed = speed + 5  # Maximum speed (m/s)
        self.speed_variance = 5  # Speed random fluctuation variance
        self.acceleration = 0  # Current acceleration (m/s^2)
    
    def move(self, time_slot=0.1):
        """Vehicle movement"""
        if not self.use_idm:
            # Use fixed speed model
            self.position += self.speed * time_slot
        else:
            # Use free flow IDM model
            self._update_speed_idm(time_slot)
            self.position += self.current_speed * time_slot
    
    def _update_speed_idm(self, time_slot):
        """
        Update vehicle speed using Free Flow IDM model
        
        Args:
            time_slot (float): Time slot size, default 0.1 seconds
        """
        # Calculate speed difference
        speed_diff = self.current_speed - self.desired_speed
        
        # Base acceleration calculation (acceleration in free flow mode)
        free_road_term = -self.comfortable_deceleration * (speed_diff / self.desired_speed)
        
        # Add random fluctuation
        random_fluctuation = np.random.normal(0, self.speed_variance)
        
        # Calculate total acceleration
        self.acceleration = free_road_term + random_fluctuation
        
        # Limit acceleration range
        self.acceleration = max(-self.comfortable_deceleration, 
                               min(self.acceleration, self.max_acceleration))
        
        # Update speed
        self.current_speed += self.acceleration * time_slot
        
        # Limit speed range
        self.current_speed = max(self.min_speed, min(self.current_speed, self.max_speed))
    
    def local_process(self, current_time, time_slot=0.1):
        """
        Process tasks in vehicle's local task queue
        
        Args:
            current_time (float): Current time
            time_slot (float): Time slot size, default 0.1 seconds
        """
        # First check if there are overdue tasks in the queue
        for task in self.task_queue[:]:
            if task.is_overdue(current_time):
                task.start_processing(current_time)
                task.fail_overtime(current_time)
                self.task_queue.remove(task)
        
        # Check if there is a task in processing queue
        if self.current_task:
            task = self.current_task[0]
            if task.is_overdue(current_time):
                self.cache_available += task.size
                task.fail_overtime(current_time)
                self.current_task.pop(0)
            else:
                # Reduce task's computation requirement
                processing_amount = self.cpu * time_slot * 1e9  # Convert to Hz
                task.current_process -= processing_amount
                # Update task progress
                total_cycles = task.complexity * task.size * 8
                task.progress = 1.0 - (task.current_process / total_cycles)
                if task.progress > 1.0:
                    task.progress = 1.0
                # Check if task is completed
                if task.current_process <= 0:
                    self.cache_available += task.size
                    task.complete(current_time)
                    self.current_task.pop(0)
        
        # If no task in processing and queue not empty, start processing a task
        if not self.current_task and self.task_queue:
            self.current_task.append(self.task_queue.pop(0))
            task = self.current_task[0]
            task.start_processing(current_time)
            # Reduce task's computation requirement
            processing_amount = self.cpu * time_slot * 1e9
            task.current_process -= processing_amount
            # Update task progress
            total_cycles = task.complexity * task.size * 8
            task.progress = 1.0 - (task.current_process / total_cycles)
            # Check if task is completed
            if task.current_process <= 0:
                self.cache_available += task.size
                task.complete(current_time)
                self.current_task.pop(0)

