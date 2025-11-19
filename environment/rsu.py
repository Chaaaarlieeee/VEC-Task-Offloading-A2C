"""
RSU (Road Side Unit) Module
Roadside computing unit module: defines RSU class and task processing functionality
"""

import numpy as np
from .task import TaskStatus


class RSU:
    """
    Roadside Unit Class
    Provides edge computing service with stronger computing capability than vehicles
    """
    
    def __init__(self, rsu_id, position, coverage_radius=250, 
                 max_parallel_tasks=2, max_queue_size=10):
        """
        Initialize RSU object
        
        Args:
            rsu_id (int): RSU ID, starting from 1
            position (float): RSU position in meters
            coverage_radius (float): Coverage radius in meters
            max_parallel_tasks (int): Maximum parallel tasks
            max_queue_size (int): Maximum queue length
        """
        self.rsu_id = rsu_id
        self.position = position
        self.coverage_radius = coverage_radius
        
        # Randomly select computing capability
        computing_options = [round(x, 2) for x in [5, 5.5, 6]]
        self.computing_capability = np.random.choice(computing_options)  # GHz
        
        # Randomly select cache size
        cache_options = [round(x, 2) for x in [20, 25, 30]]
        self.cache_size = np.random.choice(cache_options) * 1024 * 1024  # Convert to bytes
        self.cache_available = self.cache_size
        
        self.current_tasks = []  # Currently processing tasks list
        self.max_parallel_tasks = max_parallel_tasks
        self.task_queue = []  # Task waiting queue
        self.max_queue_size = max_queue_size
        
        # Failure simulation
        self.failure = 0  # Failure status
        self.failure_prob = 0.005  # Failure probability
        self.failure_time = 0  # Failure recovery time
    
    def is_vehicle_in_coverage(self, vehicle_position):
        """Check if vehicle is within RSU coverage"""
        return abs(self.position - vehicle_position) <= self.coverage_radius
    
    def failure_judge(self):
        """Check and update RSU failure status"""
        if self.failure == 1:
            self.failure_time -= 0.1
        if self.failure_time <= 0:
            self.failure = 0
        if self.failure == 0:
            if np.random.rand() < self.failure_prob:
                self.failure = 1
                self.failure_time = 2  # Set failure time to 2 seconds
    
    def process_tasks(self, vehicle_position, current_time, time_slot=0.1, 
                     resource_allocation=None):
        """
        Process tasks in RSU task queue with dynamic resource allocation
        
        Args:
            vehicle_position (float): Vehicle current position
            current_time (float): Current time
            time_slot (float): Time slot size
            resource_allocation (list, optional): Resource allocation ratios for current tasks
        """
        # Check if RSU is in failure state
        if self.failure == 1:
            return
        
        # Check if current tasks are overdue or vehicle left coverage
        for task in self.current_tasks[:]:
            if task.is_overdue(current_time):
                self.cache_available += task.size
                task.fail_overtime(current_time)
                self.current_tasks.remove(task)
            elif not self.is_vehicle_in_coverage(vehicle_position):
                self.cache_available += task.size
                task.fail_out_of_rsu(current_time)
                self.current_tasks.remove(task)
        
        # Check if tasks in queue are overdue
        for task in self.task_queue[:]:
            if task.is_overdue(current_time):
                task.start_processing(current_time)
                self.cache_available += task.size
                task.fail_overtime(current_time)
                self.task_queue.remove(task)
        
        # Add new tasks from queue to processing list
        while len(self.current_tasks) < self.max_parallel_tasks and self.task_queue:
            new_task = self.task_queue.pop(0)
            new_task.start_processing(current_time)
            self.current_tasks.append(new_task)
        
        # If no resource allocation provided, distribute evenly
        if resource_allocation is None and self.current_tasks:
            if len(self.current_tasks) == 1:
                resource_allocation = [1.0]
            else:
                resource_allocation = [1.0 / len(self.current_tasks)] * len(self.current_tasks)
        
        # Process current tasks
        for i, task in enumerate(self.current_tasks[:]):
            if i < len(resource_allocation):
                # Calculate processing amount based on allocated resource ratio
                allocation_ratio = resource_allocation[i]
                allocated_computing_power = self.computing_capability * allocation_ratio
                processing_amount = allocated_computing_power * time_slot * 1e9
                
                # Set task's allocated computing resource
                task.allocated_computing_power = allocated_computing_power
                
                # Update task progress
                task.current_process -= processing_amount
                total_cycles = task.complexity * task.size * 8
                task.progress = 1.0 - (task.current_process / total_cycles)
                if task.progress > 1.0:
                    task.progress = 1.0
                
                # Check if task is completed
                if task.current_process <= 0:
                    self.cache_available += task.size
                    if not self.is_vehicle_in_coverage(vehicle_position):
                        task.fail_out_of_rsu(current_time)
                    else:
                        task.complete(current_time)
                    self.current_tasks.remove(task)

