"""
RL Environment Wrapper
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from environment import Vehicle, RSU, TaskPoolGenerator, Communication, VehicularEdgeEnvironment


class RLEnvironmentWrapper:
    """Wrapper for VehicularEdgeEnvironment to provide standard RL interface"""
    
    def __init__(self, simulation_time=None, seed=42):
        self.rsus = []
        for i in range(10):
            self.rsus.append(RSU(rsu_id=i+1, position=250 + i*500))
        
        self.task_generator = TaskPoolGenerator(seed=seed)
        self.task_generator.generate_task_pool(100)
        
        self.vehicle = Vehicle(position=0, cpu=2.5, speed=20, use_idm=True)
        self.communication = Communication()
        
        self.env = VehicularEdgeEnvironment(
            vehicle=self.vehicle,
            rsus=self.rsus,
            task_generator=self.task_generator,
            communication=self.communication,
            simulation_time=simulation_time
        )
        
        self.state = self.env.reset()
    
    def reset(self):
        self.state = self.env.reset()
        return self.state
    
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info
    
    def _get_action_space(self):
        """Return action space size: local(0) + RSUs(1-10) + skip(11)"""
        return len(self.rsus) + 2

