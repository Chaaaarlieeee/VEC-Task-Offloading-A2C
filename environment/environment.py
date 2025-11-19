"""
Environment Module  
车联网边缘计算环境模块：强化学习环境实现
"""

import numpy as np
from .task import TaskStatus


class VehicularEdgeEnvironment:
    """
    车联网边缘计算环境类
    实现强化学习环境接口
    """
    
    def __init__(self, vehicle, rsus, task_generator, communication, 
                 simulation_time=None, time_slot=0.1, task_generation_prob=0.15):
        """
        初始化车联网边缘计算环境
        
        参数:
            vehicle (Vehicle): 车辆对象
            rsus (list): RSU对象列表
            task_generator (TaskPoolGenerator): 任务生成器
            communication (Communication): 通信对象
            simulation_time (float, optional): 模拟总时间
            time_slot (float): 时间槽大小，默认0.1秒
            task_generation_prob (float): 每个时间槽生成新任务的概率，默认0.15
        """
        self.vehicle = vehicle
        self.rsus = rsus
        self.task_generator = task_generator
        self.communication = communication
        self.time_slot = time_slot
        self.task_generation_prob = task_generation_prob
        self.simulation_time = simulation_time
        # Calculate end_position
        self.end_position = len(rsus) * (2 * rsus[0].coverage_radius)
        self.transmitting_tasks = []
        
        # Environment state
        self.current_time = 0
        self.tasks = []
        self.completed_tasks = []
        self.new_task = None
        
        # Statistics
        self.total_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
    
    def reset(self):
        """Reset environment state"""
        # Reset time and statistics
        self.current_time = 0
        self.tasks = []
        self.completed_tasks = []
        self.new_task = None
        self.total_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        
        # Reset vehicle state
        self.vehicle.position = 0
        self.vehicle.task_queue = []
        self.vehicle.current_task = []
        self.vehicle.cache_available = self.vehicle.cache_size
        
        # Reset RSU state
        for rsu in self.rsus:
            rsu.task_queue = []
            rsu.current_tasks = []
            rsu.cache_available = rsu.cache_size
            rsu.failure = 0
            rsu.failure_time = 0
        
        return self.get_state()
    
    def step(self, action):
        """
        Execute one simulation timestep
        
        Args:
            action (int): Agent's action
                        0: Local processing
                        1~N: Offload to RSU
                        N+1: Skip allocation
        
        Returns:
            tuple: (next_state, reward, done, info)
        """
        reward = 0
        
        # Handle task allocation
        reward = self._handle_task_allocation(action)
        
        self._handle_transmitting_tasks()
        self._process_tasks()
        self.vehicle.move(self.time_slot)
        
        for rsu in self.rsus:
            rsu.failure_judge()
        
        self.current_time += self.time_slot
        self._update_task_status()
        self._generate_new_task()
        
        next_state = self.get_state()
        done = self._check_terminal()
        info = {
            'total_tasks': self.total_tasks,
            'successful_tasks': self.successful_tasks,
            'failed_tasks': self.failed_tasks,
            'success_rate': self.successful_tasks / max(1, self.successful_tasks + self.failed_tasks),
        }
        
        return next_state, reward, done, info
    
    def _handle_task_allocation(self, action):
        """Handle task allocation"""
        max_action = len(self.rsus) + 1
        
        if action is None or action < 0 or action > max_action:
            action = len(self.rsus) + 1
        
        if action == len(self.rsus) + 1:  # Skip action
            if self.new_task:
                task = self.new_task
                self.new_task = None
                task.fail_no_allocation(self.current_time)
                self.completed_tasks.append(task)
                self.failed_tasks += 1
                return -0.7
            else:
                return 0.1
        
        task = None
        if self.new_task:
            task = self.new_task
            self.new_task = None
        else:
            return -0.2
        
        return self._allocate_task(task, action)
    
    def _allocate_task(self, task, action):
        """Allocate a single task"""
        reward = 0
        
        def _task_cycles(t):
            return t.current_process if t.status == TaskStatus.PROCESSING else t.complexity * t.size * 8
        
        def _estimate_completion_time(task_list, new_task, cpu_ghz):
            remaining_seconds = sum(_task_cycles(t) for t in task_list) / (cpu_ghz * 1e9)
            new_task_seconds = (new_task.complexity * new_task.size * 8) / (cpu_ghz * 1e9)
            return remaining_seconds + new_task_seconds, remaining_seconds
        
        if action == 0:  # Local processing
            if task.size <= self.vehicle.cache_available:
                if len(self.vehicle.task_queue) >= self.vehicle.max_queue_length:
                    task.fail_queue_overflow(self.current_time)
                    self.completed_tasks.append(task)
                    self.failed_tasks += 1
                    reward = -0.7
                    return reward
                
                total_time, wait_time = _estimate_completion_time(
                    self.vehicle.current_task + self.vehicle.task_queue,
                    task, self.vehicle.cpu)
                
                if total_time > task.deadline:
                    wasted = max(0.0, total_time - task.deadline)
                    if wait_time >= task.deadline:
                        reward = -0.5 * min(1.0, wasted / task.deadline)
                    else:
                        reward = -1.0 * min(1.0, wasted / task.deadline)
                else:
                    ratio = total_time / task.deadline
                    reward = max(0.0, 1 - ratio)
                
                self.vehicle.cache_available -= task.size
                task.assign(0, self.current_time)
                self.vehicle.task_queue.append(task)
            else:
                task.fail_no_cache(self.current_time)
                self.completed_tasks.append(task)
                self.failed_tasks += 1
                reward = -0.5
        
        else:  # Offload to RSU
            rsu_id = action
            rsu = next((r for r in self.rsus if r.rsu_id == rsu_id), None)
            
            if rsu is None:
                reward = -0.7
                task.fail_no_allocation(self.current_time)
                self.completed_tasks.append(task)
                self.failed_tasks += 1
                return reward
            
            if not rsu.is_vehicle_in_coverage(self.vehicle.position):
                reward = -1.0
                task.fail_out_of_rsu(self.current_time)
                self.completed_tasks.append(task)
                self.failed_tasks += 1
                return reward
            
            if rsu.failure != 0:
                reward = -0.3
                task.fail_rsu_failure(self.current_time)
                self.completed_tasks.append(task)
                self.failed_tasks += 1
                return reward
            
            if task.size > rsu.cache_available:
                task.fail_no_cache(self.current_time)
                self.completed_tasks.append(task)
                self.failed_tasks += 1
                reward = -0.5
                return reward
            
            if len(rsu.task_queue) >= rsu.max_queue_size:
                reward = -0.7
                task.fail_queue_overflow(self.current_time)
                self.completed_tasks.append(task)
                self.failed_tasks += 1
                return reward
            
            if len(self.transmitting_tasks) >= 5:
                reward = -0.7
                task.fail_queue_overflow(self.current_time)
                self.completed_tasks.append(task)
                self.failed_tasks += 1
                return reward
            
            transmission_time = self.communication.calculate_transmission_time(
                task.size, self.vehicle, rsu)
            
            total_processing_time, wait_time = _estimate_completion_time(
                rsu.current_tasks + rsu.task_queue, task,
                rsu.computing_capability)
            
            total_time = transmission_time + total_processing_time
            wait_time = transmission_time + wait_time
            
            if total_time > task.deadline:
                wasted = max(0.0, total_time - task.deadline)
                if wait_time >= task.deadline:
                    reward = -0.3 * min(1.0, wasted / task.deadline)
                else:
                    reward = -0.8 * min(1.0, wasted / task.deadline)
            else:
                ratio = total_time / task.deadline
                reward = max(0.0, 1 - ratio)
            
            rsu.cache_available -= task.size
            task.assign(rsu_id, self.current_time)
            task.start_transmission(self.current_time)
            self.transmitting_tasks.append({'task': task, 'rsu': rsu})
        
        reward = max(-1.0, min(1.0, reward))
        return reward
    
    def _handle_transmitting_tasks(self):
        """Handle tasks currently transmitting"""
        if not self.transmitting_tasks:
            return
        
        transmit_info = self.transmitting_tasks[0]
        task = transmit_info['task']
        rsu = transmit_info['rsu']
        
        if task.is_overdue(self.current_time):
            rsu.cache_available += task.size
            task.fail_overtime(self.current_time)
            self.completed_tasks.append(task)
            self.failed_tasks += 1
            self.transmitting_tasks.pop(0)
            return
        
        if not rsu.is_vehicle_in_coverage(self.vehicle.position):
            rsu.cache_available += task.size
            task.fail_out_of_rsu(self.current_time)
            self.completed_tasks.append(task)
            self.failed_tasks += 1
            self.transmitting_tasks.pop(0)
            return
        
        transmission_rate = self._calculate_current_transmission_rate(task, rsu)
        transmitted_bytes = transmission_rate * self.time_slot
        task.update_transmission(transmitted_bytes)
        
        if task.get_remaining_data() <= 0:
            task.end_transmission(self.current_time)
            rsu.task_queue.append(task)
            self.transmitting_tasks.pop(0)
    
    def _calculate_current_transmission_rate(self, task, rsu):
        """Calculate current transmission rate"""
        transmission_time_per_byte = self.communication.calculate_transmission_time(1, self.vehicle, rsu)
        
        if transmission_time_per_byte and transmission_time_per_byte > 0:
            transmission_rate = 1 / transmission_time_per_byte
        else:
            transmission_rate = 0
        
        return transmission_rate
    
    def _process_tasks(self):
        """Process all ongoing tasks"""
        self.vehicle.local_process(self.current_time, self.time_slot)
        
        for rsu in self.rsus:
            rsu.process_tasks(self.vehicle.position, self.current_time, self.time_slot)
    
    def _update_task_status(self):
        """Update task status and statistics"""
        for task in self.tasks:
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED_OVERTIME, 
                              TaskStatus.FAILED_NO_CACHE, TaskStatus.FAILED_OUT_OF_RSU]:
                if task not in self.completed_tasks:
                    self.completed_tasks.append(task)
                    if task.success:
                        self.successful_tasks += 1
                    else:
                        self.failed_tasks += 1
    
    def _generate_new_task(self):
        """Generate new task"""
        if not self.new_task and np.random.random() < self.task_generation_prob:
            self.new_task = self.task_generator.get_task(self.current_time)
            if self.new_task:
                self.tasks.append(self.new_task)
                self.total_tasks += 1
    
    def _check_terminal(self):
        """Check if termination condition is met"""
        if self.simulation_time is not None:
            return self.current_time >= self.simulation_time
        else:
            return self.vehicle.position >= self.end_position
    
    def get_state(self):
        """Get current environment state"""
        def _task_to_state(task):
            return {
                'remaining_work': task.current_process,
                'time_to_deadline': task.time_to_deadline(self.current_time),
                'priority': task.priority.value,
                'size': task.size,
                'complexity': task.complexity
            }
        
        transmission_rate = 0.0
        current_rsu = None
        for rsu in self.rsus:
            if rsu.is_vehicle_in_coverage(self.vehicle.position) and rsu.failure == 0:
                current_rsu = rsu
                break
        if current_rsu is not None:
            time_per_byte = self.communication.calculate_transmission_time(1, self.vehicle, current_rsu)
            if time_per_byte and time_per_byte > 0:
                transmission_rate = 1 / time_per_byte
        
        state = {
            'current_time': self.current_time,
            'vehicle': {
                'cpu': self.vehicle.cpu,
                'cache_available': self.vehicle.cache_available,
                'position': self.vehicle.position,
                'speed': self.vehicle.speed,
                'acceleration': self.vehicle.acceleration,
                'transmission_rate': transmission_rate,
                'processing_length': len(self.vehicle.current_task),
                'queue_length': len(self.vehicle.task_queue)
            },
            'rsus': [],
            'transmitting_tasks': [],
            'new_task': None,
        }
        
        for rsu in self.rsus:
            rsu_info = {
                'id': rsu.rsu_id,
                'position': rsu.position,
                'distance': rsu.position - self.vehicle.position,
                'cpu': rsu.computing_capability,
                'cache_available': rsu.cache_available,
                'failure': rsu.failure,
                'processing_length': len(rsu.current_tasks),
                'queue_length': len(rsu.task_queue),
                'coverage_radius': rsu.coverage_radius
            }
            state['rsus'].append(rsu_info)
        
        def _transmitting_task_to_state(task):
            return {
                'transmitted_data': task.transmitted_data,
                'remaining_data': task.get_remaining_data(),
                'time_to_deadline': task.time_to_deadline(self.current_time),
                'priority': task.priority.value,
                'size': task.size,
                'complexity': task.complexity
            }
        
        transmitting_tasks_info = {
            'current_task': None,
            'queue_length': len(self.transmitting_tasks)
        }
        
        if self.transmitting_tasks:
            info = self.transmitting_tasks[0]
            task = info['task']
            rsu = info['rsu']
            trans_rate = self._calculate_current_transmission_rate(task, rsu)
            
            current_task_state = _transmitting_task_to_state(task)
            current_task_state.update({
                'rsu_id': rsu.rsu_id,
                'transmission_rate': trans_rate
            })
            transmitting_tasks_info['current_task'] = current_task_state
        
        state['transmitting_tasks'] = transmitting_tasks_info
        
        if self.new_task:
            state['new_task'] = _task_to_state(self.new_task)
        
        return state


def find_nearest_rsu(vehicle_position, rsus):
    """
    找到离车辆最近且在覆盖范围内的RSU
    
    参数:
        vehicle_position (float): 车辆位置
        rsus (list): RSU对象列表
        
    返回:
        int: 最近RSU的ID，如果没有在覆盖范围内的RSU则返回None
    """
    in_range_rsus = [(rsu.rsu_id, abs(rsu.position - vehicle_position))
                    for rsu in rsus if rsu.is_vehicle_in_coverage(vehicle_position) and rsu.failure == 0]
    
    if not in_range_rsus:
        return None
    
    nearest_rsu = min(in_range_rsus, key=lambda x: x[1])
    return nearest_rsu[0]

