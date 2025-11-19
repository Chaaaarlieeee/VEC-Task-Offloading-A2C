import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from environment import Vehicle, RSU, TaskPoolGenerator, Communication, VehicularEdgeEnvironment, TaskStatus, find_nearest_rsu

def run_random_baseline(vehicle=None, rsus=None, task_generator=None, communication=None,simulation_time=None, seed=42, verbose=True):
    """Run random baseline strategy: 50% local, 50% offload to nearest RSU"""

    np.random.seed(seed)
     
    if rsus == None:

        rsus = []
        for i in range(10):
            rsus.append(RSU(rsu_id=i+1, position=250 + i*500))

    if task_generator==None:

        task_generator = TaskPoolGenerator(seed=seed)
        task_generator.generate_task_pool(100)

    if vehicle == None:
      
        vehicle = Vehicle(position=0, cpu=2.5, speed=20,use_idm=True)
    
    if communication ==None:
      
        communication = Communication()
    
 
    env = VehicularEdgeEnvironment(
        vehicle=vehicle,
        rsus=rsus,
        task_generator=task_generator,
        communication=communication,
        simulation_time=simulation_time
    )
    
   
    env.reset()
    

    history = {
        'time': [],
        'vehicle_position': [],
        'cache_available': [],
        'queue_length': [],
        'total_tasks': [],
        'successful_tasks': [],
        'failed_tasks': [],
        'success_rate': [],
        'transmitting_tasks': [],
        'rewards': []  
    }
    
  
    cumulative_reward = 0
    

    while not env._check_terminal():
        
        if env.new_task:
          
            if np.random.random() < 0.5:
               
                action = 0
            else:
               
                nearest_rsu_id = None
                in_range_rsus = []
                
                for rsu in env.rsus:
                    if rsu.is_vehicle_in_coverage(env.vehicle.position) and rsu.failure == 0:
                        distance = abs(rsu.position - env.vehicle.position)
                        in_range_rsus.append((rsu.rsu_id, distance))
                
                if in_range_rsus:
                   
                    nearest_rsu = min(in_range_rsus, key=lambda x: x[1])
                    nearest_rsu_id = nearest_rsu[0]  
               
                    action = nearest_rsu_id
                else:
                   
                    action = len(env.rsus) + 1
        else:
           
            action = len(env.rsus) + 1
        
        
        next_state, reward, done, info = env.step(action)
        
      
        cumulative_reward += reward
        
        
        history['time'].append(env.current_time)
        history['vehicle_position'].append(env.vehicle.position)
        history['cache_available'].append(env.vehicle.cache_available)
        history['queue_length'].append(len(env.vehicle.task_queue) + len(env.vehicle.current_task))
        history['total_tasks'].append(info['total_tasks'])
        history['successful_tasks'].append(info['successful_tasks'])
        history['failed_tasks'].append(info['failed_tasks'])
        history['success_rate'].append(info['success_rate'])
        history['transmitting_tasks'].append(len(env.transmitting_tasks))
        history['rewards'].append(cumulative_reward)  
        
   
        if verbose and env.current_time % 10 < 0.1:
            rsu_status = "No RSU" if nearest_rsu_id is None else f"RSU-{nearest_rsu_id}"
            print(f"Time: {env.current_time:.1f}s, "
                  f"Position: {env.vehicle.position:.1f}m, "
                  f"Nearest RSU: {rsu_status}, "
                  f"Transmitting: {len(env.transmitting_tasks)}, "
                  f"Total: {info['total_tasks']}, "
                  f"Success: {info['successful_tasks']}, "
                  f"Failed: {info['failed_tasks']}, "
                  f"Rate: {info['success_rate']*100:.2f}%")
    

    if verbose:
        print("\n=== Random Strategy Results ===")
        print(f"Total tasks: {env.total_tasks}")
        print(f"Successful: {env.successful_tasks}")
        print(f"Failed: {env.failed_tasks}")
        print(f"Success rate: {env.successful_tasks / max(1, env.total_tasks) * 100:.2f}%")

    
   
    return {
        'total_tasks': env.total_tasks,
        'successful_tasks': env.successful_tasks,
        'failed_tasks': env.failed_tasks,
        'success_rate': env.successful_tasks / max(1, env.total_tasks),
        'completed_tasks': env.completed_tasks,
        'history': history,
        'rewards':cumulative_reward
        
    }

def analyze_task_failures(results):

    
  
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False   
    completed_tasks = results['completed_tasks']
    
    
    failed_tasks = [task for task in completed_tasks if not task.success]
    
    if not failed_tasks:
        print("No failed tasks")
        return
    
    
    failure_reasons = {}
    for task in failed_tasks:
        reason = task.status.name
        if reason not in failure_reasons:
            failure_reasons[reason] = 0
        failure_reasons[reason] += 1
    
    print("\nTask failure analysis:")
    for reason, count in failure_reasons.items():
        print(f"{reason}: {count} tasks ({count/len(failed_tasks)*100:.2f}%)")

def analyze_task_sizes(results):

    

    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False   
    
    completed_tasks = results['completed_tasks']
    
 
    all_sizes = [task.size/1024/1024 for task in completed_tasks]  
    

    successful_sizes = [task.size/1024/1024 for task in completed_tasks if task.success]
    failed_sizes = [task.size/1024/1024 for task in completed_tasks if not task.success]
    
    print("\nTask size analysis:")
    print(f"Average task size: {np.mean(all_sizes):.2f}MB")
    
    if successful_sizes:
        print(f"Successful task avg size: {np.mean(successful_sizes):.2f}MB")
    
    if failed_sizes:
        print(f"Failed task avg size: {np.mean(failed_sizes):.2f}MB")
    
 
    plt.figure(figsize=(10, 6))
    plt.hist(all_sizes, bins=20, alpha=0.5, label='所有任务')
    if successful_sizes:
        plt.hist(successful_sizes, bins=20, alpha=0.5, label='成功任务')
    if failed_sizes:
        plt.hist(failed_sizes, bins=20, alpha=0.5, label='失败任务')
    
    plt.xlabel('任务大小 (MB)')
    plt.ylabel('任务数量')
    plt.title('任务大小分布')
    plt.legend()
    plt.grid(True)
    plt.savefig('image/random_task_size_distribution.png', dpi=300)

def plot_random_baseline_results(results):
   
    history = results['history']
    
  
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False    

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
 
    axs[0, 0].plot(history['time'], history['total_tasks'], label='总任务数')
    axs[0, 0].plot(history['time'], history['successful_tasks'], label='成功任务数')
    axs[0, 0].plot(history['time'], history['failed_tasks'], label='失败任务数')
    axs[0, 0].set_xlabel('时间 (秒)')
    axs[0, 0].set_ylabel('任务数量')
    axs[0, 0].set_title('任务统计')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    

    axs[0, 1].plot(history['time'], [rate * 100 for rate in history['success_rate']])
    axs[0, 1].set_xlabel('时间 (秒)')
    axs[0, 1].set_ylabel('成功率 (%)')
    axs[0, 1].set_title('任务成功率')
    axs[0, 1].grid(True)
    
 
    axs[1, 0].plot(history['time'], history['queue_length'], label='本地队列长度')
    axs[1, 0].plot(history['time'], history['transmitting_tasks'], label='传输中任务数')
    axs[1, 0].set_xlabel('时间 (秒)')
    axs[1, 0].set_ylabel('任务数量')
    axs[1, 0].set_title('队列长度和传输中任务')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    

    axs[1, 1].plot(history['time'], [cache/1024/1024 for cache in history['cache_available']])
    axs[1, 1].set_xlabel('时间 (秒)')
    axs[1, 1].set_ylabel('可用缓存 (MB)')
    axs[1, 1].set_title('车辆可用缓存')
    axs[1, 1].grid(True)
    

    plt.tight_layout()
    plt.savefig('image/random_baseline_results.png', dpi=300)
    
 
    plt.figure(figsize=(12, 6))
    plt.plot(history['time'], history['vehicle_position'])
    plt.xlabel('时间 (秒)')
    plt.ylabel('位置 (米)')
    plt.title('车辆位置')
    plt.grid(True)
    plt.savefig('image/random_vehicle_position.png', dpi=300)

if __name__ == "__main__":

    print("===== 运行随机处理基线测试 =====")
    random_results = run_random_baseline(simulation_time=250, seed=42)
    

    analyze_task_failures(random_results)
    
 
    analyze_task_sizes(random_results)
 
    plot_random_baseline_results(random_results)
    
    print("\n结果图表已保存到当前目录。")
