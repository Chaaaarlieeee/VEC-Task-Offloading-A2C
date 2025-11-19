import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from environment import Vehicle, RSU, TaskPoolGenerator, Communication, VehicularEdgeEnvironment, TaskStatus

def run_local_baseline(vehicle=None, rsus=None, task_generator=None, communication=None,simulation_time=None, seed=42, verbose=True):


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
        'rewards': []  
    }
    

    cumulative_reward = 0
    

    if verbose:
        print("Starting local processing baseline test...")
        print(f"Initial cache: {env.vehicle.cache_available/1024/1024:.2f}MB")
        print(f"CPU capacity: {env.vehicle.cpu}GHz")
    
    while not env._check_terminal():
       
        if env.new_task:
      
            action = 0
        else:
          
            action = len(env.rsus) + 1
            
        next_state, reward, done, info = env.step(action=action)
        
  
        cumulative_reward += reward
        

        history['time'].append(env.current_time)
        history['vehicle_position'].append(env.vehicle.position)
        history['cache_available'].append(env.vehicle.cache_available)
        history['queue_length'].append(len(env.vehicle.task_queue) + len(env.vehicle.current_task))
        history['total_tasks'].append(info['total_tasks'])
        history['successful_tasks'].append(info['successful_tasks'])
        history['failed_tasks'].append(info['failed_tasks'])
        history['success_rate'].append(info['success_rate'])
        history['rewards'].append(cumulative_reward)  
        

        if verbose and env.current_time % 10 < 0.1:
            print(f"Time: {env.current_time:.1f}s, "
                  f"Position: {env.vehicle.position:.1f}m, "
                  f"Cache: {env.vehicle.cache_available/1024:.2f}KB, "
                  f"Queue: {len(env.vehicle.task_queue) + len(env.vehicle.current_task)}, "
                  f"Total: {info['total_tasks']}, "
                  f"Success: {info['successful_tasks']}, "
                  f"Failed: {info['failed_tasks']}, "
                  f"Rate: {info['success_rate']*100:.2f}%")
    

    final_results = {
        'total_tasks': env.total_tasks,
        'successful_tasks': env.successful_tasks,
        'failed_tasks': env.failed_tasks,
        'success_rate': env.successful_tasks / max(1, env.total_tasks),
        'completed_tasks': env.completed_tasks,
        'history': history,
        'rewards':cumulative_reward
    }
    
    if verbose:
        print("\nLocal baseline test completed!")
        print(f"Total tasks: {final_results['total_tasks']}")
        print(f"Successful: {final_results['successful_tasks']}")
        print(f"Failed: {final_results['failed_tasks']}")
        print(f"Success rate: {final_results['success_rate']*100:.2f}%")
        print(f"Total rewards: {final_results['rewards']}")
        

        completion_times = [task.total_time for task in env.completed_tasks if task.success]
        if completion_times:
            print(f"Avg completion time: {np.mean(completion_times):.3f}s")
            print(f"Min completion time: {min(completion_times):.3f}s")
            print(f"Max completion time: {max(completion_times):.3f}s")
    
    return final_results

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

    completed_tasks = results['completed_tasks']
    

    all_sizes = [task.size/1024/1024 for task in completed_tasks] 
    

    successful_sizes = [task.size/1024/1024 for task in completed_tasks if task.success]
    failed_sizes = [task.size/1024/1024 for task in completed_tasks if not task.success]
    
    print("\nTask size analysis:")
    print(f"Average task size: {np.mean(all_sizes):.2f}MB")
    
    if successful_sizes:
        print(f"Successful task avg: {np.mean(successful_sizes):.2f}MB")
    
    if failed_sizes:
        print(f"Failed task avg: {np.mean(failed_sizes):.2f}MB")
    

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
    plt.savefig('image/local_baseline_task_size_distribution.png', dpi=300)

def plot_local_baseline_results(results):

    history = results['history']
    

    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False    
    

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    

    axs[0, 0].plot(history['time'], history['total_tasks'], label='总任务数')
    axs[0, 0].plot(history['time'], history['successful_tasks'], label='成功任务数')
    axs[0, 0].plot(history['time'], history['failed_tasks'], label='失败任务数')
    axs[0, 0].set_xlabel('时间 (秒)')
    axs[0, 0].set_ylabel('任务数')
    axs[0, 0].set_title('任务统计')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    

    axs[0, 1].plot(history['time'], [rate*100 for rate in history['success_rate']])
    axs[0, 1].set_xlabel('时间 (秒)')
    axs[0, 1].set_ylabel('成功率 (%)')
    axs[0, 1].set_title('任务成功率')
    axs[0, 1].grid(True)
    

    axs[1, 0].plot(history['time'], history['vehicle_position'])
    axs[1, 0].set_xlabel('时间 (秒)')
    axs[1, 0].set_ylabel('位置 (米)')
    axs[1, 0].set_title('车辆位置')
    axs[1, 0].grid(True)
    

    ax4 = axs[1, 1]
    ax4.plot(history['time'], [cache/1024/1024 for cache in history['cache_available']], label='可用缓存 (MB)')
    ax4_2 = ax4.twinx()
    ax4_2.plot(history['time'], history['queue_length'], 'r-', label='队列长度')
    ax4.set_xlabel('时间 (秒)')
    ax4.set_ylabel('可用缓存 (MB)')
    ax4_2.set_ylabel('队列长度')
    ax4.set_title('资源使用情况')
    ax4.grid(True)
    

    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_2.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    

    plt.tight_layout()
    plt.savefig('image/local_baseline_results.png', dpi=300)
    plt.show()

if __name__ == "__main__":

    print("===== 运行本地处理基线测试 =====")
    local_results = run_local_baseline(simulation_time=None, seed=42)
    

    analyze_task_failures(local_results)
    

    analyze_task_sizes(local_results)
    

    plot_local_baseline_results(local_results) 