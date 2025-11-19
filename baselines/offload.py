import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from environment import Vehicle, RSU, TaskPoolGenerator, Communication, VehicularEdgeEnvironment, TaskStatus

def find_nearest_rsu(vehicle_position, rsus):

    in_range_rsus = [(rsu.rsu_id, abs(rsu.position - vehicle_position)) 
                    for rsu in rsus if rsu.is_vehicle_in_coverage(vehicle_position) and rsu.failure == 0]
    
    if not in_range_rsus:
        return None
    

    nearest_rsu = min(in_range_rsus, key=lambda x: x[1])
    return nearest_rsu[0]  


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
    plt.savefig('image/offload_baseline_task_size_distribution.png', dpi=300)




def run_offload_baseline(vehicle=None, rsus=None, task_generator=None, communication=None,simulation_time=None, seed=None, verbose=True):

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
        'total_tasks': [],
        'successful_tasks': [],
        'failed_tasks': [],
        'success_rate': [],
        'transmitting_tasks': [],
        'rewards': [] 
    }
    

    cumulative_reward = 0
    

    if verbose:
        print("开始全部卸载基线测试...")
    
    while not env._check_terminal():

        if env.new_task:

            nearest_rsu_id = find_nearest_rsu(env.vehicle.position, env.rsus)
            
      
            if nearest_rsu_id is not None:
            
                rsu = next((r for r in env.rsus if r.rsu_id == nearest_rsu_id), None)
                if rsu and rsu.failure == 0:  
                    action = nearest_rsu_id
                else:  
                    action = 11
            else:  
                action = 11
        else:
          
            action = len(env.rsus) + 1
        
     
        next_state, reward, done, info = env.step(action=action)
        
 
        cumulative_reward += reward
        
    
        history['time'].append(env.current_time)
        history['vehicle_position'].append(env.vehicle.position)
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
        print("\nFull offloading baseline test completed!")
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

def plot_offload_baseline_results(results):

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
    
    
    axs[1, 1].plot(history['time'], history['transmitting_tasks'], 'g-')
    axs[1, 1].set_xlabel('时间 (秒)')
    axs[1, 1].set_ylabel('传输中任务数')
    axs[1, 1].set_title('传输中任务数')
    axs[1, 1].grid(True)
    
  
    plt.tight_layout()
    plt.savefig('image/offload_baseline_results.png', dpi=300)
    plt.show()

if __name__ == "__main__":
  
    print("===== 运行全部卸载基线测试 =====")
    offload_results = run_offload_baseline(simulation_time= None, seed=None)
    

    analyze_task_failures(offload_results)
    analyze_task_sizes(offload_results)

   
    plot_offload_baseline_results(offload_results) 