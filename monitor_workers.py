"""
实时监控 Worker 工作状态
用法: python monitor_workers.py
"""
import psutil
import time
import os
from pathlib import Path

def get_python_processes():
    """获取所有 Python 进程"""
    python_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower():
                python_procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return python_procs

def identify_ray_workers(procs):
    """识别 Ray Worker 进程"""
    workers = []
    driver = None
    
    for proc in procs:
        try:
            cmdline = proc.info.get('cmdline', [])
            if not cmdline:
                continue
            
            cmdline_str = ' '.join(cmdline)
            
            # 识别 Driver (主训练进程)
            if 'dfjspt_train.py' in cmdline_str:
                driver = proc
            # 识别 RolloutWorker
            elif 'ray::RolloutWorker' in cmdline_str or 'RolloutWorker' in cmdline_str:
                workers.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    return driver, workers

def format_bytes(bytes_val):
    """格式化字节为人类可读格式"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} TB"

def monitor_workers(duration=60, interval=2):
    """
    监控 Worker 进程
    
    Args:
        duration: 监控时长（秒）
        interval: 更新间隔（秒）
    """
    print("=" * 80)
    print("🔍 Ray Worker 实时监控")
    print("=" * 80)
    print(f"监控时长: {duration}秒，更新间隔: {interval}秒")
    print("按 Ctrl+C 停止监控\n")
    
    start_time = time.time()
    iteration = 0
    
    try:
        while time.time() - start_time < duration:
            iteration += 1
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("=" * 80)
            print(f"🔍 Ray Worker 监控 - 迭代 {iteration}")
            print(f"运行时间: {time.time() - start_time:.1f}秒")
            print("=" * 80)
            
            # 获取所有 Python 进程
            python_procs = get_python_processes()
            
            # 识别 Driver 和 Workers
            driver, workers = identify_ray_workers(python_procs)
            
            # 显示 Driver 信息
            if driver:
                try:
                    driver_cpu = driver.cpu_percent(interval=0.1)
                    driver_mem = driver.memory_info().rss
                    print(f"\n📊 Driver (主进程)")
                    print(f"  PID: {driver.pid}")
                    print(f"  CPU: {driver_cpu:.1f}%")
                    print(f"  内存: {format_bytes(driver_mem)}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    print("\n📊 Driver: 进程已结束或无访问权限")
            else:
                print("\n⚠️  未检测到 Driver 进程（训练可能未启动）")
            
            # 显示 Workers 信息
            if workers:
                print(f"\n👷 Workers ({len(workers)} 个):")
                print(f"{'PID':<8} {'CPU%':<8} {'内存':<12} {'状态'}")
                print("-" * 50)
                
                total_cpu = 0
                total_mem = 0
                active_count = 0
                
                for i, worker in enumerate(workers, 1):
                    try:
                        cpu = worker.cpu_percent(interval=0.1)
                        mem = worker.memory_info().rss
                        
                        # 判断是否活跃（CPU > 1%）
                        status = "🟢 工作中" if cpu > 1 else "🔴 空闲"
                        if cpu > 1:
                            active_count += 1
                        
                        print(f"{worker.pid:<8} {cpu:<8.1f} {format_bytes(mem):<12} {status}")
                        
                        total_cpu += cpu
                        total_mem += mem
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        print(f"{worker.pid:<8} {'N/A':<8} {'N/A':<12} ❌ 已结束")
                
                print("-" * 50)
                print(f"总计:    {total_cpu:<8.1f} {format_bytes(total_mem):<12}")
                print(f"\n活跃 Workers: {active_count}/{len(workers)}")
                
                # 性能提示
                if active_count == 0:
                    print("\n⚠️  警告: 所有 Workers 都处于空闲状态！")
                    print("   可能原因: 训练暂停、等待数据、或配置错误")
                elif active_count < len(workers):
                    print(f"\n💡 提示: {len(workers) - active_count} 个 Workers 空闲")
                else:
                    print("\n✅ 所有 Workers 正常工作")
            else:
                print("\n⚠️  未检测到 Worker 进程")
                print("   可能原因:")
                print("   1. 训练尚未启动 Worker")
                print("   2. num_workers = 0 (本地模式)")
                print("   3. Workers 已结束")
            
            # 检查训练日志
            log_path = Path("DFJSPT/training_results")
            if log_path.exists():
                latest_exp = max(
                    [p for p in log_path.rglob("progress.csv") if p.is_file()],
                    key=lambda p: p.stat().st_mtime,
                    default=None
                )
                if latest_exp:
                    print(f"\n📁 最新实验日志: {latest_exp.parent.name}")
                    print(f"   文件大小: {format_bytes(latest_exp.stat().st_size)}")
                    print(f"   修改时间: {time.ctime(latest_exp.stat().st_mtime)}")
            
            # 系统总体资源
            cpu_percent = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            print(f"\n💻 系统资源")
            print(f"  总 CPU: {cpu_percent:.1f}%")
            print(f"  总内存: {mem.percent:.1f}% ({format_bytes(mem.used)}/{format_bytes(mem.total)})")
            
            print(f"\n按 Ctrl+C 停止监控...")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n监控已停止")

def show_worker_history():
    """显示 Worker 历史工作记录（从训练日志）"""
    print("\n" + "=" * 80)
    print("📊 Worker 历史工作记录分析")
    print("=" * 80)
    
    log_path = Path("DFJSPT/training_results")
    if not log_path.exists():
        print("未找到训练结果目录")
        return
    
    # 找到最新的 progress.csv
    progress_files = list(log_path.rglob("progress.csv"))
    if not progress_files:
        print("未找到训练日志文件")
        return
    
    latest_progress = max(progress_files, key=lambda p: p.stat().st_mtime)
    print(f"\n分析文件: {latest_progress}")
    
    try:
        import pandas as pd
        df = pd.read_csv(latest_progress)
        
        print(f"\n训练迭代数: {len(df)}")
        
        if 'episodes_this_iter' in df.columns:
            total_episodes = df['episodes_this_iter'].sum()
            avg_episodes = df['episodes_this_iter'].mean()
            print(f"总 Episodes: {total_episodes}")
            print(f"平均每次迭代: {avg_episodes:.1f} episodes")
        
        if 'num_env_steps_sampled' in df.columns:
            total_steps = df['num_env_steps_sampled'].iloc[-1] if len(df) > 0 else 0
            print(f"总采样步数: {total_steps:,}")
        
        if 'timers/sample_time_ms' in df.columns:
            avg_sample_time = df['timers/sample_time_ms'].mean()
            print(f"平均采样时间: {avg_sample_time:.1f} ms")
        
        if 'num_env_steps_sampled_throughput_per_sec' in df.columns:
            avg_throughput = df['num_env_steps_sampled_throughput_per_sec'].mean()
            print(f"平均采样吞吐量: {avg_throughput:.1f} steps/秒")
            print(f"\n✅ Workers 工作效率: {'优秀' if avg_throughput > 3000 else '良好' if avg_throughput > 1500 else '需优化'}")
        
        # 检查 Worker 健康状态
        if 'num_healthy_workers' in df.columns:
            latest_healthy = df['num_healthy_workers'].iloc[-1] if len(df) > 0 else 0
            print(f"\n当前健康 Workers: {latest_healthy}")
        
    except ImportError:
        print("\n需要安装 pandas: pip install pandas")
    except Exception as e:
        print(f"\n读取日志文件出错: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="监控 Ray Workers 工作状态")
    parser.add_argument("--duration", type=int, default=60, help="监控时长（秒），默认60秒")
    parser.add_argument("--interval", type=int, default=2, help="更新间隔（秒），默认2秒")
    parser.add_argument("--history", action="store_true", help="显示历史工作记录")
    
    args = parser.parse_args()
    
    if args.history:
        show_worker_history()
    else:
        monitor_workers(duration=args.duration, interval=args.interval)
