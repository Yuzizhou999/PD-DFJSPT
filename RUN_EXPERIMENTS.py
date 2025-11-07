#!/usr/bin/env python3
"""
实验管理脚本：在服务器上运行多次实验
支持自定义参数、进度跟踪、结果保存
"""

import os
import sys
import json
import time
import subprocess
import argparse
from datetime import datetime
from pathlib import Path


class ExperimentManager:
    def __init__(self, config_file=None):
        self.config_file = config_file or "experiment_config.json"
        self.experiments = []
        self.results = []
        self.load_config()
    
    def load_config(self):
        """加载实验配置"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.experiments = config.get("experiments", [])
        else:
            print(f"[WARNING] 配置文件 {self.config_file} 不存在")
    
    def add_experiment(self, name, description, params):
        """添加新实验"""
        exp = {
            "name": name,
            "description": description,
            "params": params
        }
        self.experiments.append(exp)
    
    def run_experiment(self, exp_idx, exp_config, dry_run=False):
        """运行单个实验"""
        exp_name = exp_config["name"]
        params = exp_config["params"]
        
        print(f"\n{'='*80}")
        print(f"[EXP {exp_idx+1}/{len(self.experiments)}] 开始: {exp_name}")
        print(f"{'='*80}")
        print(f"描述: {exp_config.get('description', 'N/A')}")
        print(f"参数: {json.dumps(params, indent=2, ensure_ascii=False)}")
        
        # 修改 dfjspt_params.py
        print(f"\n[STEP 1] 修改参数...")
        self.update_dfjspt_params(params)
        
        # 运行训练
        if not dry_run:
            print(f"[STEP 2] 运行训练...")
            start_time = time.time()
            try:
                # 设置环境变量，让训练脚本知道当前实验名称
                env = os.environ.copy()
                env['DFJSPT_EXPERIMENT_NAME'] = exp_name
                
                result = subprocess.run(
                    [sys.executable, "-m", "DFJSPT.dfjspt_train"],
                    capture_output=True,
                    text=True,
                    timeout=None,
                    env=env  # 传递包含实验名称的环境变量
                )
                elapsed = time.time() - start_time
                
                if result.returncode == 0:
                    print(f"[SUCCESS] 训练完成，耗时 {elapsed/3600:.2f} 小时")
                    return {
                        "name": exp_name,
                        "status": "success",
                        "elapsed_hours": elapsed/3600,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    print(f"[ERROR] 训练失败")
                    print(f"STDOUT: {result.stdout[-500:]}")
                    print(f"STDERR: {result.stderr[-500:]}")
                    return {
                        "name": exp_name,
                        "status": "failed",
                        "error": result.stderr[-200:],
                        "timestamp": datetime.now().isoformat()
                    }
            except Exception as e:
                print(f"[ERROR] 执行异常: {e}")
                return {
                    "name": exp_name,
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        else:
            print(f"[DRY-RUN] 跳过实际训练")
            return {
                "name": exp_name,
                "status": "dry_run",
                "timestamp": datetime.now().isoformat()
            }
    
    def update_dfjspt_params(self, params):
        """更新 dfjspt_params.py 中的参数"""
        params_file = "DFJSPT/dfjspt_params.py"
        
        with open(params_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换参数
        for key, value in params.items():
            if isinstance(value, str):
                pattern = f'{key} = ".*?"'
                replacement = f'{key} = "{value}"'
            elif isinstance(value, bool):
                pattern = f'{key} = (True|False)'
                replacement = f'{key} = {str(value)}'
            else:
                pattern = f'{key} = [0-9.e-]+'
                replacement = f'{key} = {value}'
            
            import re
            content = re.sub(pattern, replacement, content)
        
        with open(params_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  已更新参数文件")
    
    def run_all_experiments(self, exp_indices=None, dry_run=False):
        """运行所有实验或指定的实验"""
        if not self.experiments:
            print("[ERROR] 没有要运行的实验")
            return
        
        # 确定要运行的实验
        if exp_indices:
            exps_to_run = [self.experiments[i] for i in exp_indices if i < len(self.experiments)]
        else:
            exps_to_run = self.experiments
        
        print(f"\n[INFO] 总共 {len(exps_to_run)} 个实验，将在 {len(exps_to_run)} 个 checkpoint 后完成")
        
        self.results = []
        start_total = time.time()
        
        for idx, exp in enumerate(exps_to_run):
            result = self.run_experiment(idx, exp, dry_run=dry_run)
            self.results.append(result)
        
        elapsed_total = time.time() - start_total
        self.print_summary(elapsed_total)
        self.save_results()
    
    def print_summary(self, elapsed_total):
        """打印实验总结"""
        print(f"\n{'='*80}")
        print(f"[SUMMARY] 实验总结")
        print(f"{'='*80}")
        print(f"总耗时: {elapsed_total/3600:.2f} 小时\n")
        
        for i, result in enumerate(self.results, 1):
            status = result.get("status", "unknown").upper()
            name = result.get("name", "unknown")
            
            if status == "SUCCESS":
                elapsed = result.get("elapsed_hours", 0)
                print(f"[{i}] {name}: SUCCESS (耗时 {elapsed:.2f} h)")
            elif status == "DRY_RUN":
                print(f"[{i}] {name}: DRY-RUN")
            else:
                error = result.get("error", "unknown")[:50]
                print(f"[{i}] {name}: {status} ({error})")
        
        success_count = sum(1 for r in self.results if r.get("status") == "success")
        print(f"\n成功: {success_count}/{len(self.results)}")
    
    def save_results(self):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"experiment_results_{timestamp}.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False)
        
        print(f"\n[INFO] 结果已保存到: {result_file}")
    
    def list_experiments(self):
        """列出所有实验"""
        print(f"\n{'='*80}")
        print(f"[EXPERIMENTS] 配置的实验列表")
        print(f"{'='*80}\n")
        
        for i, exp in enumerate(self.experiments):
            print(f"[{i}] {exp['name']}")
            print(f"    描述: {exp.get('description', 'N/A')}")
            print(f"    参数: {exp['params']}\n")


def main():
    parser = argparse.ArgumentParser(
        description="实验管理脚本：管理和运行多个实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 列出所有实验
  python experiment_manager.py --list

  # 运行所有实验
  python experiment_manager.py --run-all

  # 运行指定实验（0 和 1）
  python experiment_manager.py --run 0 1

  # 干运行模式（只显示参数，不实际运行）
  python experiment_manager.py --run-all --dry-run

  # 使用自定义配置文件
  python experiment_manager.py --config my_config.json --run-all

  # 添加新实验并运行
  python experiment_manager.py --add-exp exp_003 "测试实验" n_jobs=15 learning_rate=2e-5 --run-all
        """
    )
    
    parser.add_argument("--config", type=str, default="experiment_config.json",
                        help="配置文件路径 (默认: experiment_config.json)")
    parser.add_argument("--list", action="store_true",
                        help="列出所有配置的实验")
    parser.add_argument("--run-all", action="store_true",
                        help="运行所有实验")
    parser.add_argument("--run", type=int, nargs="+",
                        help="运行指定索引的实验 (例如: --run 0 2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="干运行模式（只显示参数，不实际运行）")
    parser.add_argument("--add-exp", type=str, nargs="+",
                        help="添加新实验: --add-exp NAME DESCRIPTION param1=value1 param2=value2 ...")
    
    args = parser.parse_args()
    
    # 初始化管理器
    manager = ExperimentManager(args.config)
    
    # 添加新实验
    if args.add_exp:
        if len(args.add_exp) < 3:
            print("[ERROR] --add-exp 需要至少 3 个参数: 名称 描述 参数...")
            return
        
        exp_name = args.add_exp[0]
        description = args.add_exp[1]
        params = {}
        
        for param_str in args.add_exp[2:]:
            if "=" in param_str:
                key, value = param_str.split("=")
                try:
                    params[key] = float(value) if "." in value else int(value)
                except ValueError:
                    params[key] = value
        
        manager.add_experiment(exp_name, description, params)
        print(f"[INFO] 已添加实验: {exp_name}")
    
    # 执行命令
    if args.list:
        manager.list_experiments()
    elif args.run_all:
        manager.run_all_experiments(dry_run=args.dry_run)
    elif args.run:
        manager.run_all_experiments(exp_indices=args.run, dry_run=args.dry_run)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
