"""
快速评估结果诊断工具
自动分析评估结果，给出改进建议
"""

import json
import os
import pandas as pd
import numpy as np
import sys


def load_evaluation_results(checkpoint_dir):
    """加载评估结果"""
    results_dir = os.path.join(checkpoint_dir, "evaluation_results")
    
    # 加载指标
    metrics_path = os.path.join(results_dir, "evaluation_metrics.json")
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # 加载目标值
    csv_path = os.path.join(results_dir, "pareto_front_objectives.csv")
    objectives_df = pd.read_csv(csv_path)
    
    return metrics, objectives_df


def analyze_hypervolume(hv_score, baseline_hv=None):
    """分析 Hypervolume"""
    print("\n" + "="*80)
    print("【1】Hypervolume 分析")
    print("="*80)
    
    print(f"您的 Hypervolume: {hv_score:,.2f}")
    
    if baseline_hv is not None:
        improvement = (hv_score - baseline_hv) / baseline_hv * 100
        print(f"基线 Hypervolume: {baseline_hv:,.2f}")
        print(f"提升幅度: {improvement:+.2f}%")
        
        if improvement > 20:
            print("✅ 评价：优秀！显著优于基线")
            score = 5
        elif improvement > 10:
            print("✅ 评价：良好！明显优于基线")
            score = 4
        elif improvement > 0:
            print("✅ 评价：及格，略优于基线")
            score = 3
        elif improvement > -10:
            print("⚠️ 评价：一般，接近基线")
            score = 2
        else:
            print("❌ 评价：较差，低于基线")
            score = 1
    else:
        print("⚠️ 未提供基线数据，无法进行比较")
        score = 0
    
    return score


def analyze_sparsity(sparsity_score):
    """分析 Sparsity"""
    print("\n" + "="*80)
    print("【2】Sparsity 分析（分布均匀性）")
    print("="*80)
    
    print(f"您的 Sparsity: {sparsity_score:,.2f}")
    
    if sparsity_score < 1000:
        print("✅ 评价：优秀！解分布非常均匀")
        score = 5
    elif sparsity_score < 5000:
        print("✅ 评价：良好！解分布比较均匀")
        score = 4
    elif sparsity_score < 10000:
        print("✅ 评价：及格，解分布基本均匀")
        score = 3
    elif sparsity_score < 20000:
        print("⚠️ 评价：一般，解分布不够均匀")
        score = 2
    else:
        print("❌ 评价：较差，解分布很不均匀，存在明显空白区域")
        score = 1
    
    return score


def analyze_objectives(objectives_df, baseline_makespan=None, baseline_tardiness=None):
    """分析目标值"""
    print("\n" + "="*80)
    print("【3】目标值分析")
    print("="*80)
    
    # 提取目标值（注意是负值）
    makespans = -objectives_df["Makespan (negative)"].values
    tardiness = -objectives_df["Total Tardiness (negative)"].values
    
    avg_makespan = np.mean(makespans)
    avg_tardiness = np.mean(tardiness)
    
    print(f"平均 Makespan: {avg_makespan:.2f}")
    print(f"平均 Tardiness: {avg_tardiness:.2f}")
    
    print(f"\nMakespan 范围: [{np.min(makespans):.2f}, {np.max(makespans):.2f}]")
    print(f"Tardiness 范围: [{np.min(tardiness):.2f}, {np.max(tardiness):.2f}]")
    
    score_makespan = 3
    score_tardiness = 3
    
    if baseline_makespan is not None:
        gap_makespan = (avg_makespan - baseline_makespan) / baseline_makespan * 100
        print(f"\n与基线 Makespan 比较: {gap_makespan:+.2f}%")
        
        if gap_makespan < -5:
            print("  ✅ Makespan 显著优于基线")
            score_makespan = 5
        elif gap_makespan < 0:
            print("  ✅ Makespan 略优于基线")
            score_makespan = 4
        elif gap_makespan < 5:
            print("  ⚠️ Makespan 略差于基线（可接受）")
            score_makespan = 3
        elif gap_makespan < 10:
            print("  ⚠️ Makespan 明显差于基线")
            score_makespan = 2
        else:
            print("  ❌ Makespan 显著差于基线")
            score_makespan = 1
    
    if baseline_tardiness is not None:
        gap_tardiness = (avg_tardiness - baseline_tardiness) / baseline_tardiness * 100
        print(f"与基线 Tardiness 比较: {gap_tardiness:+.2f}%")
        
        if gap_tardiness < -5:
            print("  ✅ Tardiness 显著优于基线")
            score_tardiness = 5
        elif gap_tardiness < 0:
            print("  ✅ Tardiness 略优于基线")
            score_tardiness = 4
        elif gap_tardiness < 5:
            print("  ⚠️ Tardiness 略差于基线（可接受）")
            score_tardiness = 3
        elif gap_tardiness < 10:
            print("  ⚠️ Tardiness 明显差于基线")
            score_tardiness = 2
        else:
            print("  ❌ Tardiness 显著差于基线")
            score_tardiness = 1
    
    return (score_makespan + score_tardiness) / 2


def analyze_extreme_preferences(objectives_df):
    """分析极端偏好的表现"""
    print("\n" + "="*80)
    print("【4】极端偏好分析")
    print("="*80)
    
    # 提取目标值
    makespans = -objectives_df["Makespan (negative)"].values
    tardiness = -objectives_df["Total Tardiness (negative)"].values
    w_makespan = objectives_df["Preference_Weight_Makespan"].values
    
    # 找到极端偏好
    idx_makespan_focus = np.argmax(w_makespan)  # w = [1.0, 0.0] 或接近
    idx_tardiness_focus = np.argmin(w_makespan)  # w = [0.0, 1.0] 或接近
    
    best_makespan = np.min(makespans)
    best_tardiness = np.min(tardiness)
    
    makespan_at_makespan_pref = makespans[idx_makespan_focus]
    tardiness_at_tardiness_pref = tardiness[idx_tardiness_focus]
    
    print(f"关注 Makespan 的偏好 (w≈[1.0, 0.0]):")
    print(f"  实际 Makespan: {makespan_at_makespan_pref:.2f}")
    print(f"  所有偏好中最优 Makespan: {best_makespan:.2f}")
    
    if np.abs(makespan_at_makespan_pref - best_makespan) < 10:
        print(f"  ✅ 结果：该偏好成功学习到优化 Makespan")
        score_makespan = 5
    elif np.abs(makespan_at_makespan_pref - best_makespan) < 50:
        print(f"  ✅ 结果：该偏好基本学习到优化 Makespan")
        score_makespan = 4
    else:
        print(f"  ⚠️ 结果：该偏好没有很好地优化 Makespan")
        score_makespan = 2
    
    print(f"\n关注 Tardiness 的偏好 (w≈[0.0, 1.0]):")
    print(f"  实际 Tardiness: {tardiness_at_tardiness_pref:.2f}")
    print(f"  所有偏好中最优 Tardiness: {best_tardiness:.2f}")
    
    if np.abs(tardiness_at_tardiness_pref - best_tardiness) < 100:
        print(f"  ✅ 结果：该偏好成功学习到优化 Tardiness")
        score_tardiness = 5
    elif np.abs(tardiness_at_tardiness_pref - best_tardiness) < 500:
        print(f"  ✅ 结果：该偏好基本学习到优化 Tardiness")
        score_tardiness = 4
    else:
        print(f"  ⚠️ 结果：该偏好没有很好地优化 Tardiness")
        score_tardiness = 2
    
    return (score_makespan + score_tardiness) / 2


def check_domination(objectives_df):
    """检查是否有被支配的解"""
    print("\n" + "="*80)
    print("【5】Pareto 支配关系检查")
    print("="*80)
    
    # 提取目标值（转为最小化问题）
    makespans = -objectives_df["Makespan (negative)"].values
    tardiness = -objectives_df["Total Tardiness (negative)"].values
    
    n = len(makespans)
    dominated_count = 0
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # 检查 i 是否被 j 支配
                if makespans[j] <= makespans[i] and tardiness[j] <= tardiness[i]:
                    if makespans[j] < makespans[i] or tardiness[j] < tardiness[i]:
                        dominated_count += 1
                        print(f"⚠️ 解 {i} 被解 {j} 支配")
                        print(f"   解 {i}: Makespan={makespans[i]:.2f}, Tardiness={tardiness[i]:.2f}")
                        print(f"   解 {j}: Makespan={makespans[j]:.2f}, Tardiness={tardiness[j]:.2f}")
                        break
    
    if dominated_count == 0:
        print("✅ 未发现被支配的解，所有解都位于 Pareto 前沿")
        score = 5
    elif dominated_count < 3:
        print(f"⚠️ 发现 {dominated_count} 个被支配的解（少量，可接受）")
        score = 3
    else:
        print(f"❌ 发现 {dominated_count} 个被支配的解（较多，需要改进）")
        score = 1
    
    return score


def generate_recommendations(scores):
    """生成改进建议"""
    print("\n" + "="*80)
    print("【6】改进建议")
    print("="*80)
    
    avg_score = np.mean(list(scores.values()))
    
    if avg_score >= 4.5:
        print("🎉 总体评价：优秀！")
        print("✅ 当前模型已经达到很高的水平，可以部署使用。")
        print("✅ 建议：保存当前 checkpoint 作为最佳模型。")
    elif avg_score >= 3.5:
        print("👍 总体评价：良好！")
        print("✅ 当前模型表现不错，可以使用。")
        print("💡 可选改进方向：")
    elif avg_score >= 2.5:
        print("⚠️ 总体评价：及格")
        print("⚠️ 当前模型基本可用，但仍有改进空间。")
        print("💡 建议改进方向：")
    else:
        print("❌ 总体评价：需要改进")
        print("❌ 当前模型表现不佳，建议重新训练或调整超参数。")
        print("🔧 必须改进：")
    
    # 具体建议
    if scores.get('hypervolume', 3) < 3:
        print("  1. Hypervolume 偏低：")
        print("     - 延长训练时间（增加 iterations）")
        print("     - 增加偏好向量多样性")
        print("     - 调整奖励函数归一化")
    
    if scores.get('sparsity', 3) < 3:
        print("  2. Sparsity 偏高（分布不均）：")
        print("     - 训练时使用更多偏好向量")
        print("     - 添加 diversity bonus")
        print("     - 增加探索概率")
    
    if scores.get('objectives', 3) < 3:
        print("  3. 目标值表现不佳：")
        print("     - 检查奖励函数设计")
        print("     - 调整归一化基线（N1, N2）")
        print("     - 增加训练数据多样性")
    
    if scores.get('extreme_prefs', 3) < 3:
        print("  4. 极端偏好学习不足：")
        print("     - 确保训练时采样到极端偏好")
        print("     - 检查偏好向量是否正确传入模型")
        print("     - 增加极端偏好的训练权重")
    
    if scores.get('domination', 3) < 3:
        print("  5. 存在被支配的解：")
        print("     - 使用 Non-dominated Sorting")
        print("     - 添加 Pareto-aware 的训练目标")
        print("     - 后处理：移除被支配的解")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("多目标评估结果诊断工具")
    print("="*80)
    
    # 获取 checkpoint 目录
    if len(sys.argv) > 1:
        checkpoint_dir = sys.argv[1]
    else:
        # 默认使用最新的实验
        checkpoint_dir = input("请输入 checkpoint 目录路径（或按回车使用默认）：").strip()
        if not checkpoint_dir:
            # 使用默认路径
            from DFJSPT import dfjspt_params
            checkpoint_dir = (
                f"DFJSPT/training_results/J{dfjspt_params.n_jobs}_M{dfjspt_params.n_machines}_T{dfjspt_params.n_transbots}/"
                f"MyTrainable_DfjsptMaEnv_PDMORL_1792b_00000_0_2025-11-05_10-06-50/checkpoint_000018"
            )
    
    print(f"\n正在分析: {checkpoint_dir}")
    
    # 加载结果
    try:
        metrics, objectives_df = load_evaluation_results(checkpoint_dir)
    except Exception as e:
        print(f"❌ 加载评估结果失败: {e}")
        print("请先运行 python -m DFJSPT.dfjspt_test 生成评估结果")
        return
    
    # 基线数据（可选，手动输入）
    print("\n是否提供基线数据进行比较？(y/n)")
    use_baseline = input().strip().lower() == 'y'
    
    baseline_hv = None
    baseline_makespan = None
    baseline_tardiness = None
    
    if use_baseline:
        print("请输入基线 Hypervolume（或按回车跳过）：")
        hv_input = input().strip()
        if hv_input:
            baseline_hv = float(hv_input)
        
        print("请输入基线 Makespan（或按回车跳过）：")
        ms_input = input().strip()
        if ms_input:
            baseline_makespan = float(ms_input)
        
        print("请输入基线 Tardiness（或按回车跳过）：")
        td_input = input().strip()
        if td_input:
            baseline_tardiness = float(td_input)
    
    # 执行分析
    scores = {}
    
    scores['hypervolume'] = analyze_hypervolume(metrics['hypervolume'], baseline_hv)
    scores['sparsity'] = analyze_sparsity(metrics['sparsity'])
    scores['objectives'] = analyze_objectives(objectives_df, baseline_makespan, baseline_tardiness)
    scores['extreme_prefs'] = analyze_extreme_preferences(objectives_df)
    scores['domination'] = check_domination(objectives_df)
    
    # 生成建议
    generate_recommendations(scores)
    
    # 总结
    print("\n" + "="*80)
    print("评分总结（满分 5 分）")
    print("="*80)
    for key, score in scores.items():
        stars = "★" * int(score) + "☆" * (5 - int(score))
        print(f"{key:20s}: {score:.1f}/5.0  {stars}")
    
    avg_score = np.mean(list(scores.values()))
    print(f"\n{'总体评分':20s}: {avg_score:.1f}/5.0")
    
    print("\n" + "="*80)
    print("诊断完成！")
    print("="*80)
    print(f"\n详细解读请参考: docs/HOW_TO_INTERPRET_EVALUATION_RESULTS.md")


if __name__ == "__main__":
    main()
