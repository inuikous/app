"""
精度評価結果をグラフ化するスクリプト
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import List, Dict, Tuple

# 日本語フォント設定
matplotlib.rcParams['font.sans-serif'] = ['Yu Gothic', 'MS Gothic', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def load_results(result_path: str, ground_truth_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    解析結果と正解データを読み込む
    
    Args:
        result_path: 解析結果のCSVパス
        ground_truth_path: 正解データのCSVパス
    
    Returns:
        (解析結果リスト, 正解データリスト)
    """
    # 解析結果を読み込み
    results = []
    with open(result_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    
    # 正解データを読み込み
    ground_truth = {}
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ground_truth[row['filename']] = row
    
    return results, ground_truth


def calculate_angle_error(angle1: float, angle2: float) -> float:
    """
    2つの角度の誤差を計算（360度の境界を考慮）
    
    Args:
        angle1: 角度1
        angle2: 角度2
    
    Returns:
        誤差（度）
    """
    error = abs(angle1 - angle2)
    if error > 180:
        error = 360 - error
    return error


def analyze_accuracy(results: List[Dict], ground_truth: Dict) -> Dict:
    """
    精度を分析する
    
    Args:
        results: 解析結果リスト
        ground_truth: 正解データ辞書
    
    Returns:
        分析結果の辞書
    """
    heart_angle_errors = []
    top_angle_errors = []
    bottom_angle_errors = []
    top_char_correct = []
    bottom_char_correct = []
    filenames = []
    
    for result in results:
        filename = result['filename']
        if filename not in ground_truth:
            continue
        
        gt = ground_truth[filename]
        filenames.append(filename)
        
        # ハート角度の誤差
        gt_heart_angle = float(gt['heart_angle'])
        result_heart_angle = float(result['heart_angle'])
        error = calculate_angle_error(result_heart_angle, gt_heart_angle)
        heart_angle_errors.append(error)
        
        # 右上文字の正解/不正解
        top_char_correct.append(1 if result['top_char'] == gt['top_letter'] else 0)
        
        # 右上角度の誤差
        gt_top_angle = float(gt['top_angle'])
        result_top_angle = float(result['top_angle'])
        error = calculate_angle_error(result_top_angle, gt_top_angle)
        top_angle_errors.append(error)
        
        # 右下文字の正解/不正解
        bottom_char_correct.append(1 if result['bottom_char'] == gt['bottom_letter'] else 0)
        
        # 右下角度の誤差
        gt_bottom_angle = float(gt['bottom_angle'])
        result_bottom_angle = float(result['bottom_angle'])
        error = calculate_angle_error(result_bottom_angle, gt_bottom_angle)
        bottom_angle_errors.append(error)
    
    return {
        'filenames': filenames,
        'heart_angle_errors': heart_angle_errors,
        'top_angle_errors': top_angle_errors,
        'bottom_angle_errors': bottom_angle_errors,
        'top_char_correct': top_char_correct,
        'bottom_char_correct': bottom_char_correct
    }


def plot_accuracy_graphs(analysis: Dict, output_dir: str):
    """
    精度グラフを生成する
    
    Args:
        analysis: 分析結果
        output_dir: 出力ディレクトリ
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. ハート角度誤差の分布
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('画像解析システム 精度評価', fontsize=16, fontweight='bold')
    
    # 1-1. ハート角度誤差のヒストグラム
    ax = axes[0, 0]
    ax.hist(analysis['heart_angle_errors'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('角度誤差 (度)', fontsize=12)
    ax.set_ylabel('画像数', fontsize=12)
    ax.set_title(f'ハート角度誤差の分布\n平均: {np.mean(analysis["heart_angle_errors"]):.2f}°', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 1-2. 右上角度誤差のヒストグラム
    ax = axes[0, 1]
    ax.hist(analysis['top_angle_errors'], bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
    ax.set_xlabel('角度誤差 (度)', fontsize=12)
    ax.set_ylabel('画像数', fontsize=12)
    ax.set_title(f'右上アルファベット角度誤差の分布\n平均: {np.mean(analysis["top_angle_errors"]):.2f}°', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 1-3. 右下角度誤差のヒストグラム
    ax = axes[1, 0]
    ax.hist(analysis['bottom_angle_errors'], bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
    ax.set_xlabel('角度誤差 (度)', fontsize=12)
    ax.set_ylabel('画像数', fontsize=12)
    ax.set_title(f'右下アルファベット角度誤差の分布\n平均: {np.mean(analysis["bottom_angle_errors"]):.2f}°', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 1-4. 文字認識正解率
    ax = axes[1, 1]
    labels = ['右上文字', '右下文字']
    accuracy = [
        np.mean(analysis['top_char_correct']) * 100,
        np.mean(analysis['bottom_char_correct']) * 100
    ]
    bars = ax.bar(labels, accuracy, color=['coral', 'lightseagreen'], edgecolor='black', alpha=0.7)
    ax.set_ylabel('正解率 (%)', fontsize=12)
    ax.set_title('文字認識正解率', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 正解率の値を表示
    for bar, acc in zip(bars, accuracy):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_summary.png'), dpi=150, bbox_inches='tight')
    print(f"保存: {os.path.join(output_dir, 'accuracy_summary.png')}")
    plt.close()
    
    # 2. 画像ごとの誤差推移
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('画像ごとの角度誤差推移', fontsize=16, fontweight='bold')
    
    x = range(len(analysis['filenames']))
    
    # 2-1. ハート角度誤差
    ax = axes[0]
    ax.plot(x, analysis['heart_angle_errors'], marker='o', linewidth=2, 
            color='steelblue', markersize=5, alpha=0.7)
    ax.axhline(y=np.mean(analysis['heart_angle_errors']), color='red', 
               linestyle='--', linewidth=2, label=f'平均: {np.mean(analysis["heart_angle_errors"]):.2f}°')
    ax.set_ylabel('誤差 (度)', fontsize=12)
    ax.set_title('ハート角度誤差', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2-2. 右上角度誤差
    ax = axes[1]
    ax.plot(x, analysis['top_angle_errors'], marker='s', linewidth=2, 
            color='indianred', markersize=5, alpha=0.7)
    ax.axhline(y=np.mean(analysis['top_angle_errors']), color='red', 
               linestyle='--', linewidth=2, label=f'平均: {np.mean(analysis["top_angle_errors"]):.2f}°')
    ax.set_ylabel('誤差 (度)', fontsize=12)
    ax.set_title('右上アルファベット角度誤差', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2-3. 右下角度誤差
    ax = axes[2]
    ax.plot(x, analysis['bottom_angle_errors'], marker='^', linewidth=2, 
            color='seagreen', markersize=5, alpha=0.7)
    ax.axhline(y=np.mean(analysis['bottom_angle_errors']), color='red', 
               linestyle='--', linewidth=2, label=f'平均: {np.mean(analysis["bottom_angle_errors"]):.2f}°')
    ax.set_xlabel('画像番号', fontsize=12)
    ax.set_ylabel('誤差 (度)', fontsize=12)
    ax.set_title('右下アルファベット角度誤差', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_transition.png'), dpi=150, bbox_inches='tight')
    print(f"保存: {os.path.join(output_dir, 'error_transition.png')}")
    plt.close()
    
    # 3. 統計情報のボックスプロット
    fig, ax = plt.subplots(figsize=(10, 8))
    
    data = [
        analysis['heart_angle_errors'],
        analysis['top_angle_errors'],
        analysis['bottom_angle_errors']
    ]
    labels = ['ハート角度', '右上角度', '右下角度']
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))
    
    # 各ボックスに色を付ける
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('角度誤差 (度)', fontsize=12)
    ax.set_title('角度誤差の統計分布（ボックスプロット）', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_boxplot.png'), dpi=150, bbox_inches='tight')
    print(f"保存: {os.path.join(output_dir, 'error_boxplot.png')}")
    plt.close()
    
    # 4. 詳細統計レポート
    print_statistics(analysis)


def print_statistics(analysis: Dict):
    """
    統計情報を表示する
    
    Args:
        analysis: 分析結果
    """
    print("\n" + "="*60)
    print("詳細統計レポート")
    print("="*60)
    
    print("\n【ハート角度誤差】")
    print(f"  平均誤差: {np.mean(analysis['heart_angle_errors']):.2f}°")
    print(f"  中央値: {np.median(analysis['heart_angle_errors']):.2f}°")
    print(f"  最大誤差: {np.max(analysis['heart_angle_errors']):.2f}°")
    print(f"  最小誤差: {np.min(analysis['heart_angle_errors']):.2f}°")
    print(f"  標準偏差: {np.std(analysis['heart_angle_errors']):.2f}°")
    
    print("\n【右上アルファベット】")
    print(f"  文字認識正解率: {np.mean(analysis['top_char_correct'])*100:.1f}%")
    print(f"  角度平均誤差: {np.mean(analysis['top_angle_errors']):.2f}°")
    print(f"  角度中央値: {np.median(analysis['top_angle_errors']):.2f}°")
    print(f"  角度最大誤差: {np.max(analysis['top_angle_errors']):.2f}°")
    print(f"  角度標準偏差: {np.std(analysis['top_angle_errors']):.2f}°")
    
    print("\n【右下アルファベット】")
    print(f"  文字認識正解率: {np.mean(analysis['bottom_char_correct'])*100:.1f}%")
    print(f"  角度平均誤差: {np.mean(analysis['bottom_angle_errors']):.2f}°")
    print(f"  角度中央値: {np.median(analysis['bottom_angle_errors']):.2f}°")
    print(f"  角度最大誤差: {np.max(analysis['bottom_angle_errors']):.2f}°")
    print(f"  角度標準偏差: {np.std(analysis['bottom_angle_errors']):.2f}°")
    
    print("\n【総合評価】")
    total_images = len(analysis['filenames'])
    print(f"  評価画像数: {total_images}")
    print(f"  全体平均角度誤差: {np.mean(analysis['heart_angle_errors'] + analysis['top_angle_errors'] + analysis['bottom_angle_errors']):.2f}°")
    print(f"  全体文字認識正解率: {(np.sum(analysis['top_char_correct']) + np.sum(analysis['bottom_char_correct'])) / (total_images * 2) * 100:.1f}%")
    print("="*60)


def main():
    """
    メイン関数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='精度評価グラフ生成')
    parser.add_argument('--results', '-r', 
                       default='outputs/analysis_results.csv',
                       help='解析結果のCSVファイル')
    parser.add_argument('--ground-truth', '-g',
                       default='../dataset/labels.csv',
                       help='正解データのCSVファイル')
    parser.add_argument('--output', '-o',
                       default='outputs/plots',
                       help='グラフ出力ディレクトリ')
    
    args = parser.parse_args()
    
    print("="*60)
    print("精度評価グラフ生成")
    print("="*60)
    print(f"解析結果: {args.results}")
    print(f"正解データ: {args.ground_truth}")
    print(f"出力先: {args.output}")
    print("="*60)
    print()
    
    # データを読み込み
    print("データを読み込んでいます...")
    results, ground_truth = load_results(args.results, args.ground_truth)
    print(f"✓ {len(results)}件の解析結果を読み込みました")
    print(f"✓ {len(ground_truth)}件の正解データを読み込みました")
    print()
    
    # 精度を分析
    print("精度を分析しています...")
    analysis = analyze_accuracy(results, ground_truth)
    print(f"✓ {len(analysis['filenames'])}件のデータを分析しました")
    print()
    
    # グラフを生成
    print("グラフを生成しています...")
    plot_accuracy_graphs(analysis, args.output)
    print()
    
    print("="*60)
    print("完了！")
    print(f"グラフは {args.output} に保存されました。")
    print("="*60)


if __name__ == '__main__':
    main()
