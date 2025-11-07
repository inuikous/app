"""
ユーティリティ関数
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from typing import List, Dict
from collections import Counter

def _get_japanese_font():
    """日本語対応フォントを取得"""
    font_candidates = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
    for font_name in font_candidates:
        try:
            font_path = font_manager.findfont(font_manager.FontProperties(family=font_name))
            if font_path:
                return font_name
        except:
            continue
    return None

japanese_font = _get_japanese_font()
if japanese_font:
    plt.rcParams['font.family'] = japanese_font
plt.rcParams['axes.unicode_minus'] = False


def calculate_angle_error(pred_angle: float, true_angle: float) -> float:
    """
    角度の誤差を計算（360度の境界を考慮）
    
    Args:
        pred_angle: 予測角度
        true_angle: 正解角度
    
    Returns:
        誤差（度）
    """
    error = abs(pred_angle - true_angle)
    if error > 180:
        error = 360 - error
    return error


def generate_accuracy_plots(results: List[Dict], output_dir: str):
    """
    精度評価グラフを生成
    
    Args:
        results: 解析結果のリスト
        output_dir: 出力ディレクトリ
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 正解データがある結果のみフィルタ
    results_with_gt = [r for r in results if 'gt_top' in r and 'gt_bottom' in r]
    
    if not results_with_gt:
        print("正解データがないため、グラフを生成できません。")
        return
    
    # 文字認識の正解/不正解を集計
    top_correct = []
    bottom_correct = []
    
    for r in results_with_gt:
        top_correct.append(1 if r['top_character'] == r['gt_top'] else 0)
        bottom_correct.append(1 if r['bottom_character'] == r['gt_bottom'] else 0)
    
    # グラフ作成
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 右上文字の正解率
    ax = axes[0, 0]
    accuracy = np.mean(top_correct) * 100
    ax.bar(['正解', '不正解'], 
           [np.sum(top_correct), len(top_correct) - np.sum(top_correct)],
           color=['green', 'red'], alpha=0.7)
    ax.set_title(f'右上文字認識正解率: {accuracy:.1f}%', fontsize=14, fontweight='bold')
    ax.set_ylabel('画像数', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # 2. 右下文字の正解率
    ax = axes[0, 1]
    accuracy = np.mean(bottom_correct) * 100
    ax.bar(['正解', '不正解'],
           [np.sum(bottom_correct), len(bottom_correct) - np.sum(bottom_correct)],
           color=['green', 'red'], alpha=0.7)
    ax.set_title(f'右下文字認識正解率: {accuracy:.1f}%', fontsize=14, fontweight='bold')
    ax.set_ylabel('画像数', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # 3. 文字別正解率（右上）
    ax = axes[1, 0]
    letters = ['A', 'B', 'C', 'D', 'E', 'F']
    letter_accuracy = {}
    for letter in letters:
        letter_results = [r for r in results_with_gt if r['gt_top'] == letter]
        if letter_results:
            correct = sum(1 for r in letter_results if r['top_character'] == letter)
            letter_accuracy[letter] = correct / len(letter_results) * 100
        else:
            letter_accuracy[letter] = 0
    
    ax.bar(letters, [letter_accuracy[l] for l in letters], 
           color='steelblue', alpha=0.7)
    ax.set_title('右上文字別正解率', fontsize=14, fontweight='bold')
    ax.set_xlabel('文字', fontsize=12)
    ax.set_ylabel('正解率 (%)', fontsize=12)
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    for i, letter in enumerate(letters):
        ax.text(i, letter_accuracy[letter] + 2, f'{letter_accuracy[letter]:.1f}%',
               ha='center', fontsize=10)
    
    # 4. 文字別正解率（右下）
    ax = axes[1, 1]
    letter_accuracy = {}
    for letter in letters:
        letter_results = [r for r in results_with_gt if r['gt_bottom'] == letter]
        if letter_results:
            correct = sum(1 for r in letter_results if r['bottom_character'] == letter)
            letter_accuracy[letter] = correct / len(letter_results) * 100
        else:
            letter_accuracy[letter] = 0
    
    ax.bar(letters, [letter_accuracy[l] for l in letters],
           color='coral', alpha=0.7)
    ax.set_title('右下文字別正解率', fontsize=14, fontweight='bold')
    ax.set_xlabel('文字', fontsize=12)
    ax.set_ylabel('正解率 (%)', fontsize=12)
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    for i, letter in enumerate(letters):
        ax.text(i, letter_accuracy[letter] + 2, f'{letter_accuracy[letter]:.1f}%',
               ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # 保存
    output_path = os.path.join(output_dir, 'accuracy_summary.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  保存: {output_path}")
    plt.close()
    
    # 統計情報をテキストファイルに保存
    stats_path = os.path.join(output_dir, 'statistics.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("D-FINE 精度評価結果\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"総画像数: {len(results_with_gt)}\n\n")
        
        f.write("【右上文字認識】\n")
        f.write(f"  全体正解率: {np.mean(top_correct)*100:.2f}%\n")
        f.write(f"  正解数: {np.sum(top_correct)}/{len(top_correct)}\n\n")
        
        f.write("【右下文字認識】\n")
        f.write(f"  全体正解率: {np.mean(bottom_correct)*100:.2f}%\n")
        f.write(f"  正解数: {np.sum(bottom_correct)}/{len(bottom_correct)}\n\n")
        
        f.write("="*60 + "\n")
    
    print(f"  保存: {stats_path}")


def create_confusion_matrix(results: List[Dict], position: str = 'top') -> np.ndarray:
    """
    混同行列を作成
    
    Args:
        results: 解析結果のリスト
        position: 'top' or 'bottom'
    
    Returns:
        混同行列（numpy配列）
    """
    letters = ['A', 'B', 'C', 'D', 'E', 'F']
    letter_to_idx = {l: i for i, l in enumerate(letters)}
    
    matrix = np.zeros((len(letters), len(letters)), dtype=int)
    
    gt_key = f'gt_{position}'
    pred_key = f'{position}_character'
    
    for r in results:
        if gt_key in r and pred_key in r:
            gt = r[gt_key]
            pred = r[pred_key]
            if gt in letter_to_idx and pred in letter_to_idx:
                matrix[letter_to_idx[gt], letter_to_idx[pred]] += 1
    
    return matrix


if __name__ == '__main__':
    print("ユーティリティモジュール")
    print("角度誤差計算とグラフ生成の関数を提供します。")
