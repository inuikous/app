"""
テンプレートマッチングの精度をテストデータで評価
"""
import os
import sys
import re
from pathlib import Path
from collections import defaultdict

# analyzeモジュールをインポート
sys.path.insert(0, os.path.dirname(__file__))
from analyze import process_directory

def extract_gt_from_filename(filename):
    """
    ファイル名からground truthを抽出
    例: test_image_0000_heart213_topF5_bottomC188.png
    """
    pattern = r'test_image_\d+_heart\d+_top([A-F])\d+_bottom([A-F])\d+\.png'
    match = re.match(pattern, filename)
    if match:
        return match.group(1), match.group(2)
    return None, None


def main():
    # テストデータセットのパス
    test_dir = Path(__file__).parent.parent / 'test_dataset'
    output_dir = Path(__file__).parent / 'outputs_template'
    
    if not test_dir.exists():
        print(f"❌ テストデータセットが見つかりません: {test_dir}")
        return
    
    print("="*70)
    print("テンプレートマッチング精度評価")
    print("="*70)
    print(f"テストデータ: {test_dir}")
    print(f"出力先: {output_dir}")
    print()
    
    # テンプレートマッチングで処理
    print("テンプレートマッチングで推論中...")
    results = process_directory(
        str(test_dir),
        str(output_dir),
        save_results=True,
        generate_plots=False
    )
    
    print(f"処理完了: {len(results)} 枚")
    print()
    
    # 精度評価
    total = len(results)
    heart_detected = 0
    top_detected = 0
    top_correct = 0
    bottom_detected = 0
    bottom_correct = 0
    
    confusion_top = defaultdict(lambda: defaultdict(int))
    confusion_bottom = defaultdict(lambda: defaultdict(int))
    
    errors = []
    
    for result in results:
        filename = result['filename']
        gt_top, gt_bottom = extract_gt_from_filename(filename)
        
        if gt_top is None:
            continue
        
        # ハート検出
        if result['heart_angle'] is not None:
            heart_detected += 1
        
        # 上の文字
        pred_top = result['top_character']
        if pred_top is not None and pred_top != 'Unknown':
            top_detected += 1
            if pred_top == gt_top:
                top_correct += 1
            else:
                errors.append(('上', gt_top, pred_top, filename))
            confusion_top[gt_top][pred_top] += 1
        else:
            confusion_top[gt_top]['未検出'] += 1
            errors.append(('上', gt_top, '未検出', filename))
        
        # 下の文字
        pred_bottom = result['bottom_character']
        if pred_bottom is not None and pred_bottom != 'Unknown':
            bottom_detected += 1
            if pred_bottom == gt_bottom:
                bottom_correct += 1
            else:
                errors.append(('下', gt_bottom, pred_bottom, filename))
            confusion_bottom[gt_bottom][pred_bottom] += 1
        else:
            confusion_bottom[gt_bottom]['未検出'] += 1
            errors.append(('下', gt_bottom, '未検出', filename))
    
    # 結果表示
    print("="*70)
    print("精度結果")
    print("="*70)
    print(f"\n総画像数: {total}")
    
    print(f"\nハート検出率: {heart_detected}/{total} ({100*heart_detected/total:.1f}%)")
    
    print(f"\n上の文字:")
    print(f"  検出率: {top_detected}/{total} ({100*top_detected/total:.1f}%)")
    print(f"  正解率: {top_correct}/{total} ({100*top_correct/total:.1f}%)")
    if top_detected > 0:
        print(f"  精度 (検出された中で正解): {top_correct}/{top_detected} ({100*top_correct/top_detected:.1f}%)")
    
    print(f"\n下の文字:")
    print(f"  検出率: {bottom_detected}/{total} ({100*bottom_detected/total:.1f}%)")
    print(f"  正解率: {bottom_correct}/{total} ({100*bottom_correct/total:.1f}%)")
    if bottom_detected > 0:
        print(f"  精度 (検出された中で正解): {bottom_correct}/{bottom_detected} ({100*bottom_correct/bottom_detected:.1f}%)")
    
    # 混同行列（上の文字）
    print(f"\n{'='*70}")
    print("上の文字 - 混同行列")
    print(f"{'='*70}")
    
    header = "GT \\ Pred"
    print(f"{header:<10}", end="")
    letters = ['A', 'B', 'C', 'D', 'E', 'F', '未検出']
    for letter in letters:
        print(f"{letter:>8}", end="")
    print()
    
    for gt in ['A', 'B', 'C', 'D', 'E', 'F']:
        print(f"{gt:<10}", end="")
        for pred in letters:
            count = confusion_top[gt][pred]
            print(f"{count:>8}", end="")
        print()
    
    # 混同行列（下の文字）
    print(f"\n{'='*70}")
    print("下の文字 - 混同行列")
    print(f"{'='*70}")
    
    print(f"{header:<10}", end="")
    for letter in letters:
        print(f"{letter:>8}", end="")
    print()
    
    for gt in ['A', 'B', 'C', 'D', 'E', 'F']:
        print(f"{gt:<10}", end="")
        for pred in letters:
            count = confusion_bottom[gt][pred]
            print(f"{count:>8}", end="")
        print()
    
    # 誤認識パターン Top 10
    print(f"\n{'='*70}")
    print("誤認識パターン Top 10")
    print(f"{'='*70}")
    
    error_counts = defaultdict(int)
    for pos, gt, pred, _ in errors:
        error_counts[(pos, gt, pred)] += 1
    
    sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
    for (pos, gt, pred), count in sorted_errors[:10]:
        print(f"  {pos}の文字 {gt} → {pred}: {count}回")
    
    # 結論
    print(f"\n{'='*70}")
    print("結論")
    print(f"{'='*70}")
    print(f"1. ハート検出率: {100*heart_detected/total:.1f}%")
    print(f"2. 上の文字 正解率: {100*top_correct/total:.1f}%")
    print(f"3. 下の文字 正解率: {100*bottom_correct/total:.1f}%")
    
    # D-FINEとの比較用
    print(f"\n{'='*70}")
    print("D-FINEとの比較")
    print(f"{'='*70}")
    print("D-FINE (物体検出):  上99.0% / 下98.4%")
    print(f"テンプレート:       上{100*top_correct/total:.1f}% / 下{100*bottom_correct/total:.1f}%")
    
    diff_top = (top_correct/total) - 0.99
    diff_bottom = (bottom_correct/total) - 0.984
    print(f"\n差分: 上 {diff_top*100:+.1f}% / 下 {diff_bottom*100:+.1f}%")
    
    if top_correct/total >= 0.99 and bottom_correct/total >= 0.984:
        print("✅ テンプレートマッチングがD-FINEと同等以上の精度")
    else:
        print("❌ D-FINEの方が高精度")


if __name__ == '__main__':
    main()
