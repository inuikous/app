"""
画像解析ユーティリティ
ハート角度検出、文字認識、文字角度検出
"""

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUIなしでプロット生成
import os
from pathlib import Path


# テンプレートマッチング用のグローバルキャッシュ
_TEMPLATES_CACHE = None


def load_templates(template_dir='templates'):
    """
    テンプレート画像を読み込んでキャッシュ
    
    Args:
        template_dir: テンプレートディレクトリのパス
    
    Returns:
        dict: {文字: [(角度, テンプレート画像), ...], ...}
    """
    global _TEMPLATES_CACHE
    
    if _TEMPLATES_CACHE is not None:
        return _TEMPLATES_CACHE
    
    templates = {}
    template_path = Path(template_dir)
    
    if not template_path.exists():
        print(f"警告: テンプレートディレクトリが見つかりません: {template_dir}")
        print("analyzer/template_generator.py を実行してテンプレートを生成してください。")
        return {}
    
    for char in ['A', 'B', 'C', 'D', 'E', 'F']:
        templates[char] = []
        
        # 各角度のテンプレートを読み込み
        for angle in range(0, 360, 10):
            filename = f"{char}_{angle:03d}.png"
            filepath = template_path / filename
            
            if filepath.exists():
                img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    templates[char].append((angle, img))
    
    _TEMPLATES_CACHE = templates
    print(f"✓ テンプレート読み込み完了: {sum(len(v) for v in templates.values())} 個")
    
    return templates


def detect_heart_angle(image_region):
    """
    ハートの角度を検出
    
    Args:
        image_region: ハート領域の画像（NumPy配列、RGB）
    
    Returns:
        角度（度）、0度=上向き、時計回り=正
    """
    # グレースケール変換
    if len(image_region.shape) == 3:
        gray = cv2.cvtColor(image_region, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_region
    
    # 二値化
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # 輪郭検出
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0
    
    # 最大の輪郭（ハート）
    heart_contour = max(contours, key=cv2.contourArea)
    
    # モーメント計算
    M = cv2.moments(heart_contour)
    if M['m00'] == 0:
        return 0.0
    
    # 重心
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    
    # 重心から最も遠い点を見つける（上部の丸い部分の頂点、t=180付近）
    max_dist = 0
    farthest_point = (cx, cy)
    
    for point in heart_contour:
        px, py = point[0]
        dist = math.sqrt((px - cx)**2 + (py - cy)**2)
        if dist > max_dist:
            max_dist = dist
            farthest_point = (px, py)
    
    # 重心から最も遠い点への方向
    dx = farthest_point[0] - cx
    dy = farthest_point[1] - cy
    
    # 角度計算（atan2: 右=0°, 下=90°, 左=180°/-180°, 上=-90°）
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    
    # データセット生成ロジック:
    # t=180の点（上部頂点）は回転前にy>0（上向き、atan2=90°）
    # angle度回転すると、その点の方向は(angle+90)°になる
    # したがって: dataset_angle = (atan2_angle - 90) % 360
    detected_angle = (angle_deg - 90) % 360
    
    return detected_angle


def recognize_character(image_region):
    """
    円内の文字を認識（テンプレートマッチング版）
    
    データセット構造: 黒円の中に白文字
    
    Args:
        image_region: 円領域の画像（NumPy配列、RGB）
    
    Returns:
        tuple: (文字 (A-F or Unknown), マッチングスコア (0.0-1.0))
    """
    # テンプレートを読み込み
    templates = load_templates()
    
    if not templates:
        # テンプレートがない場合は旧手法にフォールバック
        result = recognize_character_legacy(image_region)
        return (result, 0.0)  # 旧手法ではスコアなし
    
    # グレースケール変換
    if len(image_region.shape) == 3:
        gray = cv2.cvtColor(image_region, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_region
    
    # 画像を200x200にリサイズ（テンプレートと同サイズ）
    target_size = 200
    h, w = gray.shape
    
    # 常にリサイズ（アスペクト比は維持しない - 円なので問題なし）
    gray_resized = cv2.resize(gray, (target_size, target_size))
    
    # 全テンプレートとマッチング
    best_score = -1
    best_char = 'Unknown'
    best_angle = 0
    
    for char, template_list in templates.items():
        for angle, template in template_list:
            # テンプレートマッチング（正規化相関係数）
            result = cv2.matchTemplate(gray_resized, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            if max_val > best_score:
                best_score = max_val
                best_char = char
                best_angle = angle
    
    # スコアが低すぎる場合はUnknown
    if best_score < 0.5:  # 閾値（調整可能）
        return ('Unknown', best_score)
    
    return (best_char, best_score)


def recognize_character_legacy(image_region):
    """
    円内の文字を認識（Hu Momentsベース）
    
    データセット構造: 黒円の中に白文字
    
    Args:
        image_region: 円領域の画像（NumPy配列、RGB）
    
    Returns:
        文字 (A-F or Unknown)
    """
    # グレースケール変換
    if len(image_region.shape) == 3:
        gray = cv2.cvtColor(image_region, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_region
    
    # 黒円を検出
    _, circle_binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    # 円の輪郭を検出
    contours, _ = cv2.findContours(circle_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 'Unknown'
    
    # 最大の輪郭（円）
    circle_contour = max(contours, key=cv2.contourArea)
    
    # 円の中心と半径
    (cx, cy), radius = cv2.minEnclosingCircle(circle_contour)
    
    if radius < 30:
        return 'Unknown'
    
    # 円の内部マスク
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), int(radius * 0.85), 255, -1)
    
    # 白文字を抽出
    _, text_binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    text_in_circle = cv2.bitwise_and(text_binary, text_binary, mask=mask)
    
    # 文字輪郭を検出
    text_contours, _ = cv2.findContours(text_in_circle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not text_contours:
        return 'Unknown'
    
    # 全輪郭を結合
    all_points = np.vstack(text_contours)
    x, y, w, h = cv2.boundingRect(all_points)
    
    if w == 0 or h == 0 or w * h < 100:
        return 'Unknown'
    
    # 文字領域を抽出
    char_roi = text_in_circle[y:y+h, x:x+w]
    
    # 固定サイズにリサイズ（アスペクト比を保持しない）
    char_roi_resized = cv2.resize(char_roi, (50, 50), interpolation=cv2.INTER_AREA)
    
    # 回転不変な特徴
    # 1. Hu Moments
    moments = cv2.moments(char_roi_resized)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # 2. 穴の数（Euler数）- 輪郭の階層構造から判定
    _, hierarchy = cv2.findContours(char_roi_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0
    if hierarchy is not None:
        # 親を持つ輪郭の数 = 穴の数
        holes = np.sum(hierarchy[0, :, 3] != -1)
    
    # 3. 総ピクセル数（正規化）
    total_pixels = np.sum(char_roi_resized > 128) / (50 * 50)
    
    # 4. 周囲長と面積の比
    contours_resized, _ = cv2.findContours(char_roi_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_resized:
        perimeter = cv2.arcLength(contours_resized[0], True)
        area = cv2.contourArea(contours_resized[0])
        if area > 0:
            compactness = (perimeter * perimeter) / area
        else:
            compactness = 0
    else:
        compactness = 0
    
    # 統計ベースの識別ルール
    # 
    # 混同行列から判明した問題:
    # - A → B誤認識: Aの穴が2-3個検出される → 密度でも判定
    # - E → C誤認識: EとCのcompactnessが重なる → 密度で区別
    # - F → C/E誤認識: Fとの境界があいまい → 密度範囲を厳密化
    # 
    # 統計:
    # B: holes 2-3, density 0.298-0.464 (avg 0.346), compactness 17.7-18.9
    # D: holes 1, density 0.266-0.365 (avg 0.310), compactness 15.5-16.9
    # A: holes 1-3, density 0.253-0.284 (avg 0.279), compactness 37.4-42.5
    # C: holes 0, density 0.261-0.292 (avg 0.271), compactness 94.9-107.5
    # E: holes 0, density 0.204-0.259 (avg 0.229), compactness 105.7-119.4
    # F: holes 0, density 0.231-0.332 (avg 0.265), compactness 81.2-99.6
    
    # B: holes >= 2 AND 高密度
    # 問題: Aがholes=2-3になることがある → 密度で区別
    if holes >= 2 and total_pixels > 0.29:  # Aの上限(0.284)より上
        return 'B'
    
    # D: 1つの穴、非常にコンパクト（丸い）
    if holes == 1 and compactness < 20:
        return 'D'
    
    # A: 1つの穴 OR (複数穴だが密度が低い)
    if holes >= 1 and total_pixels < 0.29:
        return 'A'
    
    # 穴なし（C, E, F）
    if holes == 0:
        # compactnessと密度を組み合わせて分類
        # 
        # 統計（50サンプル）:
        # E: compactness 105.7-130.3 (avg 115.4), density 0.204-0.393 (avg 0.255)
        # C: compactness 91.5-107.5 (avg 100.5), density 0.261-0.292 (avg 0.274)
        # F: compactness 81.2-100.5 (avg 90.6), density 0.231-0.332 (avg 0.270)
        # 
        # 問題:
        # - C vs E: compactness 91-108 で重なる
        # - C vs F: compactness 91-100 で重なる
        # 
        # 解決策: compactnessをメイン、densityを補助的に使用
        
        # E: compactness > 108 OR (compactness > 100 AND density < 0.27)
        if compactness > 108:
            return 'E'
        if compactness > 100 and total_pixels < 0.27:
            return 'E'
        
        # C: compactness 92-108 AND density 0.26-0.29
        if compactness >= 92 and 0.26 <= total_pixels <= 0.29:
            return 'C'
        
        # F: それ以外（compactness < 100 が多い）
        return 'F'
    
    return 'Unknown'


def draw_results_on_image(image, heart_angle, top_char, bottom_char, 
                         top_score=None, bottom_score=None, ground_truth=None):
    """
    解析結果を画像に描画（正解データがあれば比較表示）
    
    Args:
        image: 元の画像（PIL Image）
        heart_angle: ハートの角度
        top_char: 右上の文字
        bottom_char: 右下の文字
        top_score: 右上文字のマッチングスコア（0.0-1.0）
        bottom_score: 右下文字のマッチングスコア（0.0-1.0）
        ground_truth: 正解データの辞書 {'heart_angle': int, 'top_char': str, 'bottom_char': str}
                     Noneの場合は予測結果のみ表示
    
    Returns:
        結果を描画した画像（PIL Image）
    """
    from PIL import ImageDraw, ImageFont
    
    # コピーを作成
    result_img = image.copy()
    draw = ImageDraw.Draw(result_img)
    
    # フォント（英語のみなので文字化けしない）
    try:
        font_title = ImageFont.truetype("arial.ttf", 20)
        font_large = ImageFont.truetype("arial.ttf", 18)
        font_small = ImageFont.truetype("arial.ttf", 16)
    except:
        font_title = ImageFont.load_default()
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # 背景付きテキスト描画関数
    def draw_text_with_bg(text, position, font, text_color=(0, 0, 0), bg_color=(255, 255, 255)):
        bbox = draw.textbbox(position, text, font=font)
        padding = 5
        draw.rectangle(
            [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding],
            fill=bg_color
        )
        draw.text(position, text, fill=text_color, font=font)
    
    y_offset = 10
    
    if ground_truth is not None:
        # 正解データがある場合：正解 vs 予測を表示
        
        # タイトル
        draw_text_with_bg("Ground Truth vs Prediction", (10, y_offset), font_title, 
                         text_color=(0, 0, 255))
        y_offset += 30
        
        # ハート角度
        gt_heart = ground_truth['heart_angle']
        heart_error = abs(heart_angle - gt_heart)
        if heart_error > 180:
            heart_error = 360 - heart_error
        
        heart_color = (0, 128, 0) if heart_error < 5 else (255, 0, 0)
        heart_text = f"Heart: {gt_heart}deg -> {heart_angle:.1f}deg (err {heart_error:.1f}deg)"
        draw_text_with_bg(heart_text, (10, y_offset), font_small, text_color=heart_color)
        y_offset += 25
        
        # 上部文字
        gt_top = ground_truth['top_char']
        top_match = (top_char == gt_top)
        top_color = (0, 128, 0) if top_match else (255, 0, 0)
        top_symbol = "OK" if top_match else "NG"
        score_text = f" (score: {top_score:.3f})" if top_score is not None else ""
        top_text = f"Top: {gt_top} -> {top_char} [{top_symbol}]{score_text}"
        draw_text_with_bg(top_text, (10, y_offset), font_small, text_color=top_color)
        y_offset += 25
        
        # 下部文字
        gt_bottom = ground_truth['bottom_char']
        bottom_match = (bottom_char == gt_bottom)
        bottom_color = (0, 128, 0) if bottom_match else (255, 0, 0)
        bottom_symbol = "OK" if bottom_match else "NG"
        score_text = f" (score: {bottom_score:.3f})" if bottom_score is not None else ""
        bottom_text = f"Bottom: {gt_bottom} -> {bottom_char} [{bottom_symbol}]{score_text}"
        draw_text_with_bg(bottom_text, (10, y_offset), font_small, text_color=bottom_color)
        
    else:
        # 正解データがない場合：予測結果のみ表示
        heart_text = f"Heart: {heart_angle:.1f}deg"
        score_top = f" (score: {top_score:.3f})" if top_score is not None else ""
        score_bottom = f" (score: {bottom_score:.3f})" if bottom_score is not None else ""
        top_text = f"Top: {top_char}{score_top}"
        bottom_text = f"Bottom: {bottom_char}{score_bottom}"
        
        draw_text_with_bg(heart_text, (10, y_offset), font_large)
        y_offset += 35
        draw_text_with_bg(top_text, (10, y_offset), font_small)
        y_offset += 30
        draw_text_with_bg(bottom_text, (10, y_offset), font_small)
    
    return result_img


def generate_accuracy_plots(results, output_dir):
    """
    精度グラフを生成
    
    Args:
        results: 解析結果のリスト (各要素は辞書で、gt_*, pred_*, error_heartなどを含む)
        output_dir: グラフ保存先ディレクトリ
    """
    import os
    from collections import defaultdict
    
    os.makedirs(output_dir, exist_ok=True)
    
    # データ集計
    total = len(results)
    heart_errors = []
    top_correct = 0
    bottom_correct = 0
    char_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    for r in results:
        # ハート角度誤差
        if 'error_heart' in r:
            heart_errors.append(r['error_heart'])
        
        # 上部文字
        if 'gt_top' in r and 'pred_top' in r:
            gt = r['gt_top']
            pred = r['pred_top']
            char_stats[gt]['total'] += 1
            confusion_matrix[gt][pred] += 1
            if gt == pred:
                top_correct += 1
                char_stats[gt]['correct'] += 1
        
        # 下部文字
        if 'gt_bottom' in r and 'pred_bottom' in r:
            gt = r['gt_bottom']
            pred = r['pred_bottom']
            char_stats[gt]['total'] += 1
            confusion_matrix[gt][pred] += 1
            if gt == pred:
                bottom_correct += 1
                char_stats[gt]['correct'] += 1
    
    # 1. 全体精度グラフ
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # ハート角度精度
    if heart_errors:
        axes[0].hist(heart_errors, bins=20, color='steelblue', edgecolor='black')
        axes[0].set_title('Heart Angle Error Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Error (degrees)', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].axvline(np.mean(heart_errors), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(heart_errors):.2f}°')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # 文字認識精度
    total_chars = top_correct + bottom_correct
    total_possible = total * 2
    char_acc = (total_chars / total_possible * 100) if total_possible > 0 else 0
    
    categories = ['Top Character', 'Bottom Character', 'Overall']
    accuracies = [
        (top_correct / total * 100) if total > 0 else 0,
        (bottom_correct / total * 100) if total > 0 else 0,
        char_acc
    ]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    bars = axes[1].bar(categories, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    axes[1].set_title('Character Recognition Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # バー上に数値表示
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.1f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 文字別精度
    chars = sorted(char_stats.keys())
    char_accuracies = [(char_stats[c]['correct'] / char_stats[c]['total'] * 100) 
                       if char_stats[c]['total'] > 0 else 0 
                       for c in chars]
    
    bars = axes[2].bar(chars, char_accuracies, color='#9b59b6', edgecolor='black', linewidth=1.5)
    axes[2].set_title('Per-Character Accuracy', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Character', fontsize=12)
    axes[2].set_ylabel('Accuracy (%)', fontsize=12)
    axes[2].set_ylim(0, 105)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # バー上に数値表示
    for bar, acc, char in zip(bars, char_accuracies, chars):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.0f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_accuracy.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 混同行列
    chars_sorted = sorted(set(list(confusion_matrix.keys()) + 
                             [k for d in confusion_matrix.values() for k in d.keys()]))
    
    if chars_sorted:
        matrix = np.zeros((len(chars_sorted), len(chars_sorted)))
        for i, true_char in enumerate(chars_sorted):
            for j, pred_char in enumerate(chars_sorted):
                matrix[i, j] = confusion_matrix[true_char][pred_char]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(matrix, cmap='Blues', aspect='auto')
        
        ax.set_xticks(np.arange(len(chars_sorted)))
        ax.set_yticks(np.arange(len(chars_sorted)))
        ax.set_xticklabels(chars_sorted, fontsize=12)
        ax.set_yticklabels(chars_sorted, fontsize=12)
        
        ax.set_xlabel('Predicted', fontsize=14, fontweight='bold')
        ax.set_ylabel('Ground Truth', fontsize=14, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        
        # セルに数値表示
        for i in range(len(chars_sorted)):
            for j in range(len(chars_sorted)):
                value = int(matrix[i, j])
                if value > 0:
                    color = 'white' if value > matrix.max() / 2 else 'black'
                    ax.text(j, i, str(value), ha='center', va='center',
                           color=color, fontsize=11, fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 3. 詳細統計テキスト
    stats_file = os.path.join(output_dir, 'statistics.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("ANALYSIS STATISTICS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total Images: {total}\n\n")
        
        if heart_errors:
            f.write("Heart Angle Detection:\n")
            f.write(f"  Mean Error: {np.mean(heart_errors):.2f}°\n")
            f.write(f"  Std Error:  {np.std(heart_errors):.2f}°\n")
            f.write(f"  Max Error:  {np.max(heart_errors):.2f}°\n")
            f.write(f"  Min Error:  {np.min(heart_errors):.2f}°\n\n")
        
        f.write("Character Recognition:\n")
        f.write(f"  Overall Accuracy: {char_acc:.2f}% ({total_chars}/{total_possible})\n")
        f.write(f"  Top Accuracy:     {(top_correct/total*100):.2f}% ({top_correct}/{total})\n")
        f.write(f"  Bottom Accuracy:  {(bottom_correct/total*100):.2f}% ({bottom_correct}/{total})\n\n")
        
        f.write("Per-Character Accuracy:\n")
        for char in chars:
            correct = char_stats[char]['correct']
            total_char = char_stats[char]['total']
            acc = (correct / total_char * 100) if total_char > 0 else 0
            f.write(f"  {char}: {acc:.1f}% ({correct}/{total_char})\n")
    
    print(f"✓ Plots saved to: {output_dir}")
    print(f"  - overall_accuracy.png")
    print(f"  - confusion_matrix.png")
    print(f"  - statistics.txt")


