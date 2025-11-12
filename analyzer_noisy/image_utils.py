"""
画像解析ユーティリティ（ノイズ対応強化版）
ハート角度検出、文字認識、文字角度検出
ノイズに対するロバスト性を向上
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


def preprocess_for_noise(image, noise_type='auto'):
    """
    ノイズに応じた前処理を適用
    
    Args:
        image: 入力画像（グレースケール）
        noise_type: ノイズタイプ（'auto', 'gaussian', 'salt_pepper', 'blur', 'shadow', 'vignette'）
    
    Returns:
        前処理済み画像
    """
    if noise_type == 'auto':
        # 複数の前処理を組み合わせて適用（軽量版）
        # 1. ガウシアンぼかしでノイズ除去
        denoised = cv2.GaussianBlur(image, (5, 5), 0)
        
        # 2. 適応的ヒストグラム等化（コントラスト・明るさ対策）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 3. Median filterで塩コショウノイズ除去
        smoothed = cv2.medianBlur(enhanced, 3)
        
        return smoothed
    
    elif noise_type == 'gaussian':
        # ガウシアンノイズ対策
        return cv2.GaussianBlur(image, (5, 5), 0)
    
    elif noise_type == 'salt_pepper':
        # 塩コショウノイズ対策
        return cv2.medianBlur(image, 5)
    
    elif noise_type == 'blur':
        # ぼかし対策（シャープネス強調）
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)
    
    elif noise_type in ['shadow', 'vignette']:
        # 照明ムラ対策（CLAHE）
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        return clahe.apply(image)
    
    else:
        return image


def normalize_illumination(image):
    """
    照明の正規化（影・ビネット対策）
    
    Args:
        image: 入力画像（グレースケール）
    
    Returns:
        正規化された画像
    """
    # 背景推定（大きなぼかし）
    background = cv2.GaussianBlur(image, (51, 51), 0)
    
    # 背景を除去
    normalized = cv2.divide(image.astype(float), background.astype(float) + 1e-6) * 255
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    
    return normalized


def detect_heart_angle(image_region):
    """
    ハートの角度を検出（ノイズ対応強化版）
    
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
    
    # 最小限の前処理: 塩コショウノイズ除去のみ
    processed = cv2.medianBlur(gray, 3)
    
    # 適応的二値化
    binary = cv2.adaptiveThreshold(
        processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 5
    )
    
    # モルフォロジー処理でノイズ除去
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 輪郭検出
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0
    
    # 最大の輪郭（ハート）
    heart_contour = max(contours, key=cv2.contourArea)
    
    # 輪郭の面積チェック（ノイズによる誤検出を防ぐ）
    area = cv2.contourArea(heart_contour)
    if area < 100:  # 小さすぎる輪郭は無視
        return 0.0
    
    # モーメント計算
    M = cv2.moments(heart_contour)
    if M['m00'] == 0:
        return 0.0
    
    # 重心
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    
    # 重心から最も遠い点を見つける（上部の丸い部分の頂点）
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
    
    # 角度計算
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    
    # データセット生成ロジックに合わせて調整
    detected_angle = (angle_deg - 90) % 360
    
    return detected_angle


def recognize_character(image_region):
    """
    円内の文字を認識（テンプレートマッチング版、ノイズ対応強化）
    
    Args:
        image_region: 円領域の画像（NumPy配列、RGB）
    
    Returns:
        tuple: (文字 (A-F or Unknown), マッチングスコア (0.0-1.0))
    """
    # テンプレートを読み込み
    templates = load_templates()
    
    if not templates:
        return ('Unknown', 0.0)
    
    # グレースケール変換
    if len(image_region.shape) == 3:
        gray = cv2.cvtColor(image_region, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_region
    
    # 最小限の前処理: 塩コショウノイズ除去のみ
    processed = cv2.medianBlur(gray, 3)
    
    # 円領域の抽出（エッジ検出ベース）
    # 適応的二値化
    binary = cv2.adaptiveThreshold(
        processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 5
    )
    
    # モルフォロジー処理
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 輪郭検出
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return ('Unknown', 0.0)
    
    # 最大の輪郭（円）を探す
    circle_contour = max(contours, key=cv2.contourArea)
    
    # バウンディングボックス
    x, y, w, h = cv2.boundingRect(circle_contour)
    
    # 円内部のマスクを作成
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [circle_contour], -1, 255, -1)
    
    # 文字領域を抽出（白文字）
    char_region = cv2.bitwise_and(binary, mask)
    
    # 文字部分だけを切り出し
    if w > 0 and h > 0:
        char_region_cropped = char_region[y:y+h, x:x+w]
        
        # リサイズしてテンプレートサイズに合わせる
        target_size = 60  # テンプレートのサイズ
        char_resized = cv2.resize(char_region_cropped, (target_size, target_size))
    else:
        return ('Unknown', 0.0)
    
    # テンプレートマッチング
    best_match = None
    best_score = 0
    
    for char, template_list in templates.items():
        for angle, template in template_list:
            # 正規化相互相関でマッチング
            result = cv2.matchTemplate(char_resized, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            if max_val > best_score:
                best_score = max_val
                best_match = char
    
    # スコアが低すぎる場合は Unknown
    if best_score < 0.5:  # 閾値を引き上げて誤認識を防ぐ
        return ('Unknown', best_score)
    
    return (best_match, best_score)


def draw_results_on_image(image, results, ground_truth=None):
    """
    画像に解析結果を描画
    
    Args:
        image: 元画像（NumPy配列、RGB）
        results: 解析結果の辞書
        ground_truth: 正解データの辞書（オプション）
    
    Returns:
        描画済み画像（NumPy配列、RGB）
    """
    # 画像をコピー
    output_image = image.copy()
    
    # テキスト描画の準備
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    # 結果を描画
    y_offset = 30
    
    # ハート角度
    text = f"Heart: {results['heart_angle']:.1f}°"
    if ground_truth:
        gt_angle = ground_truth.get('heart_angle', 0)
        error = abs(results['heart_angle'] - gt_angle)
        if error > 180:
            error = 360 - error
        text += f" (GT: {gt_angle:.1f}°, Err: {error:.1f}°)"
        color = (0, 255, 0) if error < 5 else (255, 165, 0) if error < 15 else (255, 0, 0)
    else:
        color = (255, 255, 255)
    
    cv2.putText(output_image, text, (10, y_offset), font, font_scale, color, thickness)
    y_offset += 35
    
    # 右上の文字
    text = f"Top: {results['top_character']} ({results['top_score']:.2f})"
    if ground_truth:
        gt_char = ground_truth.get('top_letter', 'Unknown')
        correct = results['top_character'] == gt_char
        text += f" (GT: {gt_char})"
        color = (0, 255, 0) if correct else (255, 0, 0)
    else:
        color = (255, 255, 255)
    
    cv2.putText(output_image, text, (10, y_offset), font, font_scale, color, thickness)
    y_offset += 35
    
    # 右下の文字
    text = f"Bottom: {results['bottom_character']} ({results['bottom_score']:.2f})"
    if ground_truth:
        gt_char = ground_truth.get('bottom_letter', 'Unknown')
        correct = results['bottom_character'] == gt_char
        text += f" (GT: {gt_char})"
        color = (0, 255, 0) if correct else (255, 0, 0)
    else:
        color = (255, 255, 255)
    
    cv2.putText(output_image, text, (10, y_offset), font, font_scale, color, thickness)
    
    return output_image


def generate_accuracy_plots(results_df, output_dir):
    """
    精度グラフを生成
    
    Args:
        results_df: pandas DataFrame（解析結果）
        output_dir: 出力ディレクトリ
    """
    # 統計情報を計算
    stats = {}
    
    # ハート角度の誤差
    if 'heart_angle_error' in results_df.columns:
        heart_errors = results_df['heart_angle_error'].dropna()
        stats['heart_mean_error'] = heart_errors.mean()
        stats['heart_max_error'] = heart_errors.max()
        stats['heart_std_error'] = heart_errors.std()
    
    # 文字認識精度
    if 'top_correct' in results_df.columns:
        stats['top_accuracy'] = (results_df['top_correct'].sum() / len(results_df)) * 100
    
    if 'bottom_correct' in results_df.columns:
        stats['bottom_accuracy'] = (results_df['bottom_correct'].sum() / len(results_df)) * 100
    
    # 統計情報を保存
    stats_path = os.path.join(output_dir, 'statistics.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write('解析統計（ノイズデータセット）\n')
        f.write('='*50 + '\n\n')
        
        if 'heart_mean_error' in stats:
            f.write(f"ハート角度:\n")
            f.write(f"  平均誤差: {stats['heart_mean_error']:.2f}°\n")
            f.write(f"  最大誤差: {stats['heart_max_error']:.2f}°\n")
            f.write(f"  標準偏差: {stats['heart_std_error']:.2f}°\n\n")
        
        if 'top_accuracy' in stats:
            f.write(f"右上文字認識精度: {stats['top_accuracy']:.2f}%\n")
        
        if 'bottom_accuracy' in stats:
            f.write(f"右下文字認識精度: {stats['bottom_accuracy']:.2f}%\n")
    
    print(f"✓ 統計情報を保存: {stats_path}")
