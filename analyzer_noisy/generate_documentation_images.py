"""
ドキュメント用の画像例を生成するスクリプト
各処理段階での画像を可視化してassetフォルダに保存
"""

import cv2
import numpy as np
import os
from pathlib import Path
import sys

# 親ディレクトリのモジュールをインポート
sys.path.append(str(Path(__file__).parent.parent / 'analyzer'))
from image_utils import detect_heart_angle, recognize_character
import preprocessing

def create_output_dir():
    """出力ディレクトリを作成"""
    output_dir = Path(__file__).parent / 'asset'
    output_dir.mkdir(exist_ok=True)
    return output_dir

def load_sample_image(dataset_dir):
    """サンプル画像を読み込み"""
    dataset_path = Path(dataset_dir)
    
    # データセットから最初の画像を取得
    files = list(dataset_path.glob('image_*.png'))
    
    if files:
        # 最初の画像を使用
        return {'sample': str(files[0])}
    
    return {}

def visualize_preprocessing_steps(image_path, output_dir):
    """前処理の各ステップを可視化"""
    # 画像読み込み
    img = cv2.imread(image_path)
    if img is None:
        print(f"画像を読み込めません: {image_path}")
        return
    
    # ファイル名から情報を取得
    filename = Path(image_path).stem
    noise_type = 'example'
    
    # 1. 元画像
    cv2.imwrite(str(output_dir / f'01_original_{noise_type}.png'), img)
    
    # 2. ハート領域の切り出し
    HEART_REGION = (80, 280, 320, 520)
    x1, y1, x2, y2 = HEART_REGION
    heart_region = img[y1:y2, x1:x2].copy()
    cv2.imwrite(str(output_dir / f'02_heart_region_{noise_type}.png'), heart_region)
    
    # 3. グレースケール変換
    gray = cv2.cvtColor(heart_region, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(output_dir / f'03_grayscale_{noise_type}.png'), gray)
    
    # 4. メディアンフィルタ適用
    median_filtered = cv2.medianBlur(gray, 3)
    cv2.imwrite(str(output_dir / f'04_median_filter_{noise_type}.png'), median_filtered)
    
    # 5. Otsuの二値化
    _, otsu_binary = cv2.threshold(median_filtered, 0, 255, 
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite(str(output_dir / f'05_otsu_binary_{noise_type}.png'), otsu_binary)
    
    # 6. 文字領域の処理（右上）
    TOP_RIGHT = (540, 40, 760, 260)
    x1, y1, x2, y2 = TOP_RIGHT
    char_region = img[y1:y2, x1:x2].copy()
    cv2.imwrite(str(output_dir / f'06_char_region_{noise_type}.png'), char_region)
    
    # 7. 文字領域の前処理
    char_gray = cv2.cvtColor(char_region, cv2.COLOR_BGR2GRAY)
    char_inverted = 255 - char_gray
    cv2.imwrite(str(output_dir / f'07_char_inverted_{noise_type}.png'), char_inverted)
    
    # 8. 文字の適応的二値化
    char_binary = cv2.adaptiveThreshold(char_inverted, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(str(output_dir / f'08_char_binary_{noise_type}.png'), char_binary)
    
    print(f"前処理ステップの画像を生成しました: {noise_type}")

def visualize_template_matching(image_path, output_dir):
    """テンプレートマッチングの結果を可視化"""
    img = cv2.imread(image_path)
    if img is None:
        return
    
    filename = Path(image_path).stem
    noise_type = 'example'
    
    # テンプレートディレクトリ
    heart_template_dir = Path(__file__).parent.parent / 'analyzer' / 'heart_templates'
    char_template_dir = Path(__file__).parent.parent / 'analyzer' / 'templates'
    
    # 1. ハートテンプレートの例（0°, 90°, 180°, 270°）
    angles = [0, 90, 180, 270]
    template_examples = []
    
    for angle in angles:
        template_path = heart_template_dir / f'heart_{angle:03d}.png'
        if template_path.exists():
            template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
            template_examples.append(template)
    
    # テンプレートを横に並べて保存
    if template_examples:
        combined = np.hstack(template_examples)
        cv2.imwrite(str(output_dir / '09_heart_templates_examples.png'), combined)
    
    # 2. 文字テンプレートの例（A-F、角度0度のみ）
    char_examples = []
    for char in ['A', 'B', 'C', 'D', 'E', 'F']:
        template_path = char_template_dir / f'{char}_000.png'
        if template_path.exists():
            template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
            char_examples.append(template)
    
    if char_examples:
        # 横一列に配置
        combined = np.hstack(char_examples)
        cv2.imwrite(str(output_dir / '10_char_templates_examples.png'), combined)
    
    # 3. マッチング結果のヒートマップ（簡易版）
    HEART_REGION = (80, 280, 320, 520)
    x1, y1, x2, y2 = HEART_REGION
    heart_region = img[y1:y2, x1:x2].copy()
    gray = cv2.cvtColor(heart_region, cv2.COLOR_BGR2GRAY)
    processed = cv2.medianBlur(gray, 3)
    _, binary = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 最もマッチする角度のテンプレートでマッチング
    best_match_angle = None
    best_match_val = -1
    best_result = None
    
    for angle in range(0, 360, 10):
        template_path = heart_template_dir / f'heart_{angle:03d}.png'
        if not template_path.exists():
            continue
        
        template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            continue
        
        result = cv2.matchTemplate(binary, template, cv2.TM_CCORR_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        if max_val > best_match_val:
            best_match_val = max_val
            best_match_angle = angle
            best_result = result
    
    # マッチング結果をヒートマップとして可視化
    if best_result is not None:
        # 正規化して0-255に変換
        heatmap = cv2.normalize(best_result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # カラーマップ適用
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        cv2.imwrite(str(output_dir / f'11_matching_heatmap_{noise_type}.png'), heatmap_colored)
        
        # マッチング位置を元画像に描画
        h, w = cv2.imread(str(heart_template_dir / f'heart_{best_match_angle:03d}.png'), 
                          cv2.IMREAD_GRAYSCALE).shape
        _, _, _, max_loc = cv2.minMaxLoc(best_result)
        
        result_img = heart_region.copy()
        cv2.rectangle(result_img, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 0), 2)
        cv2.putText(result_img, f'{best_match_angle} deg', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite(str(output_dir / f'12_matching_result_{noise_type}.png'), result_img)
    
    print(f"テンプレートマッチングの画像を生成しました: {noise_type}")

def create_comparison_image(output_dir):
    """適応的閾値 vs Otsu の比較画像を作成"""
    # サンプル画像を探す
    dataset_path = Path(__file__).parent.parent / 'dataset_noisy'
    sample_files = list(dataset_path.glob('*.png'))[:1]
    
    if not sample_files:
        print("比較用のサンプル画像が見つかりません")
        return
    
    img = cv2.imread(str(sample_files[0]))
    HEART_REGION = (80, 280, 320, 520)
    x1, y1, x2, y2 = HEART_REGION
    heart_region = img[y1:y2, x1:x2].copy()
    
    gray = cv2.cvtColor(heart_region, cv2.COLOR_BGR2GRAY)
    processed = cv2.medianBlur(gray, 3)
    
    # 適応的閾値
    adaptive_binary = cv2.adaptiveThreshold(processed, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)
    
    # Otsuの二値化
    _, otsu_binary = cv2.threshold(processed, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # ラベルを追加
    adaptive_labeled = adaptive_binary.copy()
    otsu_labeled = otsu_binary.copy()
    
    cv2.putText(adaptive_labeled, 'Adaptive Threshold', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
    cv2.putText(otsu_labeled, 'Otsu Threshold', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
    
    # 横に並べて保存
    comparison = np.hstack([adaptive_labeled, otsu_labeled])
    cv2.imwrite(str(output_dir / '13_threshold_comparison.png'), comparison)
    
    print("閾値手法の比較画像を生成しました")

def main():
    """メイン処理"""
    print("ドキュメント用画像の生成を開始します...")
    
    # 出力ディレクトリ作成
    output_dir = create_output_dir()
    print(f"出力先: {output_dir}")
    
    # データセットディレクトリ
    dataset_dir = Path(__file__).parent.parent / 'dataset_noisy'
    
    # サンプル画像を取得
    samples = load_sample_image(dataset_dir)
    
    if not samples:
        print("サンプル画像が見つかりません")
        return
    
    print(f"{len(samples)}種類のノイズタイプのサンプルを見つけました")
    
    # 1つのサンプルで詳細な処理ステップを可視化
    first_sample = list(samples.values())[0]
    print(f"\n詳細な処理ステップを可視化: {Path(first_sample).name}")
    visualize_preprocessing_steps(first_sample, output_dir)
    visualize_template_matching(first_sample, output_dir)
    
    # 閾値手法の比較画像
    create_comparison_image(output_dir)
    
    print("\n完了！生成された画像:")
    for img_file in sorted(output_dir.glob('*.png')):
        print(f"  - {img_file.name}")

if __name__ == '__main__':
    main()
