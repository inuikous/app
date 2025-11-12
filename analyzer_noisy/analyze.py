"""
ノイズデータセット解析プログラム
dataset_noisy/ フォルダ内の画像を解析
"""

import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import sys

os.chdir(os.path.dirname(__file__))

# 現在のスクリプトのディレクトリを取得
current_dir = Path(__file__).parent

# テンプレートは analyzer フォルダのものを使用
template_dir = current_dir.parent / 'analyzer' / 'templates'

from image_utils import (
    detect_heart_angle,
    recognize_character,
    draw_results_on_image,
    generate_accuracy_plots,
    load_templates
)


# 領域定義（analyzer/analyze.py と同じ）
HEART_REGION = (180, 100, 420, 340)  # (left, top, right, bottom)
TOP_RIGHT_REGION = (480, 180, 600, 300)  # 右上の円
BOTTOM_RIGHT_REGION = (480, 340, 600, 460)  # 右下の円


def extract_ground_truth_from_filename(filename):
    """
    ファイル名から正解データを抽出
    
    フォーマット: image_{index}_heart{angle}_top{letter}{angle}_bottom{letter}{angle}.png
    例: image_0000_heart327_topA12_bottomF140.png 
        → heart_angle=327, top_letter='A', bottom_letter='F'
    
    Args:
        filename: 画像ファイル名
    
    Returns:
        dict: {heart_angle, top_letter, bottom_letter}
    """
    try:
        # 拡張子を除去
        name_without_ext = filename.replace('.png', '').replace('.jpg', '')
        
        # '_' で分割
        parts = name_without_ext.split('_')
        
        if len(parts) >= 5 and parts[0] == 'image':
            # parts[1] = インデックス（使用しない）
            # parts[2] = "heart327" → 327度
            # parts[3] = "topA12" → 文字A（角度は無視）
            # parts[4] = "bottomF140" → 文字F（角度は無視）
            
            # ハート角度を抽出: "heart327" → 327
            heart_part = parts[2]
            if heart_part.startswith('heart'):
                heart_angle = float(heart_part[5:])  # "heart" の5文字を除去
            else:
                return None
            
            # 右上文字を抽出: "topA12" → 'A'
            top_part = parts[3]
            if top_part.startswith('top') and len(top_part) > 3:
                top_letter = top_part[3]  # "top" の次の1文字
            else:
                return None
            
            # 右下文字を抽出: "bottomF140" → 'F'
            bottom_part = parts[4]
            if bottom_part.startswith('bottom') and len(bottom_part) > 6:
                bottom_letter = bottom_part[6]  # "bottom" の次の1文字
            else:
                return None
            
            return {
                'heart_angle': heart_angle,
                'top_letter': top_letter,
                'bottom_letter': bottom_letter
            }
    except Exception as e:
        print(f"ファイル名解析エラー: {filename} - {e}")
    
    return None


def analyze_single_image(image_path, ground_truth=None, save_visualization=False, output_dir=None):
    """
    1枚の画像を解析
    
    Args:
        image_path: 画像ファイルのパス
        ground_truth: 正解データの辞書（オプション）
        save_visualization: 可視化結果を保存するか
        output_dir: 可視化結果の保存先ディレクトリ
    
    Returns:
        dict: 解析結果
    """
    # 画像読み込み
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # RGB変換（必要なら）
    if img_array.ndim == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    results = {}
    
    # ハート領域を抽出して角度を検出
    heart_region = img_array[HEART_REGION[1]:HEART_REGION[3], 
                             HEART_REGION[0]:HEART_REGION[2]]
    heart_angle = detect_heart_angle(heart_region)
    results['heart_angle'] = heart_angle
    
    # 右上の円領域を抽出して文字を認識
    top_region = img_array[TOP_RIGHT_REGION[1]:TOP_RIGHT_REGION[3],
                           TOP_RIGHT_REGION[0]:TOP_RIGHT_REGION[2]]
    top_char, top_score = recognize_character(top_region)
    results['top_character'] = top_char
    results['top_score'] = top_score
    
    # 右下の円領域を抽出して文字を認識
    bottom_region = img_array[BOTTOM_RIGHT_REGION[1]:BOTTOM_RIGHT_REGION[3],
                              BOTTOM_RIGHT_REGION[0]:BOTTOM_RIGHT_REGION[2]]
    bottom_char, bottom_score = recognize_character(bottom_region)
    results['bottom_character'] = bottom_char
    results['bottom_score'] = bottom_score
    
    # 正解データとの比較
    if ground_truth:
        # ハート角度の誤差計算
        gt_angle = ground_truth['heart_angle']
        angle_diff = abs(heart_angle - gt_angle)
        
        # 角度差が180度を超える場合は、逆方向の差を使う
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        results['heart_angle_error'] = angle_diff
        
        # 文字認識の正誤
        results['top_correct'] = (top_char == ground_truth['top_letter'])
        results['bottom_correct'] = (bottom_char == ground_truth['bottom_letter'])
    
    # 可視化
    if save_visualization and output_dir:
        # 結果を画像に描画
        output_image = draw_results_on_image(img_array, results, ground_truth)
        
        # 保存
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)
        
        output_image_pil = Image.fromarray(output_image)
        output_image_pil.save(output_path)
    
    return results


def analyze_dataset(dataset_dir='dataset_noisy', output_dir='analyzer_noisy/outputs', 
                   save_images=True, max_images=None):
    """
    データセット全体を解析
    
    Args:
        dataset_dir: データセットディレクトリ
        output_dir: 結果出力ディレクトリ
        save_images: 可視化画像を保存するか
        max_images: 処理する最大画像数（None=全て）
    
    Returns:
        pandas.DataFrame: 解析結果
    """
    # テンプレートを事前読み込み
    print("テンプレートを読み込み中...")
    load_templates(str(template_dir))
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    if save_images:
        image_output_dir = os.path.join(output_dir, 'images')
        os.makedirs(image_output_dir, exist_ok=True)
    else:
        image_output_dir = None
    
    # 画像ファイル一覧を取得
    image_files = sorted([f for f in os.listdir(dataset_dir) 
                         if f.endswith('.png') or f.endswith('.jpg')])
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"解析対象: {len(image_files)} 枚")
    
    # 解析結果を格納
    results_list = []
    
    for i, filename in enumerate(image_files, 1):
        # 進捗表示
        if i % 10 == 0:
            print(f"進捗: {i}/{len(image_files)}")
        
        # 画像パス
        image_path = os.path.join(dataset_dir, filename)
        
        # 正解データを抽出
        ground_truth = extract_ground_truth_from_filename(filename)
        
        # 解析実行
        result = analyze_single_image(
            image_path, 
            ground_truth=ground_truth,
            save_visualization=save_images,
            output_dir=image_output_dir
        )
        
        # ファイル名を追加
        result['filename'] = filename
        
        # 正解データも追加
        if ground_truth:
            result['gt_heart_angle'] = ground_truth['heart_angle']
            result['gt_top_letter'] = ground_truth['top_letter']
            result['gt_bottom_letter'] = ground_truth['bottom_letter']
        
        results_list.append(result)
    
    # DataFrameに変換
    df = pd.DataFrame(results_list)
    
    # 結果をCSVで保存
    csv_path = os.path.join(output_dir, 'analysis_results.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ 解析結果を保存: {csv_path}")
    
    # 統計情報を表示
    print("\n" + "="*60)
    print("解析統計（ノイズデータセット）")
    print("="*60)
    
    if 'heart_angle_error' in df.columns:
        print(f"\nハート角度:")
        print(f"  平均誤差: {df['heart_angle_error'].mean():.2f}°")
        print(f"  最大誤差: {df['heart_angle_error'].max():.2f}°")
        print(f"  標準偏差: {df['heart_angle_error'].std():.2f}°")
    
    if 'top_correct' in df.columns:
        top_accuracy = (df['top_correct'].sum() / len(df)) * 100
        print(f"\n右上文字認識精度: {top_accuracy:.2f}% ({df['top_correct'].sum()}/{len(df)})")
    
    if 'bottom_correct' in df.columns:
        bottom_accuracy = (df['bottom_correct'].sum() / len(df)) * 100
        print(f"右下文字認識精度: {bottom_accuracy:.2f}% ({df['bottom_correct'].sum()}/{len(df)})")
    
    print("\n" + "="*60)
    
    # グラフ生成
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    generate_accuracy_plots(df, plot_dir)
    
    return df


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ノイズデータセット解析プログラム')
    parser.add_argument('--dataset', type=str, default='../dataset_noisy',
                       help='データセットディレクトリ')
    parser.add_argument('--output', type=str, default='outputs',
                       help='結果出力ディレクトリ')
    parser.add_argument('--save-images', action='store_false',
                       help='可視化画像を保存')
    parser.add_argument('--max-images', type=int, default=None,
                       help='処理する最大画像数')
    
    args = parser.parse_args()
    
    # 解析実行
    df = analyze_dataset(
        dataset_dir=args.dataset,
        output_dir=args.output,
        save_images=args.save_images,
        max_images=args.max_images
    )
    
    print("\n✓ 解析完了")


if __name__ == '__main__':
    main()
