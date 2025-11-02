"""
メイン解析スクリプト
入力ディレクトリの画像を解析し、結果を出力ディレクトリに保存
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import csv
from typing import List, Dict

# スクリプトとして実行される場合のための相対インポート対応
if __name__ == '__main__':
    # 親ディレクトリをパスに追加
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .image_utils import (
        detect_heart_angle,
        recognize_character,
        draw_results_on_image,
        generate_accuracy_plots
    )
except ImportError:
    # スクリプトとして直接実行された場合
    from image_utils import (
        detect_heart_angle,
        recognize_character,
        draw_results_on_image,
        generate_accuracy_plots
    )

# 画像領域の定義（データセット生成と一致）
# データセット: Heart(200,400), Top円(650,150), Bottom円(650,450), 半径100
HEART_REGION_TOP = 250
HEART_REGION_BOTTOM = 550
HEART_REGION_LEFT = 50
HEART_REGION_RIGHT = 350

TOP_RIGHT_REGION_TOP = 50
TOP_RIGHT_REGION_BOTTOM = 250
TOP_RIGHT_REGION_LEFT = 550
TOP_RIGHT_REGION_RIGHT = 750

BOTTOM_RIGHT_REGION_TOP = 350
BOTTOM_RIGHT_REGION_BOTTOM = 550
BOTTOM_RIGHT_REGION_LEFT = 550
BOTTOM_RIGHT_REGION_RIGHT = 750


def analyze_single_image(image_path: str) -> Dict:
    """
    1枚の画像を解析する
    
    Args:
        image_path: 画像ファイルのパス
    
    Returns:
        解析結果の辞書
    """
    # 画像を読み込み
    pil_image = Image.open(image_path)
    cv_image_rgb = np.array(pil_image)
    
    # 各領域を抽出して解析
    # 1. ハート領域（左下）
    heart_region = cv_image_rgb[
        HEART_REGION_TOP:HEART_REGION_BOTTOM,
        HEART_REGION_LEFT:HEART_REGION_RIGHT
    ]
    heart_angle = detect_heart_angle(heart_region)
    
    # 2. 右上アルファベット
    top_region = cv_image_rgb[
        TOP_RIGHT_REGION_TOP:TOP_RIGHT_REGION_BOTTOM,
        TOP_RIGHT_REGION_LEFT:TOP_RIGHT_REGION_RIGHT
    ]
    top_char, top_score = recognize_character(top_region)
    
    # 3. 右下アルファベット
    bottom_region = cv_image_rgb[
        BOTTOM_RIGHT_REGION_TOP:BOTTOM_RIGHT_REGION_BOTTOM,
        BOTTOM_RIGHT_REGION_LEFT:BOTTOM_RIGHT_REGION_RIGHT
    ]
    bottom_char, bottom_score = recognize_character(bottom_region)
    
    return {
        'heart_angle': heart_angle,
        'top_character': top_char,
        'bottom_character': bottom_char,
        'top_score': top_score,
        'bottom_score': bottom_score
    }


def extract_ground_truth_from_filename(filename: str) -> Dict:
    """
    データセットのファイル名から正解データを抽出
    
    ファイル名形式: image_XXXX_heartYYY_topZnn_bottomWmm.png
    - XXXX: 画像番号
    - YYY: ハート角度
    - Z: 上部文字（A-F）
    - nn: 上部文字の角度
    - W: 下部文字（A-F）
    - mm: 下部文字の角度
    
    Args:
        filename: ファイル名
    
    Returns:
        正解データの辞書、または抽出できない場合はNone
    """
    try:
        # 拡張子を除去
        name_without_ext = filename.replace('.png', '').replace('.jpg', '').replace('.jpeg', '').replace('.bmp', '')
        
        # アンダースコアで分割
        parts = name_without_ext.split('_')
        
        # データセット形式かチェック
        if len(parts) >= 5 and 'heart' in parts[2] and 'top' in parts[3] and 'bottom' in parts[4]:
            # ハート角度
            heart_angle = int(parts[2].replace('heart', ''))
            
            # 上部文字
            top_part = parts[3].replace('top', '')
            top_char = top_part[0]  # 最初の文字
            
            # 下部文字
            bottom_part = parts[4].replace('bottom', '')
            bottom_char = bottom_part[0]  # 最初の文字
            
            return {
                'heart_angle': heart_angle,
                'top_char': top_char,
                'bottom_char': bottom_char
            }
    except:
        pass
    
    return None


def process_directory(input_dir: str, output_dir: str, 
                     save_results: bool = True, generate_plots: bool = True) -> List[Dict]:
    """
    ディレクトリ内の全画像を処理する
    
    Args:
        input_dir: 入力ディレクトリ
        output_dir: 出力ディレクトリ (outputs/images と outputs/plots を作成)
        save_results: 結果を保存するかどうか
        generate_plots: 精度グラフを生成するかどうか
    
    Returns:
        全画像の解析結果リスト
    """
    # 出力ディレクトリを作成
    if save_results:
        images_dir = os.path.join(output_dir, 'images')
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(images_dir, exist_ok=True)
        if generate_plots:
            os.makedirs(plots_dir, exist_ok=True)
    
    # 画像ファイルを取得
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
    
    # 重複を削除
    image_files = list(set(image_files))
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"警告: {input_dir} に画像ファイルが見つかりませんでした。")
        return []
    
    print(f"{len(image_files)}枚の画像を処理します...\n")
    
    results = []
    
    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] 処理中: {image_path.name}")
        
        try:
            # 画像を解析
            result = analyze_single_image(str(image_path))
            
            # ファイル名を追加
            result['filename'] = image_path.name
            
            # ファイル名から正解データを抽出（データセットの場合）
            ground_truth = extract_ground_truth_from_filename(image_path.name)
            
            # 正解データがあれば精度計算用に保存
            if ground_truth:
                result['gt_heart'] = ground_truth['heart_angle']
                result['gt_top'] = ground_truth['top_char']
                result['gt_bottom'] = ground_truth['bottom_char']
                result['pred_heart'] = result['heart_angle']
                result['pred_top'] = result['top_character']
                result['pred_bottom'] = result['bottom_character']
                
                # 誤差計算
                heart_error = abs(result['heart_angle'] - ground_truth['heart_angle'])
                if heart_error > 180:
                    heart_error = 360 - heart_error
                result['error_heart'] = heart_error
            
            results.append(result)
            
            # 結果を表示
            print(f"  ハート角度: {result['heart_angle']:.1f}°")
            print(f"  右上文字: {result['top_character']}")
            print(f"  右下文字: {result['bottom_character']}")
            
            if save_results:
                # 元の画像を読み込み
                pil_image = Image.open(image_path)
                
                # 結果を描画（正解データがあれば比較表示）
                result_image = draw_results_on_image(
                    pil_image,
                    result['heart_angle'],
                    result['top_character'],
                    result['bottom_character'],
                    top_score=result.get('top_score'),
                    bottom_score=result.get('bottom_score'),
                    ground_truth=ground_truth
                )
                
                # 保存（images サブディレクトリに）
                output_path = Path(images_dir) / image_path.name
                result_image.save(output_path)
                print(f"  → 保存: {output_path}")
            
            print()
            
        except Exception as e:
            print(f"  エラー: {e}\n")
            continue
    
    # グラフ生成
    if save_results and generate_plots and results:
        print("\n" + "="*60)
        print("グラフを生成中...")
        print("="*60)
        generate_accuracy_plots(results, plots_dir)
    
    return results


def save_results_csv(results: List[Dict], output_path: str):
    """
    解析結果をCSVファイルに保存
    
    Args:
        results: 解析結果のリスト
        output_path: 出力CSVファイルのパス
    """
    if not results:
        print("保存する結果がありません。")
        return
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        # すべての結果に含まれるフィールドを取得
        if results:
            fieldnames = list(results[0].keys())
        else:
            fieldnames = ['filename', 'heart_angle', 'top_character', 'bottom_character']
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n解析結果を保存しました: {output_path}")


def compare_with_ground_truth(results: List[Dict], ground_truth_path: str):
    """
    正解データと比較して精度を評価
    
    Args:
        results: 解析結果のリスト
        ground_truth_path: 正解データのCSVファイルパス
    """
    if not os.path.exists(ground_truth_path):
        print(f"警告: 正解データが見つかりません: {ground_truth_path}")
        return
    
    # 正解データを読み込み
    ground_truth = {}
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ground_truth[row['filename']] = row
    
    # 精度を計算
    total = 0
    heart_angle_errors = []
    top_char_correct = 0
    bottom_char_correct = 0
    
    for result in results:
        filename = result['filename']
        if filename not in ground_truth:
            continue
        
        gt = ground_truth[filename]
        total += 1
        
        # ハート角度の誤差
        gt_heart_angle = float(gt['heart_angle'])
        error = abs(result['heart_angle'] - gt_heart_angle)
        # 360度の境界を考慮
        if error > 180:
            error = 360 - error
        heart_angle_errors.append(error)
        
        # 右上文字の正解率
        if result['top_character'] == gt.get('top_character', gt.get('top_letter', '')):
            top_char_correct += 1
        
        # 右下文字の正解率
        if result['bottom_character'] == gt.get('bottom_character', gt.get('bottom_letter', '')):
            bottom_char_correct += 1
    
    # 結果を表示
    print("\n" + "="*60)
    print("精度評価結果")
    print("="*60)
    print(f"評価対象画像数: {total}")
    print()
    
    if heart_angle_errors:
        print(f"ハート角度:")
        print(f"  平均誤差: {np.mean(heart_angle_errors):.2f}°")
        print(f"  最大誤差: {np.max(heart_angle_errors):.2f}°")
        print(f"  標準偏差: {np.std(heart_angle_errors):.2f}°")
    
    if total > 0:
        print(f"\n右上文字:")
        print(f"  正解率: {top_char_correct/total*100:.1f}% ({top_char_correct}/{total})")
    
    if total > 0:
        print(f"\n右下文字:")
        print(f"  正解率: {bottom_char_correct/total*100:.1f}% ({bottom_char_correct}/{total})")
    
    print("="*60)


def main():
    """
    メイン関数
    引数なしで実行した場合はデモモードで動作
    """
    parser = argparse.ArgumentParser(description='画像解析システム')
    parser.add_argument('--input', '-i', default=None,
                       help='入力ディレクトリ（デフォルト: デモモード）')
    parser.add_argument('--output', '-o', default='../outputs',
                       help='出力ディレクトリ（デフォルト: ../outputs）')
    parser.add_argument('--no-save', action='store_true',
                       help='画像を保存しない（解析のみ）')
    parser.add_argument('--no-plots', action='store_true',
                       help='グラフを生成しない')
    parser.add_argument('--ground-truth', '-g', default=None,
                       help='正解データのCSVファイルパス（精度評価用）')
    parser.add_argument('--plot', '-p', action='store_true',
                       help='精度評価グラフを生成する（非推奨：自動生成されます）')
    
    args = parser.parse_args()
    
    # デモモード: 引数なしで実行された場合
    if args.input is None:
        print("="*60)
        print("デモモード - データセットの最初の5枚を解析")
        print("="*60)
        
        # データセットのパス
        dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset_generator', 'dataset')
        
        if not os.path.exists(dataset_dir):
            print(f"\nエラー: データセットが見つかりません: {dataset_dir}")
            print("先にデータセットを生成してください:")
            print("  python dataset_generator/generate_dataset.py")
            return
        
        # 最初の5枚を取得
        image_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.png')])[:5]
        
        if not image_files:
            print(f"\nエラー: データセットに画像がありません: {dataset_dir}")
            return
        
        print(f"\n{len(image_files)}枚の画像を解析します...\n")
        
        for i, filename in enumerate(image_files, 1):
            # ファイル名から正解を取得
            parts = filename.replace('.png', '').split('_')
            true_heart = int(parts[2].replace('heart', ''))
            true_top = parts[3].replace('top', '')[0]
            true_bottom = parts[4].replace('bottom', '')[0]
            
            # 解析
            image_path = os.path.join(dataset_dir, filename)
            result = analyze_single_image(image_path)
            
            # 結果表示
            print(f"[{i}] {filename}")
            print(f"  Heart angle: {result['heart_angle']:.1f}deg (true: {true_heart}deg, error: {abs(result['heart_angle'] - true_heart):.1f}deg)")
            print(f"  Top char: {result['top_character']} (true: {true_top}) {'✓' if result['top_character'] == true_top else '✗'}")
            print(f"  Bottom char: {result['bottom_character']} (true: {true_bottom}) {'✓' if result['bottom_character'] == true_bottom else '✗'}")
            print()
        
        print("="*60)
        print("デモ完了！")
        print("\nフルテストを実行する場合:")
        print("  python tests/test_with_output.py")
        print("\n混同行列を確認する場合:")
        print("  python tests/analyze_confusion.py")
        print("="*60)
        return
    
    # 通常モード
    print("="*60)
    print("画像解析システム")
    print("="*60)
    print(f"入力: {args.input}")
    print(f"出力: {args.output}")
    print(f"画像保存: {'なし' if args.no_save else 'あり'}")
    print(f"グラフ生成: {'なし' if args.no_plots else 'あり'}")
    print("="*60)
    print()
    
    # ディレクトリを処理
    results = process_directory(args.input, args.output, 
                               save_results=not args.no_save,
                               generate_plots=not args.no_plots)
    
    # 結果をCSVに保存
    if results and not args.no_save:
        csv_path = os.path.join(args.output, 'analysis_results.csv')
        save_results_csv(results, csv_path)
    
    print(f"\n処理完了！ {len(results)}枚の画像を解析しました。")


if __name__ == '__main__':
    main()
