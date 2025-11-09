"""
D-FINEモデルを使用した推論・解析スクリプト
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import csv
import yaml
from typing import List, Dict, Tuple
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from tqdm import tqdm

os.chdir(os.path.dirname(__file__))

# ユーティリティ関数をインポート
try:
    from utils import calculate_angle_error, generate_accuracy_plots
except ImportError:
    print("警告: utils.pyが見つかりません")


class DFINEInference:
    """D-FINEモデルの推論クラス"""
    
    def __init__(self, checkpoint_path: str = None, config_path: str = 'configs/train_config.yaml'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用デバイス: {self.device}")
        
        # 設定を読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # クラス名
        self.id_to_name = {
            0: 'heart',
            1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'
        }
        
        # モデルとプロセッサーを読み込み
        model_name = self.config.get('model', 'ustc-community/dfine-xlarge-coco')
        if isinstance(model_name, dict):
            model_name = model_name.get('name', 'ustc-community/dfine-xlarge-coco')
        
        # ローカルモデルを優先的に使用（オフライン対応）
        local_model_path = "./pretrained_models/dfine-xlarge-coco"
        if os.path.exists(local_model_path):
            print(f"ローカルモデルを使用: {local_model_path}")
            model_name_or_path = local_model_path
        else:
            print(f"オンラインからモデル読み込み: {model_name}")
            print(f"  （オフライン使用には download_model.py を実行してください）")
            model_name_or_path = model_name
        
        num_classes = 7
        if isinstance(self.config.get('model'), dict):
            num_classes = self.config['model'].get('num_classes', 7)
        
        self.processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"チェックポイント読み込み: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model = AutoModelForObjectDetection.from_pretrained(
                model_name_or_path,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"  エポック: {checkpoint.get('epoch', 'N/A')}")
            print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
        else:
            print("チェックポイントなし（事前学習済みモデルを使用）")
            self.model = AutoModelForObjectDetection.from_pretrained(
                model_name_or_path,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        
        self.model.to(self.device)
        self.model.eval()
        
        # 閾値
        inference_config = self.config.get('inference', {})
        self.confidence_threshold = inference_config.get('confidence_threshold', 0.25)  # 0.5から0.25に変更
        self.iou_threshold = inference_config.get('iou_threshold', 0.5)
    
    def predict(self, image_path: str) -> Dict:
        """
        1枚の画像を推論
        
        Args:
            image_path: 画像パス
        
        Returns:
            検出結果の辞書
        """
        # 画像を読み込み
        image = Image.open(image_path).convert('RGB')
        
        # 前処理
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 推論
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 後処理
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs,
            threshold=self.confidence_threshold,
            target_sizes=target_sizes
        )[0]
        
        # 検出結果を整理
        detections = {
            'heart': None,
            'top': None,
            'bottom': None
        }
        
        for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
            score = score.item()
            label_id = label.item()
            box = box.cpu().numpy()
            
            class_name = self.id_to_name.get(label_id, 'unknown')
            
            # ハート
            if label_id == 0:
                if detections['heart'] is None or score > detections['heart']['score']:
                    detections['heart'] = {
                        'score': score,
                        'box': box,
                        'class': class_name
                    }
            
            # 文字（A-F）
            elif label_id >= 1:
                # Y座標で上下を判定
                y_center = (box[1] + box[3]) / 2
                position = 'top' if y_center < 300 else 'bottom'
                
                if detections[position] is None or score > detections[position]['score']:
                    detections[position] = {
                        'score': score,
                        'box': box,
                        'class': class_name,
                        'label_id': label_id
                    }
        
        return detections
    
    def analyze_image(self, image_path: str) -> Dict:
        """
        画像を解析して結果を返す
        
        Args:
            image_path: 画像パス
        
        Returns:
            解析結果
        """
        detections = self.predict(image_path)
        
        result = {
            'heart_angle': None,
            'top_character': None,
            'bottom_character': None,
            'heart_score': None,
            'top_score': None,
            'bottom_score': None
        }
        
        # ハート（角度は検出できないため、固定値またはN/A）
        if detections['heart']:
            result['heart_angle'] = 0.0  # 角度情報は別途必要
            result['heart_score'] = detections['heart']['score']
        
        # 右上文字
        if detections['top']:
            result['top_character'] = detections['top']['class']
            result['top_score'] = detections['top']['score']
        
        # 右下文字
        if detections['bottom']:
            result['bottom_character'] = detections['bottom']['class']
            result['bottom_score'] = detections['bottom']['score']
        
        return result
    
    def draw_results(self, image: Image.Image, detections: Dict, ground_truth: Dict = None) -> Image.Image:
        """
        検出結果を画像に描画
        
        Args:
            image: 元画像
            detections: 検出結果
            ground_truth: 正解データ（オプション）
        
        Returns:
            描画済み画像
        """
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 24)
            font_small = ImageFont.truetype("arial.ttf", 18)
        except:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # 検出結果を描画
        for position, det in detections.items():
            if det is None:
                continue
            
            box = det['box']
            score = det['score']
            class_name = det['class']
            
            # バウンディングボックスの座標を検証・修正
            x1, y1, x2, y2 = box.tolist() if hasattr(box, 'tolist') else box
            
            # 座標が逆転している場合は修正
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            
            # 座標が画像外の場合はクリップ
            img_width, img_height = image.size
            x1 = max(0, min(x1, img_width))
            x2 = max(0, min(x2, img_width))
            y1 = max(0, min(y1, img_height))
            y2 = max(0, min(y2, img_height))
            
            # バウンディングボックス
            try:
                draw.rectangle([x1, y1, x2, y2], outline='green', width=3)
                
                # ラベル
                label_text = f"{class_name} ({score:.2f})"
                label_y = max(0, y1 - 25)  # ラベルが画像外に出ないように
                draw.text((x1, label_y), label_text, fill='green', font=font_small)
            except Exception as e:
                # 描画エラーが発生した場合はスキップ
                print(f"  描画エラー ({class_name}): {e}")
        
        # 結果テキストを追加
        y_offset = 10
        if detections['heart']:
            text = f"Heart: Detected ({detections['heart']['score']:.2f})"
            draw.text((10, y_offset), text, fill='blue', font=font)
            y_offset += 30
        
        if detections['top']:
            text = f"Top: {detections['top']['class']} ({detections['top']['score']:.2f})"
            if ground_truth and 'top_char' in ground_truth:
                gt_text = ground_truth['top_char']
                is_correct = detections['top']['class'] == gt_text
                text += f" [GT: {gt_text}] {'✓' if is_correct else '✗'}"
            draw.text((10, y_offset), text, fill='blue', font=font)
            y_offset += 30
        
        if detections['bottom']:
            text = f"Bottom: {detections['bottom']['class']} ({detections['bottom']['score']:.2f})"
            if ground_truth and 'bottom_char' in ground_truth:
                gt_text = ground_truth['bottom_char']
                is_correct = detections['bottom']['class'] == gt_text
                text += f" [GT: {gt_text}] {'✓' if is_correct else '✗'}"
            draw.text((10, y_offset), text, fill='blue', font=font)
        
        return image


def extract_ground_truth_from_filename(filename: str) -> Dict:
    """ファイル名から正解データを抽出"""
    try:
        name_without_ext = filename.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
        parts = name_without_ext.split('_')
        
        if len(parts) >= 5 and 'heart' in parts[2] and 'top' in parts[3] and 'bottom' in parts[4]:
            heart_angle = int(parts[2].replace('heart', ''))
            top_part = parts[3].replace('top', '')
            top_char = top_part[0]
            bottom_part = parts[4].replace('bottom', '')
            bottom_char = bottom_part[0]
            
            return {
                'heart_angle': heart_angle,
                'top_char': top_char,
                'bottom_char': bottom_char
            }
    except:
        pass
    
    return None


def process_directory(inference: DFINEInference, input_dir: str, output_dir: str,
                     save_results: bool = True, generate_plots: bool = True) -> List[Dict]:
    """
    ディレクトリ内の全画像を処理
    
    Args:
        inference: DFINEInferenceインスタンス
        input_dir: 入力ディレクトリ
        output_dir: 出力ディレクトリ
        save_results: 結果を保存するか
        generate_plots: グラフを生成するか
    
    Returns:
        解析結果のリスト
    """
    # 出力ディレクトリを作成
    if save_results:
        images_dir = Path(output_dir) / 'images'
        plots_dir = Path(output_dir) / 'plots'
        images_dir.mkdir(parents=True, exist_ok=True)
        if generate_plots:
            plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 画像ファイルを取得
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
        image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    image_files = sorted(list(set(image_files)))
    
    if not image_files:
        print(f"警告: {input_dir} に画像ファイルが見つかりませんでした。")
        return []
    
    print(f"{len(image_files)}枚の画像を処理します...\n")
    
    results = []
    
    for image_path in tqdm(image_files, desc="推論中"):
        try:
            # 推論
            detections = inference.predict(str(image_path))
            
            # 解析結果を整理
            result = {
                'filename': image_path.name,
                'heart_angle': 0.0 if detections['heart'] else None,
                'top_character': detections['top']['class'] if detections['top'] else None,
                'bottom_character': detections['bottom']['class'] if detections['bottom'] else None,
                'heart_score': detections['heart']['score'] if detections['heart'] else 0.0,
                'top_score': detections['top']['score'] if detections['top'] else 0.0,
                'bottom_score': detections['bottom']['score'] if detections['bottom'] else 0.0
            }
            
            # 正解データを抽出
            ground_truth = extract_ground_truth_from_filename(image_path.name)
            if ground_truth:
                result['gt_heart'] = ground_truth['heart_angle']
                result['gt_top'] = ground_truth['top_char']
                result['gt_bottom'] = ground_truth['bottom_char']
            
            results.append(result)
            
            # 画像を保存
            if save_results:
                image = Image.open(image_path).convert('RGB')
                result_image = inference.draw_results(image, detections, ground_truth)
                output_path = images_dir / image_path.name
                result_image.save(output_path)
        
        except Exception as e:
            print(f"エラー ({image_path.name}): {e}")
            continue
    
    # グラフ生成
    if save_results and generate_plots and results:
        try:
            print("\nグラフを生成中...")
            generate_accuracy_plots(results, str(plots_dir))
        except Exception as e:
            print(f"グラフ生成エラー: {e}")
    
    return results


def save_results_csv(results: List[Dict], output_path: str):
    """結果をCSVに保存"""
    if not results:
        print("保存する結果がありません。")
        return
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n解析結果を保存しました: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='D-FINE推論システム')
    parser.add_argument('--input', '-i', default='../dataset',
                       help='入力ディレクトリ（デフォルト: ../dataset）')
    parser.add_argument('--output', '-o', default='outputs',
                       help='出力ディレクトリ（デフォルト: outputs）')
    parser.add_argument('--checkpoint', '-c', default='checkpoints/best_model.pth',
                       help='モデルチェックポイント（デフォルト: checkpoints/best_model.pth）')
    parser.add_argument('--config', default='configs/train_config.yaml',
                       help='設定ファイル')
    parser.add_argument('--no-save', action='store_true',
                       help='画像を保存しない')
    parser.add_argument('--no-plots', action='store_true',
                       help='グラフを生成しない')
    parser.add_argument('--ground-truth', '-g', default=None,
                       help='正解データのCSVファイル（未使用）')
    parser.add_argument('--plot', '-p', action='store_true',
                       help='グラフを生成（非推奨：自動生成）')
    
    args = parser.parse_args()
    
    print("="*60)
    print("D-FINE画像解析システム")
    print("="*60)
    print(f"入力: {args.input}")
    print(f"出力: {args.output}")
    print(f"チェックポイント: {args.checkpoint}")
    print("="*60)
    print()
    
    # 推論システムを初期化
    inference = DFINEInference(
        checkpoint_path=args.checkpoint if os.path.exists(args.checkpoint) else None,
        config_path=args.config
    )
    
    # ディレクトリを処理
    results = process_directory(
        inference,
        args.input,
        args.output,
        save_results=not args.no_save,
        generate_plots=not args.no_plots
    )
    
    # 結果をCSVに保存
    if results and not args.no_save:
        csv_path = Path(args.output) / 'analysis_results.csv'
        save_results_csv(results, str(csv_path))
    
    # 精度統計を表示
    if results:
        total = len(results)
        top_correct = sum(1 for r in results if r.get('gt_top') and r['top_character'] == r['gt_top'])
        bottom_correct = sum(1 for r in results if r.get('gt_bottom') and r['bottom_character'] == r['gt_bottom'])
        
        print("\n" + "="*60)
        print("精度統計")
        print("="*60)
        print(f"総画像数: {total}")
        if any('gt_top' in r for r in results):
            print(f"右上文字正解率: {top_correct/total*100:.1f}% ({top_correct}/{total})")
            print(f"右下文字正解率: {bottom_correct/total*100:.1f}% ({bottom_correct}/{total})")
        print("="*60)
    
    print(f"\n処理完了！ {len(results)}枚の画像を解析しました。")


if __name__ == '__main__':
    main()
