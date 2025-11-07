"""
データセットをCOCO形式に変換
"""

import os
import json
import csv
import shutil
from pathlib import Path
from PIL import Image
from typing import Dict, List
import argparse
import cv2
import numpy as np

os.chdir(os.path.dirname(__file__))

class COCOConverter:
    """COCO形式への変換クラス"""
    
    def __init__(self, dataset_dir: str, output_dir: str, train_ratio: float = 0.8):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        
        # クラス定義
        self.categories = [
            {"id": 0, "name": "heart", "supercategory": "object"},
            {"id": 1, "name": "A", "supercategory": "letter"},
            {"id": 2, "name": "B", "supercategory": "letter"},
            {"id": 3, "name": "C", "supercategory": "letter"},
            {"id": 4, "name": "D", "supercategory": "letter"},
            {"id": 5, "name": "E", "supercategory": "letter"},
            {"id": 6, "name": "F", "supercategory": "letter"},
        ]
        
        # 文字からクラスIDへのマッピング
        self.letter_to_id = {
            'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6
        }
        
        # Top/Bottomの円は固定位置（実測値）
        self.circle_positions = {
            'top': {'bbox': [549, 50, 202, 202]},
            'bottom': {'bbox': [549, 349, 201, 202]}
        }
    
    def measure_heart_bbox(self, img_path: Path) -> List[int]:
        """
        画像からheartのバウンディングボックスを実測
        
        Args:
            img_path: 画像ファイルのパス
        
        Returns:
            [x, y, width, height] (COCO形式)
        """
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 最も左側にある輪郭がheart（x座標が最小）
        heart_contour = min(contours, key=lambda c: cv2.boundingRect(c)[0])
        x, y, w, h = cv2.boundingRect(heart_contour)
        
        return [x, y, w, h]
    
    def calculate_bbox(self, obj_type: str, img_path: Path = None) -> List[int]:
        """
        オブジェクトタイプからバウンディングボックスを取得
        
        Args:
            obj_type: 'heart', 'top', 'bottom'
            img_path: 画像ファイルのパス（heartの場合に必要）
        
        Returns:
            [x, y, width, height] (COCO形式: 左上座標 + サイズ)
        """
        if obj_type == 'heart':
            if img_path is None:
                raise ValueError("heartのbbox計算には画像パスが必要です")
            return self.measure_heart_bbox(img_path)
        else:
            return self.circle_positions[obj_type]['bbox']
    
    def convert_dataset(self):
        """データセット全体を変換"""
        print("="*60)
        print("COCO形式への変換を開始")
        print("="*60)
        
        # 出力ディレクトリ作成
        (self.output_dir / 'annotations').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        
        # ラベルCSVを読み込み
        labels_file = self.dataset_dir / 'labels.csv'
        if not labels_file.exists():
            raise FileNotFoundError(f"ラベルファイルが見つかりません: {labels_file}")
        
        # データを読み込み
        data = []
        with open(labels_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        
        print(f"総データ数: {len(data)}")
        
        # Train/Valに分割
        split_idx = int(len(data) * self.train_ratio)
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        print(f"Train: {len(train_data)}, Val: {len(val_data)}")
        
        # Train/Val それぞれでCOCO形式に変換
        print("\nTrainセットを変換中...")
        self.create_coco_format(train_data, 'train')
        
        print("\nValセットを変換中...")
        self.create_coco_format(val_data, 'val')
        
        print("\n" + "="*60)
        print("変換完了！")
        print(f"出力先: {self.output_dir}")
        print("="*60)
    
    def create_coco_format(self, data: List[Dict], split: str):
        """
        COCO形式のアノテーションを作成
        
        Args:
            data: データのリスト
            split: 'train' or 'val'
        """
        coco_data = {
            "info": {
                "description": "Pattern Matching Dataset",
                "version": "1.0",
                "year": 2025,
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": self.categories
        }
        
        annotation_id = 0
        
        for image_id, row in enumerate(data):
            filename = row['filename']
            src_path = self.dataset_dir / filename
            dst_path = self.output_dir / 'images' / split / filename
            
            if not src_path.exists():
                print(f"警告: 画像が見つかりません: {src_path}")
                continue
            
            # 画像をコピー
            shutil.copy2(src_path, dst_path)
            
            # 画像情報を取得
            with Image.open(src_path) as img:
                width, height = img.size
            
            # 画像情報を追加
            image_info = {
                "id": image_id,
                "file_name": filename,
                "width": width,
                "height": height
            }
            coco_data["images"].append(image_info)
            
            # アノテーションを追加
            # 1. ハート（画像から実測）
            heart_bbox = self.calculate_bbox('heart', src_path)
            heart_annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 0,  # heart
                "bbox": heart_bbox,
                "area": heart_bbox[2] * heart_bbox[3],
                "iscrowd": 0,
                "attributes": {
                    "angle": int(row['heart_angle'])
                }
            }
            coco_data["annotations"].append(heart_annotation)
            annotation_id += 1
            
            # 2. 右上の文字
            top_letter = row['top_letter']
            top_bbox = self.calculate_bbox('top')
            top_annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": self.letter_to_id[top_letter],
                "bbox": top_bbox,
                "area": top_bbox[2] * top_bbox[3],
                "iscrowd": 0,
                "attributes": {
                    "angle": int(row['top_angle']),
                    "position": "top"
                }
            }
            coco_data["annotations"].append(top_annotation)
            annotation_id += 1
            
            # 3. 右下の文字
            bottom_letter = row['bottom_letter']
            bottom_bbox = self.calculate_bbox('bottom')
            bottom_annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": self.letter_to_id[bottom_letter],
                "bbox": bottom_bbox,
                "area": bottom_bbox[2] * bottom_bbox[3],
                "iscrowd": 0,
                "attributes": {
                    "angle": int(row['bottom_angle']),
                    "position": "bottom"
                }
            }
            coco_data["annotations"].append(bottom_annotation)
            annotation_id += 1
        
        # JSONとして保存
        json_path = self.output_dir / 'annotations' / f'{split}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"  保存: {json_path}")
        print(f"  画像数: {len(coco_data['images'])}")
        print(f"  アノテーション数: {len(coco_data['annotations'])}")


def main():
    parser = argparse.ArgumentParser(description='データセットをCOCO形式に変換')
    parser.add_argument('--dataset', '-d', default='../dataset',
                       help='データセットディレクトリ（デフォルト: ../dataset）')
    parser.add_argument('--output', '-o', default='coco_dataset',
                       help='出力ディレクトリ（デフォルト: coco_dataset）')
    parser.add_argument('--train-ratio', '-r', type=float, default=0.8,
                       help='訓練データの割合（デフォルト: 0.8）')
    
    args = parser.parse_args()
    
    converter = COCOConverter(args.dataset, args.output, args.train_ratio)
    converter.convert_dataset()


if __name__ == '__main__':
    main()
