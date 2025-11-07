"""
D-FINEモデルの学習スクリプト
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
from typing import Dict, List
import argparse

os.chdir(os.path.dirname(__file__))


class COCODetectionDataset(Dataset):
    """COCO形式のデータセット"""
    
    def __init__(self, coco_json: str, image_dir: str, processor):
        self.image_dir = Path(image_dir)
        self.processor = processor
        
        # COCOアノテーションを読み込み
        with open(coco_json, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        
        # 画像IDごとにアノテーションをグループ化
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        img_path = self.image_dir / img_info['file_name']
        
        # 画像を読み込み
        image = Image.open(img_path).convert('RGB')
        
        # アノテーションを取得
        anns = self.img_to_anns.get(img_id, [])
        
        # アノテーションをCOCO形式に準備
        # processorが期待する形式: {'image_id': int, 'annotations': [{'bbox': [x,y,w,h], 'category_id': int, 'area': float, 'iscrowd': 0}, ...]}
        annotations_list = []
        for ann in anns:
            annotations_list.append({
                'bbox': ann['bbox'],  # [x, y, w, h]
                'category_id': ann['category_id'],
                'area': ann.get('area', ann['bbox'][2] * ann['bbox'][3]),
                'iscrowd': 0
            })
        
        target = {
            'image_id': img_id,
            'annotations': annotations_list
        }

        # プロセッサで画像のみを処理
        encoding = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding['pixel_values'].squeeze(0)

        # モデルが期待する形式でlabelsを作成
        # bboxを[x,y,w,h]から[cx,cy,w,h]正規化形式に変換
        img_h, img_w = image.size[1], image.size[0]  # Pillowは(w,h)
        boxes = []
        class_labels = []
        for ann in annotations_list:
            x, y, w, h = ann['bbox']
            # 正規化された中心座標形式に変換
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            boxes.append([cx, cy, nw, nh])
            class_labels.append(ann['category_id'])
        
        labels_out = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'class_labels': torch.tensor(class_labels, dtype=torch.int64)
        }

        return {'pixel_values': pixel_values, 'labels': labels_out}


def collate_fn(batch):
    """カスタムcollate関数"""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = [item['labels'] for item in batch]
    return {'pixel_values': pixel_values, 'labels': labels}


class DFINETrainer:
    """D-FINEモデルの学習クラス"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"使用デバイス: {self.device}")
        
        # モデルとプロセッサーを初期化
        model_name = config['model']['name']
        print(f"モデル読み込み中: {model_name}")
        
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        print("モデルをロード中...")
        num_classes = config['model']['num_classes']
        
        self.model = AutoModelForObjectDetection.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        print(f"  クラス数: {num_classes}")
        print(f"  モデル設定完了")
        
        self.model.to(self.device)
        
        # バックボーンの凍結設定
        if config['training'].get('freeze_backbone', False):
            self.freeze_backbone()
            print("  ✓ バックボーンを凍結（出力層のみ学習）")
        else:
            print("  ✓ 全パラメータを学習")
        
        # データセットとデータローダーを準備
        self.setup_data()
        
        # オプティマイザーとスケジューラー
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # 出力ディレクトリ
        self.checkpoint_dir = Path(config['training']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_data(self):
        """データセットとデータローダーを準備"""
        config = self.config
        
        train_dataset = COCODetectionDataset(
            coco_json=config['data']['train_annotations'],
            image_dir=config['data']['train_images'],
            processor=self.processor
        )
        
        val_dataset = COCODetectionDataset(
            coco_json=config['data']['val_annotations'],
            image_dir=config['data']['val_images'],
            processor=self.processor
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
    
    def freeze_backbone(self):
        """バックボーンを凍結して出力層のみ学習可能にする"""
        # まず全パラメータを凍結
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 出力層（検出ヘッド）のみ学習可能にする
        # D-FINEモデルの構造に応じて調整が必要
        trainable_params = 0
        frozen_params = 0
        
        for name, param in self.model.named_parameters():
            # 'class_embed', 'bbox_embed', 'head' などの名前を持つレイヤーを学習可能にする
            if any(keyword in name.lower() for keyword in ['class_embed', 'bbox_embed', 'head', 'output']):
                param.requires_grad = True
                trainable_params += param.numel()
                print(f"    学習可能: {name} ({param.numel():,} params)")
            else:
                frozen_params += param.numel()
        
        total_params = trainable_params + frozen_params
        print(f"\n  総パラメータ数: {total_params:,}")
        print(f"  学習可能: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"  凍結: {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
    
    def train_epoch(self, epoch: int):
        """1エポック学習"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            pixel_values = batch['pixel_values'].to(self.device)
            labels = batch['labels']
            
            for target in labels:
                for k, v in target.items():
                    if isinstance(v, torch.Tensor):
                        target[k] = v.to(self.device)
            
            outputs = self.model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        """検証"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels']
                
                for target in labels:
                    for k, v in target.items():
                        if isinstance(v, torch.Tensor):
                            target[k] = v.to(self.device)
                
                outputs = self.model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def train(self):
        """学習ループ"""
        num_epochs = self.config['training']['num_epochs']
        best_val_loss = float('inf')
        
        print("\n" + "="*60)
        print("学習開始")
        print("="*60)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # 学習
            train_loss = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.4f}")
            
            # 検証
            val_loss = self.validate()
            print(f"Val Loss: {val_loss:.4f}")
            
            # ベストモデルを保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = self.checkpoint_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"✓ ベストモデル保存: {checkpoint_path}")
            
            # 定期的にチェックポイントを保存
            if (epoch + 1) % self.config['training']['save_every'] == 0:
                checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"  チェックポイント保存: {checkpoint_path}")
        
        print("\n" + "="*60)
        print("学習完了！")
        print(f"Best Val Loss: {best_val_loss:.4f}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='D-FINEモデルの学習')
    parser.add_argument('--config', '-c', default='configs/train_config.yaml',
                       help='設定ファイル（デフォルト: configs/train_config.yaml）')
    
    args = parser.parse_args()
    
    # 設定を読み込み
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 学習開始
    trainer = DFINETrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
