"""
D-FINEモデルを事前ダウンロードしてローカルに保存
オフライン環境での使用を可能にする
"""

import os
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from pathlib import Path

os.chdir(os.path.dirname(__file__))

MODEL_NAME = "ustc-community/dfine-xlarge-coco"
LOCAL_MODEL_DIR = "./pretrained_models/dfine-xlarge-coco"


def download_and_save_model():
    """モデルをダウンロードしてローカルに保存"""
    
    print("="*70)
    print("D-FINEモデルのダウンロード")
    print("="*70)
    print(f"モデル: {MODEL_NAME}")
    print(f"保存先: {LOCAL_MODEL_DIR}")
    print()
    
    # ディレクトリを作成
    Path(LOCAL_MODEL_DIR).mkdir(parents=True, exist_ok=True)
    
    # プロセッサーをダウンロード
    print("1. Image Processorをダウンロード中...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    processor.save_pretrained(LOCAL_MODEL_DIR)
    print("   ✓ 完了")
    
    # モデルをダウンロード
    print("\n2. モデルをダウンロード中（約1GB、数分かかります）...")
    model = AutoModelForObjectDetection.from_pretrained(
        MODEL_NAME,
        ignore_mismatched_sizes=True
    )
    model.save_pretrained(LOCAL_MODEL_DIR)
    print("   ✓ 完了")
    
    print()
    print("="*70)
    print("ダウンロード完了！")
    print("="*70)
    print(f"保存場所: {Path(LOCAL_MODEL_DIR).absolute()}")
    print()
    print("以降はオフラインでも使用できます。")
    print("train.py と inference.py は自動的にローカルモデルを使用します。")
    print("="*70)


if __name__ == "__main__":
    download_and_save_model()
