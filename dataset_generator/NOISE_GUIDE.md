# ノイズ付きデータセット生成ガイド

## 概要

`generate_dataset_with_noise.py` は、実世界のデータをシミュレートするために、様々なノイズを含むデータセットを生成します。

## ノイズの種類

### 1. ガウシアンノイズ
**効果**: 画像全体にランダムなピクセル値の変動を追加  
**用途**: センサーノイズ、画質劣化をシミュレート  
**設定**: `std_dev` (標準偏差、0-255)

### 2. 塩コショウノイズ
**効果**: ランダムな白黒ドットを追加  
**用途**: デジタルノイズ、伝送エラーをシミュレート  
**設定**: `amount` (ノイズの量、0.0-1.0)

### 3. ぼかし
**効果**: 画像全体をぼかす  
**用途**: ピンボケ、手ブレをシミュレート  
**設定**: `radius` (ぼかし半径、ピクセル単位)

### 4. 明るさ調整
**効果**: 画像の明るさをランダムに変更  
**用途**: 照明条件の変動をシミュレート  
**設定**: `range` (明るさの倍率範囲、例: [0.8, 1.2])

### 5. 回転のズレ
**効果**: ハートや文字の角度に微小なズレを追加  
**用途**: 配置の不正確さをシミュレート  
**設定**: `max_angle` (最大ズレ角度、度単位)

### 6. 位置のズレ
**効果**: ハートや円の位置に微小なズレを追加  
**用途**: 配置の不正確さをシミュレート  
**設定**: `max_offset` (最大ズレ距離、ピクセル単位)

### 7. 背景ノイズ
**効果**: 背景にランダムなスポットを追加  
**用途**: 汚れ、背景パターンをシミュレート  
**設定**: `num_spots` (スポット数範囲), `spot_size` (サイズ範囲)

### 8. コントラスト調整 ⭐NEW
**効果**: 明暗の差を強調または弱める  
**用途**: カメラのコントラスト設定、照明条件の変化をシミュレート  
**設定**: `range` (倍率範囲、1.0が元のコントラスト、<1で弱く、>1で強く)

### 9. 彩度調整 ⭐NEW
**効果**: 色の鮮やかさを変更  
**用途**: カメラの彩度設定、色褪せをシミュレート  
**設定**: `range` (倍率範囲、1.0が元の彩度、<1でグレースケール寄り)

### 10. ガンマ補正 ⭐NEW
**効果**: 中間調の明るさを調整（明るさ調整とは異なる非線形変換）  
**用途**: ディスプレイの特性、露出の違いをシミュレート  
**設定**: `range` (ガンマ値、1.0が補正なし、<1で明るく、>1で暗く)

### 11. 影効果 ⭐NEW
**効果**: 画像の一部に影を追加（corner/edge/spot の3タイプ）  
**用途**: 不均一な照明、部分的な遮蔽をシミュレート  
**設定**: `intensity` (影の濃さ、0.0=影なし、1.0=完全に暗い)

### 12. ビネット効果 ⭐NEW
**効果**: 写真の周辺を暗くする古典的な写真効果  
**用途**: レンズの周辺減光、古い写真風の効果をシミュレート  
**設定**: `intensity` (効果の強さ、0.0=なし、1.0=周辺真っ暗)

## 使い方

### 基本的な実行

```bash
cd dataset_generator
python generate_dataset_with_noise.py
```

### 出力

- **画像**: `dataset_noisy/image_XXXX_*.png`
- **ラベル**: `dataset_noisy/labels.csv`
- **統計**: `dataset_noisy/noise_statistics.txt`

### labels.csv の形式

```csv
filename,heart_angle,top_letter,top_angle,bottom_letter,bottom_angle,noise_types
image_0000_heart213_topF5_bottomC188.png,213,F,5,C,188,gaussian;blur
image_0001_heart45_topA120_bottomD270.png,45,A,120,D,270,position_jitter
image_0002_heart180_topB90_bottomE45.png,180,B,90,E,45,none
```

**noise_types列**: 適用されたノイズの種類（セミコロン区切り）

### noise_statistics.txt の例

```
ノイズ統計
==================================================

総画像数: 1000

gaussian: 298 (29.8%)
brightness_1.05: 152 (15.2%)
brightness_0.92: 148 (14.8%)
blur: 243 (24.3%)
position_jitter: 256 (25.6%)
salt_pepper: 189 (18.9%)
rotation_jitter: 201 (20.1%)
background_spots: 147 (14.7%)
```

## ノイズ設定のカスタマイズ

### 方法1: スクリプト内で直接編集

`generate_dataset_with_noise.py` の `NOISE_CONFIG` を編集：

```python
NOISE_CONFIG = {
    'gaussian_noise': {
        'enabled': True,
        'probability': 0.5,  # 50%に変更
        'std_dev': 20        # より強いノイズ
    },
    # ...
}
```

### 方法2: 設定ファイルを参照

`noise_config.yaml` を参考にして設定を調整（将来的にYAML読み込み機能を追加予定）

## ノイズレベルの推奨設定

### 軽度ノイズ（現実的なシナリオ）
```python
'gaussian_noise': {'probability': 0.2, 'std_dev': 5}
'blur': {'probability': 0.15, 'radius': 1.0}
'brightness': {'probability': 0.2, 'range': (0.9, 1.1)}
```

### 中度ノイズ（デフォルト、ロバスト性テスト用）
```python
'gaussian_noise': {'probability': 0.3, 'std_dev': 10}
'blur': {'probability': 0.25, 'radius': 1.5}
'brightness': {'probability': 0.3, 'range': (0.8, 1.2)}
```

### 強度ノイズ（極限テスト用）
```python
'gaussian_noise': {'probability': 0.5, 'std_dev': 20}
'blur': {'probability': 0.4, 'radius': 2.5}
'brightness': {'probability': 0.4, 'range': (0.6, 1.4)}
```

## テスト戦略

### 1. クリーンデータで学習

```bash
# クリーンデータ生成
python generate_dataset.py

# 学習
cd ../D-FINE_analyzer
python train.py
```

### 2. ノイズデータでテスト

```bash
# ノイズデータ生成
cd ../dataset_generator
python generate_dataset_with_noise.py

# テスト
cd ../D-FINE_analyzer
python inference.py --input ../dataset_noisy --ground-truth ../dataset_noisy/labels.csv
```

### 3. ノイズデータで学習（推奨）

```bash
# ノイズデータ生成
python generate_dataset_with_noise.py

# dataset_noisy を dataset にコピー
# または convert_to_coco.py を修正してdataset_noisyを読み込む

# 学習
cd ../D-FINE_analyzer
python train.py
```

## トラブルシューティング

### ノイズが強すぎて認識できない

**解決策**: `probability` を下げる、または `std_dev`/`radius` を小さくする

```python
'gaussian_noise': {'probability': 0.1, 'std_dev': 5}
```

### ノイズが弱すぎる

**解決策**: `probability` を上げる、または `std_dev`/`radius` を大きくする

```python
'gaussian_noise': {'probability': 0.5, 'std_dev': 15}
```

### 特定のノイズだけ無効にしたい

**解決策**: `enabled` を `False` に設定

```python
'blur': {'enabled': False, ...}
```

### メモリエラー

**解決策**: `NUM_IMAGES` を減らす

```python
NUM_IMAGES = 500  # 1000から500に変更
```

## 精度への影響予測

| ノイズの種類 | 予想される影響 | 対策 |
|------------|--------------|------|
| ガウシアンノイズ | エッジ検出の劣化 | 前処理でノイズ除去 |
| 塩コショウノイズ | 誤検出の増加 | メディアンフィルタ適用 |
| ぼかし | 細部の喪失 | シャープネス強調 |
| 明るさ調整 | コントラスト低下 | ヒストグラム正規化 |
| 回転/位置ズレ | 境界ボックスのズレ | データ拡張で学習 |
| 背景ノイズ | 誤検出の増加 | 背景差分、ROI設定 |
| コントラスト調整 | 文字認識の困難 | コントラスト正規化 |
| 彩度調整 | 色情報の喪失 | グレースケール変換 |
| ガンマ補正 | 中間調の変化 | ヒストグラム等化 |
| 影効果 | 部分的な検出失敗 | 照明正規化 |
| ビネット効果 | 周辺部の検出劣化 | 中心部重視の学習 |

## 次のステップ

1. **ノイズデータで学習**: モデルのロバスト性を向上
2. **精度比較**: クリーンデータ vs ノイズデータ
3. **ノイズ除去**: 前処理パイプラインの追加
4. **データ拡張**: 学習時のオーグメンテーション

## 参考

- クリーンデータ生成: `generate_dataset.py`
- テストデータ生成: `generate_test_dataset.py`
- ノイズデータ生成: `generate_dataset_with_noise.py` ← このガイド
