# Dataset Generator

データセット生成スクリプト - 画像解析システムの学習・評価用データを生成します。

## 概要

このスクリプトは、以下の要素を含む画像を自動生成します：
- **左下**: 回転したハートマーク（♡）
- **右上**: 回転したアルファベット（A～F）を含む黒い円
- **右下**: 回転したアルファベット（A～F）を含む黒い円

## ファイル構成

```
dataset_generator/
├── generate_dataset.py              # クリーンデータ生成（1000枚）
├── generate_test_dataset.py         # テストデータ生成（500枚）
├── generate_dataset_with_noise.py   # ノイズ付きデータ生成（1000枚）
├── noise_config.yaml                # ノイズ設定ファイル
├── NOISE_GUIDE.md                   # ノイズ機能の詳細ガイド
├── README.md                        # このファイル
└── requirements.txt                 # 依存パッケージ
```

## インストール

必要なパッケージをインストールします：

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. クリーンデータ生成（学習用）

```bash
cd dataset_generator
python generate_dataset.py
```

**出力**: `dataset/` に1000枚の画像

### 2. テストデータ生成（評価用）

```bash
python generate_test_dataset.py
```

**出力**: `test_dataset/` に500枚の画像（異なるシード値）

### 3. ノイズ付きデータ生成（ロバスト性テスト用）

```bash
python generate_dataset_with_noise.py
```

**出力**: `dataset_noisy/` に1000枚のノイズ付き画像

詳細は [NOISE_GUIDE.md](NOISE_GUIDE.md) を参照してください。

### 設定のカスタマイズ

`config.py`を編集して、以下のパラメータを変更できます：

#### 画像サイズ
```python
IMAGE_WIDTH = 800      # 画像の幅（ピクセル）
IMAGE_HEIGHT = 600     # 画像の高さ（ピクセル）
```

#### ハートマークの設定
```python
HEART_SIZE = 80                    # ハートのサイズ
HEART_POSITION_OFFSET_X = 100      # 左端からのオフセット
HEART_POSITION_OFFSET_Y = 100      # 下端からのオフセット
```

#### アルファベット円の設定
```python
CIRCLE_RADIUS = 60                 # 円の半径
ALPHABET_FONT_SIZE = 50            # フォントサイズ
ALPHABET_CHOICES = ['A', 'B', 'C', 'D', 'E', 'F']  # 使用する文字
```

#### データセット生成の設定
```python
NUM_IMAGES = 100                   # 生成する画像枚数
OUTPUT_DIR = '../dataset'          # 出力ディレクトリ（プロジェクトルート）
RANDOM_SEED = 42                   # ランダムシード（再現性）
```

## 出力形式

### 生成されるファイル

実行すると、プロジェクトルートの`dataset/`ディレクトリに以下のファイルが生成されます：

1. **画像ファイル**: `image_XXXX_heart{角度}_top{文字}{角度}_bottom{文字}{角度}.png`
   - 例: `image_0000_heart45_topA30_bottomB270.png`

2. **ラベルファイル**: `labels.csv`

### labels.csvの形式

| フィールド名 | 説明 | 例 |
|-------------|------|-----|
| filename | 画像ファイル名 | image_0000_heart45_topA30_bottomB270.png |
| heart_angle | ハートの角度（0-359度） | 45 |
| top_letter | 右上の文字 | A |
| top_angle | 右上の角度（0-359度） | 30 |
| bottom_letter | 右下の文字 | B |
| bottom_angle | 右下の角度（0-359度） | 270 |

## 角度の定義

- **0度**: 元の向き（上向き）
- **正の方向**: 時計回り
- **範囲**: 0～359度

## 座標系

- **原点**: 画像の左上
- **X軸**: 右方向が正
- **Y軸**: 下方向が正

## サンプル画像の特徴

生成される各画像には以下の特徴があります：

1. **白背景**: すべての画像は白い背景
2. **黒いマーク**: ハートとアルファベット円は黒色
3. **白い文字**: 円の中の文字は白色
4. **ランダムな角度**: すべての要素がランダムな角度で回転

## データセット比較

| 種類 | スクリプト | 出力先 | 枚数 | シード | 用途 |
|------|-----------|--------|------|--------|------|
| 学習用 | `generate_dataset.py` | `dataset/` | 1000 | 42 | モデル学習 |
| テスト用 | `generate_test_dataset.py` | `test_dataset/` | 500 | 12345 | 精度評価 |
| ノイズ付き | `generate_dataset_with_noise.py` | `dataset_noisy/` | 1000 | 42 | ロバスト性テスト |

## ノイズの種類（generate_dataset_with_noise.py）

1. **ガウシアンノイズ**: ランダムなピクセル値の変動
2. **塩コショウノイズ**: ランダムな白黒ドット
3. **ぼかし**: 画像全体のぼかし
4. **明るさ調整**: 明るさのランダム変更
5. **回転のズレ**: ハート・文字角度の微小なズレ
6. **位置のズレ**: 要素位置の微小なズレ
7. **背景ノイズ**: 背景にランダムなスポット

詳細は [NOISE_GUIDE.md](NOISE_GUIDE.md) を参照してください。

## トラブルシューティング

### フォントエラーが発生する場合

Windowsの場合、`arial.ttf`が見つからない場合はデフォルトフォントが使用されます。
より良い結果を得るには、適切なTrueTypeフォントをシステムにインストールしてください。

### メモリエラーが発生する場合

`NUM_IMAGES`の値を減らして、少ない枚数から試してください。

### ノイズ付きデータでnumpyエラーが出る場合

```bash
pip install numpy
```

## カスタマイズ例

### より多くの画像を生成する

```python
# config.py
NUM_IMAGES = 1000  # 1000枚生成
```

### 大きな画像を生成する

```python
# config.py
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
HEART_SIZE = 150
CIRCLE_RADIUS = 100
ALPHABET_FONT_SIZE = 80
```

### アルファベットを追加する

```python
# config.py
ALPHABET_CHOICES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
```

## ライセンス

このスクリプトは教育・研究目的で自由に使用できます。
