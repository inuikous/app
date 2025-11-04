# Image Analyzer

画像解析システム - 画像内のハートマークとアルファベット文字を検出・解析します。

## 概要

このシステムは、画像を解析して以下の情報を抽出します：
- **左下のハートマーク**: 回転角度を検出
- **右上のアルファベット**: 文字（A～F）と回転角度を検出
- **右下のアルファベット**: 文字（A～F）と回転角度を検出

## インストール

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. Tesseract OCRのインストール（オプション）

文字認識の精度を向上させるには、Tesseract OCRをインストールしてください。

#### Windows
1. [Tesseract OCR for Windows](https://github.com/UB-Mannheim/tesseract/wiki)からインストーラーをダウンロード
2. インストール後、環境変数`PATH`にTesseractのパスを追加
3. または、Pythonコード内で以下を設定：
   ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

#### Linux
```bash
sudo apt-get install tesseract-ocr
```

#### macOS
```bash
brew install tesseract
```

## 使用方法

### 基本的な使い方

```bash
cd analyzer
python analyze.py --input ../dataset_generator/dataset
```

### コマンドラインオプション

```bash
python analyze.py [オプション]

オプション:
  --input, -i DIR          入力ディレクトリ（デフォルト: デモモード）
  --output, -o DIR         出力ディレクトリ（デフォルト: outputs）
  --no-save                画像を保存しない（解析結果のみ表示）
  --ground-truth, -g FILE  正解データのCSVファイル（精度評価用）
  --plot, -p               精度評価グラフを生成（--ground-truth必須）
```

### 使用例

#### 1. 基本的な解析

```bash
python analyze.py --input ../dataset
```

#### 2. 精度評価（正解データと比較）+ グラフ生成

```bash
python analyze.py \
  --input ../dataset \
  --ground-truth ../dataset/labels.csv \
  --plot
```

グラフは`outputs/plots/`に保存されます：
- `accuracy_summary.png` - 精度サマリー（4種類のグラフ）
- `error_transition.png` - 画像ごとの誤差推移
- `error_boxplot.png` - 誤差の統計分布

#### 3. グラフのみ生成

```bash
python visualize_accuracy.py \
  --results outputs/analysis_results.csv \
  --ground-truth ../dataset/labels.csv \
  --output outputs/plots
```

#### 4. 解析のみ（画像保存なし）

```bash
python analyze.py --input ../dataset --no-save
```

## ディレクトリ構造

```
analyzer/
├── analyze.py           # メインスクリプト
├── image_utils.py       # 画像処理ユーティリティ
├── config.py            # 設定ファイル
├── requirements.txt     # 依存パッケージ
└── README.md           # このファイル
```

## 出力形式

### 1. 解析済み画像

元の画像に解析結果が赤いテキストで描画されます：
- 左下: `Heart: XX.X°`
- 右上: `Top: A (XX.X°)`
- 右下: `Bottom: B (XX.X°)`

### 2. CSV結果ファイル

`outputs/analysis_results.csv`に以下の形式で保存されます：

| フィールド | 説明 |
|-----------|------|
| filename | ファイル名 |
| heart_angle | ハートの角度 |
| top_char | 右上の文字 |
| top_angle | 右上の角度 |
| bottom_char | 右下の文字 |
| bottom_angle | 右下の角度 |

### 3. 精度評価結果（正解データがある場合）

コンソールに以下の情報が表示されます：
- ハート角度の平均誤差、最大誤差、標準偏差
- 右上・右下の文字認識正解率
- 右上・右下の角度検出の平均誤差

### 4. グラフ（--plotオプション使用時）

`outputs/plots/`に以下のグラフが生成されます：

#### accuracy_summary.png
4つのグラフを含むサマリー：
- ハート角度誤差の分布（ヒストグラム）
- 右上アルファベット角度誤差の分布（ヒストグラム）
- 右下アルファベット角度誤差の分布（ヒストグラム）
- 文字認識正解率（棒グラフ）

#### error_transition.png
画像ごとの誤差推移：
- ハート角度誤差の推移
- 右上アルファベット角度誤差の推移
- 右下アルファベット角度誤差の推移

#### error_boxplot.png
誤差の統計分布（ボックスプロット）：
- 中央値、四分位範囲、外れ値を視覚化

## アルゴリズム

### ハート角度検出

1. グレースケール変換と二値化
2. 輪郭検出
3. 最大輪郭の抽出
4. モーメント計算による重心検出
5. 楕円フィッティング
6. 最下点から角度を計算

### 文字認識

1. 円領域の検出
2. ROI（関心領域）の抽出
3. 画像の前処理（拡大、ノイズ除去、二値化）
4. Tesseract OCRによる文字認識
5. フォールバック: テンプレートマッチング

### 角度検出

1. 文字輪郭の検出
2. 最小外接矩形の計算
3. 矩形の傾きから角度を算出

## 設定のカスタマイズ

`config.py`で解析パラメータを変更できます：

```python
# 解析領域の定義
HEART_REGION_LEFT = 0
HEART_REGION_TOP = 300
HEART_REGION_RIGHT = 400
HEART_REGION_BOTTOM = 600

# 角度検出の精度
ANGLE_DETECTION_STEP = 1  # 度

# 結果描画の設定
RESULT_TEXT_COLOR = (255, 0, 0)  # RGB
RESULT_TEXT_SIZE = 30
```

## トラブルシューティング

### OCRが動作しない

Tesseractがインストールされていない、またはパスが通っていない可能性があります。
- Tesseractをインストール
- パスを確認
- テンプレートマッチングにフォールバック

### 角度検出の精度が低い

- 画像のコントラストを調整
- `config.py`の領域設定を調整
- より多くのデータセットで検証

### メモリエラー

- 大量の画像を処理する場合は、バッチサイズを小さくする
- 画像サイズを小さくする

## パフォーマンス

- 1画像あたりの処理時間: 約1-3秒（PCスペックに依存）
- バッチ処理対応

## 今後の改善予定

- [ ] ディープラーニングによる文字認識の精度向上
- [ ] GPU対応による高速化
- [ ] リアルタイム処理対応
- [ ] より多様な角度での精度向上

## ライセンス

教育・研究目的で自由に使用できます。
