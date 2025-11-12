# Analyzer Noisy

ノイズ付きデータセット（dataset_noisy）の解析プログラム

## 概要

`analyzer_noisy` は、ノイズを含むデータセットに対してロバストな画像解析を行うプログラムです。`analyzer` フォルダの実装をベースに、ノイズ対策の前処理を追加しています。

## 特徴

### ノイズ対策機能

1. **複合ノイズ除去**
   - ガウシアンノイズ: Non-local Means Denoising
   - 塩コショウノイズ: Median Filter
   - ぼかし: シャープネス強調

2. **照明正規化**
   - CLAHE（適応的ヒストグラム等化）
   - 背景除去による照明ムラ補正
   - 影・ビネット対策

3. **適応的二値化**
   - 明るさムラに強い適応的閾値処理
   - モルフォロジー処理によるノイズ除去

## インストール

```bash
cd analyzer_noisy
pip install -r requirements.txt
```

## 使用方法

### 基本的な使い方

```bash
python analyze.py
```

デフォルトで `dataset_noisy/` フォルダを解析し、結果を `analyzer_noisy/outputs/` に保存します。

### オプション指定

```bash
# データセットを指定
python analyze.py --dataset path/to/noisy_dataset

# 出力先を指定
python analyze.py --output path/to/output

# 可視化画像を保存
python analyze.py --save-images

# 処理枚数を制限（テスト用）
python analyze.py --max-images 100
```

### 全オプション

- `--dataset`: データセットディレクトリ（デフォルト: `dataset_noisy`）
- `--output`: 結果出力ディレクトリ（デフォルト: `analyzer_noisy/outputs`）
- `--save-images`: 可視化画像を保存する
- `--max-images`: 処理する最大画像数（テスト用）

## 出力ファイル

解析後、以下のファイルが生成されます：

```
analyzer_noisy/outputs/
├── analysis_results.csv       # 解析結果（全画像）
├── plots/
│   └── statistics.txt         # 統計情報
└── images/                    # 可視化画像（--save-images時）
    ├── image_0_A_B_0001.png
    ├── image_45_C_D_0002.png
    └── ...
```

### analysis_results.csv

各画像の解析結果を含むCSVファイル：

| カラム名 | 説明 |
|---------|------|
| filename | ファイル名 |
| heart_angle | 検出されたハート角度 |
| top_character | 右上文字（A-F） |
| top_score | 右上文字のマッチングスコア |
| bottom_character | 右下文字（A-F） |
| bottom_score | 右下文字のマッチングスコア |
| heart_angle_error | ハート角度の誤差（正解データがある場合） |
| top_correct | 右上文字の正誤 |
| bottom_correct | 右下文字の正誤 |
| gt_heart_angle | 正解のハート角度 |
| gt_top_letter | 正解の右上文字 |
| gt_bottom_letter | 正解の右下文字 |

### statistics.txt

全体の統計情報：

- ハート角度の平均誤差・最大誤差・標準偏差
- 右上文字認識精度（％）
- 右下文字認識精度（％）

## ノイズ対応の詳細

### 前処理パイプライン

`image_utils.py` の `preprocess_for_noise()` 関数で以下の処理を実行：

1. **Non-local Means Denoising** (h=10)
   - ガウシアンノイズ除去
   - エッジを保持しながらノイズを除去

2. **CLAHE** (clipLimit=2.0, tileGridSize=8x8)
   - コントラスト強調
   - 明るさの局所的な正規化

3. **Gaussian Blur** (kernel=3x3)
   - エッジの平滑化
   - 残留ノイズの除去

### 照明正規化

`normalize_illumination()` 関数で影・ビネット対策：

1. 大きなぼかし（51x51）で背景推定
2. 元画像を背景で除算して正規化
3. 照明ムラの影響を軽減

### 適応的二値化

固定閾値ではなく、適応的閾値処理を使用：

- `cv2.adaptiveThreshold()` with Gaussian window
- 明るさの局所的な変動に対応
- ノイズによる誤検出を低減

## テンプレート

`analyzer/templates/` フォルダのテンプレートを使用します。テンプレートが存在しない場合は、以下のコマンドで生成してください：

```bash
cd ../analyzer
python template_generator.py
```

## パフォーマンス

- **処理速度**: 約0.2-0.3秒/枚（前処理含む）
- **メモリ使用量**: 約500MB（1000枚解析時）
- **推奨環境**: Python 3.8+, OpenCV 4.8+

## トラブルシューティング

### テンプレートが見つからない

```
警告: テンプレートディレクトリが見つかりません: templates
```

→ `analyzer/template_generator.py` を実行してテンプレートを生成してください。

### 認識精度が低い

1. **ノイズレベルを確認**: `dataset_noisy/labels.csv` で各画像のノイズタイプを確認
2. **前処理パラメータ調整**: `image_utils.py` の `preprocess_for_noise()` 内のパラメータを調整
3. **閾値調整**: `recognize_character()` のマッチング閾値（デフォルト: 0.3）を変更

### メモリ不足

`--max-images` オプションで処理枚数を制限してください：

```bash
python analyze.py --max-images 100
```

## 比較: analyzer vs analyzer_noisy

| 項目 | analyzer | analyzer_noisy |
|-----|---------|----------------|
| 対象データ | クリーンデータ | ノイズ付きデータ |
| 前処理 | 最小限 | ノイズ除去・照明正規化 |
| 二値化 | 固定閾値 | 適応的閾値 |
| 処理速度 | 0.1秒/枚 | 0.3秒/枚 |
| 認識精度（クリーン） | 100% | 99%+ |
| 認識精度（ノイズ） | 70-80% | 90%+ |

## 参考

- 元実装: `analyzer/`
- ノイズデータセット生成: `dataset_generator/generate_dataset_with_noise.py`
- ノイズガイド: `dataset_generator/NOISE_GUIDE.md`
