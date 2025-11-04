# パターンマッチングシステム

画像からハート形状の角度とアルファベット文字（A-F）を検出するシステムです。

## 機能

- **ハート角度検出**: 左上のハート形状の回転角度を検出（精度: 平均誤差 0.22°）
- **文字認識**: 右上・右下の円内のアルファベット文字（A-F）を認識（精度: 91.5%）

## 精度

### ハート角度
- 平均誤差: **0.22°**
- 最大誤差: **0.59°**

### 文字認識（全体: 91.5%）
- **A**: 89.2%
- **B**: 100% ✅
- **C**: 75.0%
- **D**: 100% ✅
- **E**: 100% ✅
- **F**: 86.1%

## ディレクトリ構成

```
app/
├── analyzer/              # 解析エンジン（テンプレートマッチング）
│   ├── analyze.py        # メイン解析処理
│   ├── image_utils.py    # 画像処理関数
│   ├── visualize_accuracy.py  # グラフ生成
│   └── outputs/          # 解析結果の出力先
│       ├── images/       # 解析済み画像
│       ├── plots/        # 精度グラフ
│       └── analysis_results.csv
├── D-FINE_analyzer/       # D-FINE物体検出モデル（開発中）
├── dataset_generator/     # データセット生成
│   └── generate_dataset.py
├── dataset/               # 生成された100画像とラベル
│   ├── image_*.png
│   └── labels.csv
└── README.md
```

## 使い方

### 1. 環境構築

```cmd
python -m venv .venv
.venv\Scripts\activate
cd analyzer
pip install -r requirements.txt
```

### 2. データセット生成

```cmd
cd dataset_generator
python generate_dataset.py
```

生成された100枚の画像とラベルは `dataset/` ディレクトリに保存されます。

### 3. 画像解析の実行

```cmd
cd analyzer
python analyze.py --input ../dataset --ground-truth ../dataset/labels.csv --plot
```

解析結果は `analyzer/outputs/` に保存されます：
- `outputs/images/` - 解析済み画像
- `outputs/plots/` - 精度評価グラフ
- `outputs/analysis_results.csv` - 解析結果CSV

## アルゴリズム

### ハート角度検出
1. 黒いハート形状を輪郭検出
2. 重心から最も遠い点を検出（ハートの上部頂点）
3. atan2で角度を計算し、データセットの回転方向に合わせて調整

### 文字認識
回転不変な特徴量を使用：

1. **穴の数（Euler数）**: A=1, B=2, D=1, C/E/F=0
2. **ピクセル密度**: 文字の太さや面積
3. **コンパクトネス**: 周囲長²/面積（形状の複雑さ）

これらの特徴を組み合わせて、統計的に分類します。

## データセット

`dataset/` に100枚の画像があります：
- 白背景（255, 255, 255）
- 黒いハート（左下）
- 黒い円2個（右上・右下）、円内に白い回転した文字（A-F）

ファイル名: `image_XXXX_heartYYY_topZnn_bottomWmm.png`
- XXXX: 画像番号（0000-0099）
- YYY: ハート角度（0-359°）
- Z: 上部文字（A-F）
- nn: 上部文字の回転角度（0-359°）
- W: 下部文字（A-F）
- mm: 下部文字の回転角度（0-359°）

## アプローチ

### 1. テンプレートマッチング（現在実装済み）
- `analyzer/` ディレクトリ
- 回転不変な特徴量ベース
- 軽量で高速

### 2. D-FINE物体検出（開発中）
- `D-FINE_analyzer/` ディレクトリ
- ディープラーニングベースの高精度検出
- トランスフォーマーモデル

## 今後の改善案

- **テンプレートマッチング**:
  - Cの認識精度向上（現在75%）
  - Aの穴検出の安定化
  
- **D-FINE導入**:
  - COCO形式へのデータセット変換
  - ファインチューニング実装
  - エンドツーエンド検出システム

## ライセンス

MIT License
