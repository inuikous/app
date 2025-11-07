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
├── D-FINE_analyzer/       # D-FINE物体検出モデル
│   ├── convert_to_coco.py  # COCO形式変換
│   ├── train.py           # モデル学習
│   ├── inference.py       # 推論・解析
│   ├── utils.py           # ユーティリティ
│   ├── configs/           # 設定ファイル
│   ├── checkpoints/       # 学習済みモデル
│   ├── coco_dataset/      # COCO形式データ
│   └── outputs/           # 解析結果
│       ├── images/
│       ├── plots/
│       └── analysis_results.csv
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

#### テンプレートマッチング手法

```cmd
cd analyzer
python analyze.py --input ../dataset --ground-truth ../dataset/labels.csv --plot
```

解析結果は `analyzer/outputs/` に保存されます。

#### D-FINE物体検出手法

```cmd
cd D-FINE_analyzer
python run_pipeline.bat  # Windows
# または手動で
python convert_to_coco.py
python train.py
python inference.py --input ../dataset
```

解析結果は `D-FINE_analyzer/outputs/` に保存されます：
- `outputs/images/` - 解析済み画像
- `outputs/plots/` - 精度評価グラフ
- `outputs/analysis_results.csv` - 解析結果CSV

詳細は [D-FINE_analyzer/QUICKSTART.md](D-FINE_analyzer/QUICKSTART.md) を参照してください。

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

### 1. テンプレートマッチング（実装済み）
- **ディレクトリ**: `analyzer/`
- **手法**: 回転不変な特徴量ベース
- **特徴**: 軽量で高速、GPU不要
- **精度**: 文字認識 91.5%

### 2. D-FINE物体検出（実装済み）
- **ディレクトリ**: `D-FINE_analyzer/`
- **手法**: ディープラーニングベースの物体検出
- **モデル**: HuggingFace D-FINE（トランスフォーマー）
- **特徴**: 
  - エンドツーエンド学習
  - COCO形式データセット対応
  - GPU推奨
  - より高精度な検出が期待できる

### 手法の比較

| 項目 | テンプレートマッチング | D-FINE物体検出 |
|------|---------------------|----------------|
| 学習時間 | 不要 | 約30-60分（GPU） |
| 推論速度 | 高速（CPU可） | 中速（GPU推奨） |
| 精度 | 91.5% | 学習次第で向上可能 |
| GPU要件 | 不要 | 推奨（8GB+ VRAM） |
| 実装難易度 | 低 | 中 |

## 今後の改善案

- **テンプレートマッチング**:
  - Cの認識精度向上（現在75%）
  - Aの穴検出の安定化
  
- **D-FINE**:
  - ハート角度の回帰タスク追加
  - データ拡張による精度向上
  - より大規模なデータセットでの学習
  - マルチタスク学習の実装

## 関連ドキュメント

- [analyzer/README.md](analyzer/README.md) - テンプレートマッチング詳細
- [D-FINE_analyzer/README.md](D-FINE_analyzer/README.md) - D-FINE詳細
- [D-FINE_analyzer/QUICKSTART.md](D-FINE_analyzer/QUICKSTART.md) - D-FINEクイックスタート
- [dataset_generator/README.md](dataset_generator/README.md) - データセット生成

## ライセンス

MIT License
