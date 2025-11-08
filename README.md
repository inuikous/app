# パターンマッチングシステム

画像からハート形状の角度とアルファベット文字（A-F）を検出するシステムです。

## 機能

- **ハート角度検出**: 左上のハート形状の回転角度を検出
- **文字認識**: 右上・右下の円内のアルファベット文字（A-F）を認識

## 精度検証結果（テストデータ500枚）

### テンプレートマッチング
- **ハート検出率**: 100.0% (500/500)
- **上の文字認識**: 100.0% (500/500)
- **下の文字認識**: 100.0% (500/500)

### D-FINE物体検出
- **ハート検出率**: 100.0% (500/500)
- **上の文字認識**: 99.0% (495/500)
- **下の文字認識**: 98.4% (492/500)
- **主な誤認識**: A↔F (7回)、E↔F (5回)

### 手法比較

| 手法 | ハート検出 | 上の文字 | 下の文字 | 総合 |
|------|-----------|---------|---------|------|
| **テンプレートマッチング** | 100.0% | **100.0%** | **100.0%** | **100.0%** |
| **D-FINE物体検出** | 100.0% | 99.0% | 98.4% | 98.7% |

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
├── dataset/               # 学習用データセット（1000画像）
│   ├── image_*.png
│   └── labels.csv
├── test_dataset/          # テスト用データセット（500画像）
│   ├── test_image_*.png
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

#### 学習用データ（1000枚）

```cmd
cd dataset_generator
python generate_dataset.py
```

生成された1000枚の画像とラベルは `dataset/` ディレクトリに保存されます。

#### テスト用データ（500枚）

```cmd
cd dataset_generator
python generate_test_dataset.py
```

生成された500枚の画像とラベルは `test_dataset/` ディレクトリに保存されます。

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

### 学習用データ（dataset/）
- **枚数**: 1000枚（800 train / 200 validation）
- **用途**: D-FINEモデルの学習
- **乱数シード**: 42

### テスト用データ（test_dataset/）
- **枚数**: 500枚
- **用途**: 最終精度検証（学習に未使用）
- **乱数シード**: 12345（学習データとは異なる）

### 画像仕様
- 白背景（255, 255, 255）
- 黒いハート（左下）
- 黒い円2個（右上・右下）、円内に白い回転した文字（A-F）

### ファイル命名規則
`image_XXXX_heartYYY_topZnn_bottomWmm.png`（学習用）  
`test_image_XXXX_heartYYY_topZnn_bottomWmm.png`（テスト用）

- XXXX: 画像番号
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

### 手法の詳細比較

| 項目 | テンプレートマッチング | D-FINE物体検出 |
|------|---------------------|----------------|
| **精度（500枚テスト）** | **100.0%** ⭐ | 98.7% |
| 学習時間 | 不要 | 50エポック（約1-2時間、GPU） |
| 推論速度 | 高速（CPU可） | 中速（GPU推奨） |
| メモリ使用量 | 数MB | 数GB（モデル+VRAM） |
| GPU要件 | 不要 | 推奨（8GB+ VRAM） |
| 実装難易度 | 低 | 中 |
| 新規文字追加 | テンプレート作成のみ | 再学習が必要 |
| 保守性 | 高（シンプル） | 中（複雑） |

### 推奨手法

このプロジェクトの要件（固定フォント、シンプル背景、6文字認識）では、**テンプレートマッチングが最適解**です。

**テンプレートマッチングを推奨する理由：**
- ✅ 精度100%達成（D-FINEより1.3%高い）
- ✅ GPU不要で動作可能
- ✅ 実装・運用コストが低い
- ✅ 処理速度が高速
- ✅ メモリ使用量が少ない

**D-FINEが有効なケース：**
- フォントが複数種類ある場合
- 文字サイズが大きく変動する場合
- 背景が複雑で前処理が困難な場合
- 新しい文字クラスを頻繁に追加する場合

## 学習済みモデル

### D-FINEモデル
- **ベースモデル**: ustc-community/dfine-xlarge-coco
- **パラメータ数**: 62.5M
- **学習データ**: 1000枚（800 train / 200 validation）
- **学習エポック**: 50（Best: Epoch 30）
- **Best Val Loss**: 1.8700
- **検出クラス**: heart(0), A(1), B(2), C(3), D(4), E(5), F(6)
- **信頼度閾値**: 0.25
- **学習設定**: freeze_backbone=true

## 今後の改善案（D-FINEで100%を目指す場合）

### Phase 1: データ改善（推奨度: ★★★★★）
- A/E/F画像を重点的に増やす（各300枚）
- 紛らわしい角度・位置のバリエーション追加
- 総データセット1500枚で再学習

### Phase 2: 学習最適化（推奨度: ★★★★☆）
- エポック数を100に増加
- Early Stopping patience を30に拡大
- Learning Rate を5e-6に微調整

### Phase 3: 推論最適化（推奨度: ★★★★☆）
- テストタイム拡張（TTA）の実装
- 信頼度閾値の最適化（0.3-0.5でテスト）
- モデルアンサンブル（複数シードで学習）

## 関連ドキュメント

- [analyzer/README.md](analyzer/README.md) - テンプレートマッチング詳細
- [D-FINE_analyzer/README.md](D-FINE_analyzer/README.md) - D-FINE詳細
- [D-FINE_analyzer/QUICKSTART.md](D-FINE_analyzer/QUICKSTART.md) - D-FINEクイックスタート
- [dataset_generator/README.md](dataset_generator/README.md) - データセット生成

## ライセンス

MIT License
