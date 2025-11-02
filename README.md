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
├── analyzer/              # 解析エンジン
│   ├── analyze.py        # メイン解析処理
│   └── image_utils.py    # 画像処理関数
├── dataset_generator/     # データセット生成
│   ├── generate_dataset.py
│   └── dataset/          # 生成された100画像
├── debug/                 # デバッグ用スクリプト
├── tests/                 # テスト用スクリプト
├── test_results/          # テスト結果画像（100枚）
└── README.md

```

## 使い方

### 1. 環境構築

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install pillow opencv-python numpy
```

### 2. 単一画像の解析

```python
from analyzer.analyze import analyze_single_image

result = analyze_single_image('path/to/image.png')
print(result)
# {'heart_angle': 327.1, 'top_character': 'A', 'bottom_character': 'F'}
```

### 3. テスト実行

```cmd
python tests\test_with_output.py
```

テスト結果は `test_results/` ディレクトリに保存されます。

### 4. 混同行列の確認

```cmd
python tests\analyze_confusion.py
```

どの文字がどの文字と混同されているか確認できます。

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

`dataset_generator/dataset/` に100枚の画像があります：
- 白背景（255, 255, 255）
- 黒いハート（左上）
- 黒い円2個（右上・右下）、円内に白い回転した文字（A-F）

ファイル名: `image_XXXX_heartYYY_topZnn_bottomWmm.png`
- XXXX: 画像番号（0000-0099）
- YYY: ハート角度（0-359°）
- Z: 上部文字（A-F）
- nn: 上部文字の回転角度（0-359°）
- W: 下部文字（A-F）
- mm: 下部文字の回転角度（0-359°）

## 今後の改善案

- Cの認識精度向上（現在75%）
- Aの穴検出の安定化
- テンプレートマッチングの併用
- ディープラーニング（CNN）の導入

## ライセンス

MIT License
