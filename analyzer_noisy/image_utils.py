"""
画像解析ユーティリティ（ノイズ対応強化版）
ハート角度検出、文字認識、文字角度検出
ノイズに対するロバスト性を向上
"""

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUIなしでプロット生成
import os
from pathlib import Path
from preprocessing import apply_preprocessing_pipeline


# テンプレートマッチング用のグローバルキャッシュ
_TEMPLATES_CACHE = None
_TEMPLATES_CACHE_PREPROCESSED = {}  # 前処理済みテンプレートキャッシュ
_HEART_TEMPLATES_CACHE = None  # ハートテンプレートキャッシュ
_HEART_TEMPLATES_CACHE_PREPROCESSED = {}  # 前処理済みハートテンプレートキャッシュ

# 前処理パイプライン設定（グローバル）
_PREPROCESSING_PIPELINE = ['median_3']  # デフォルト

# デバッグモード設定（グローバル）
_DEBUG_MODE = False
_DEBUG_DIR = None


def set_preprocessing_pipeline(pipeline):
    """
    前処理パイプラインを設定
    
    Args:
        pipeline: 前処理手法のリスト ["median_3", "clahe", ...]
    """
    global _PREPROCESSING_PIPELINE, _TEMPLATES_CACHE_PREPROCESSED
    _PREPROCESSING_PIPELINE = pipeline
    # 前処理済みテンプレートキャッシュをクリア
    _TEMPLATES_CACHE_PREPROCESSED = {}
    print(f"前処理パイプラインを設定: {pipeline}")


def get_preprocessing_pipeline():
    """現在の前処理パイプラインを取得"""
    return _PREPROCESSING_PIPELINE


def set_debug_mode(enabled, debug_dir=None):
    """
    デバッグモードを設定（前処理の各段階を保存）
    
    Args:
        enabled: デバッグモードを有効にするか
        debug_dir: デバッグ画像の保存先ディレクトリ
    """
    global _DEBUG_MODE, _DEBUG_DIR
    _DEBUG_MODE = enabled
    _DEBUG_DIR = debug_dir
    if enabled:
        print(f"デバッグモード有効: {debug_dir}")
    else:
        print("デバッグモード無効")


def load_templates(template_dir='templates'):
    """
    テンプレート画像を読み込んでキャッシュ
    
    Args:
        template_dir: テンプレートディレクトリのパス
    
    Returns:
        dict: {文字: [(角度, テンプレート画像), ...], ...}
    """
    global _TEMPLATES_CACHE
    
    if _TEMPLATES_CACHE is not None:
        return _TEMPLATES_CACHE
    
    templates = {}
    template_path = Path(template_dir)
    
    if not template_path.exists():
        print(f"警告: テンプレートディレクトリが見つかりません: {template_dir}")
        print("analyzer/template_generator.py を実行してテンプレートを生成してください。")
        return {}
    
    for char in ['A', 'B', 'C', 'D', 'E', 'F']:
        templates[char] = []
        
        # 各角度のテンプレートを読み込み
        for angle in range(0, 360, 10):
            filename = f"{char}_{angle:03d}.png"
            filepath = template_path / filename
            
            if filepath.exists():
                img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    templates[char].append((angle, img))
    
    _TEMPLATES_CACHE = templates
    print(f"✓ テンプレート読み込み完了: {sum(len(v) for v in templates.values())} 個")
    
    return templates


def load_heart_templates(template_dir='heart_templates'):
    """
    ハートテンプレート画像を読み込んでキャッシュ
    
    Args:
        template_dir: ハートテンプレートディレクトリのパス
    
    Returns:
        list: [(角度, テンプレート画像), ...] のリスト
    """
    global _HEART_TEMPLATES_CACHE
    
    if _HEART_TEMPLATES_CACHE is not None:
        return _HEART_TEMPLATES_CACHE
    
    templates = []
    template_path = Path(template_dir)
    
    if not template_path.exists():
        print(f"警告: ハートテンプレートディレクトリが見つかりません: {template_dir}")
        print("analyzer/generate_heart_templates.py を実行してテンプレートを生成してください。")
        return []
    
    # 各角度のテンプレートを読み込み
    for angle in range(0, 360, 10):
        filename = f"heart_{angle:03d}.png"
        filepath = template_path / filename
        
        if filepath.exists():
            img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                templates.append((angle, img))
    
    _HEART_TEMPLATES_CACHE = templates
    print(f"✓ ハートテンプレート読み込み完了: {len(templates)} 個")
    
    return templates


def get_preprocessed_heart_templates():
    """
    前処理済みハートテンプレートを取得（キャッシュ付き）
    
    Returns:
        list: [(角度, 前処理済みテンプレート画像), ...] のリスト
    """
    global _HEART_TEMPLATES_CACHE_PREPROCESSED
    
    # 現在のパイプラインのキャッシュキー
    pipeline_key = tuple(_PREPROCESSING_PIPELINE)
    
    if pipeline_key in _HEART_TEMPLATES_CACHE_PREPROCESSED:
        return _HEART_TEMPLATES_CACHE_PREPROCESSED[pipeline_key]
    
    # 元のテンプレートを読み込み
    templates = load_heart_templates()
    
    if not templates:
        return []
    
    # 前処理を適用
    preprocessed_templates = []
    for angle, template in templates:
        # 前処理パイプラインを適用
        preprocessed = apply_preprocessing_pipeline(template, _PREPROCESSING_PIPELINE)
        preprocessed_templates.append((angle, preprocessed))
    
    # キャッシュに保存
    _HEART_TEMPLATES_CACHE_PREPROCESSED[pipeline_key] = preprocessed_templates
    
    print(f"✓ ハートテンプレートに前処理適用: {_PREPROCESSING_PIPELINE}")
    
    return preprocessed_templates


def get_preprocessed_templates():
    """
    前処理済みテンプレートを取得（キャッシュ付き）
    
    Returns:
        dict: {文字: [(角度, 前処理済みテンプレート画像), ...], ...}
    """
    global _TEMPLATES_CACHE_PREPROCESSED
    
    # 現在のパイプラインのキャッシュキー
    pipeline_key = tuple(_PREPROCESSING_PIPELINE)
    
    if pipeline_key in _TEMPLATES_CACHE_PREPROCESSED:
        return _TEMPLATES_CACHE_PREPROCESSED[pipeline_key]
    
    # 元のテンプレートを読み込み
    templates = load_templates()
    
    if not templates:
        return {}
    
    # 前処理を適用
    preprocessed_templates = {}
    for char, template_list in templates.items():
        preprocessed_templates[char] = []
        for angle, template in template_list:
            # 前処理パイプラインを適用
            preprocessed = apply_preprocessing_pipeline(template, _PREPROCESSING_PIPELINE)
            preprocessed_templates[char].append((angle, preprocessed))
    
    # キャッシュに保存
    _TEMPLATES_CACHE_PREPROCESSED[pipeline_key] = preprocessed_templates
    
    print(f"✓ テンプレートに前処理適用: {_PREPROCESSING_PIPELINE}")
    
    return preprocessed_templates


def preprocess_for_noise(image, noise_type='auto'):
    """
    ノイズに応じた前処理を適用
    
    Args:
        image: 入力画像（グレースケール）
        noise_type: ノイズタイプ（'auto', 'gaussian', 'salt_pepper', 'blur', 'shadow', 'vignette'）
    
    Returns:
        前処理済み画像
    """
    if noise_type == 'auto':
        # 複数の前処理を組み合わせて適用（軽量版）
        # 1. ガウシアンぼかしでノイズ除去
        denoised = cv2.GaussianBlur(image, (5, 5), 0)
        
        # 2. 適応的ヒストグラム等化（コントラスト・明るさ対策）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 3. Median filterで塩コショウノイズ除去
        smoothed = cv2.medianBlur(enhanced, 3)
        
        return smoothed
    
    elif noise_type == 'gaussian':
        # ガウシアンノイズ対策
        return cv2.GaussianBlur(image, (5, 5), 0)
    
    elif noise_type == 'salt_pepper':
        # 塩コショウノイズ対策
        return cv2.medianBlur(image, 5)
    
    elif noise_type == 'blur':
        # ぼかし対策（シャープネス強調）
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)
    
    elif noise_type in ['shadow', 'vignette']:
        # 照明ムラ対策（CLAHE）
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        return clahe.apply(image)
    
    else:
        return image


def normalize_illumination(image):
    """
    照明の正規化（影・ビネット対策）
    
    Args:
        image: 入力画像（グレースケール）
    
    Returns:
        正規化された画像
    """
    # 背景推定（大きなぼかし）
    background = cv2.GaussianBlur(image, (51, 51), 0)
    
    # 背景を除去
    normalized = cv2.divide(image.astype(float), background.astype(float) + 1e-6) * 255
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    
    return normalized


def detect_heart_angle(image_region):
    """
    ハートの角度を検出（テンプレートマッチング版 - 文字認識と同じアルゴリズム）
    
    Args:
        image_region: ハート領域の画像（NumPy配列、RGB）
    
    Returns:
        角度（度）、0度=上向き、時計回り=正
    """
    # ハートテンプレートを読み込み
    templates = load_heart_templates()
    
    if not templates:
        print("警告: ハートテンプレートが読み込めません")
        return 0.0
    
    # グレースケール変換
    if len(image_region.shape) == 3:
        gray = cv2.cvtColor(image_region, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_region
    
    # 前処理パイプラインを適用（デバッグモードでは各段階を保存）
    debug_dir = _DEBUG_DIR if _DEBUG_MODE else None
    processed = apply_preprocessing_pipeline(gray, _PREPROCESSING_PIPELINE,
                                            debug_dir=debug_dir,
                                            filename_prefix="heart")
    
    # 大津の二値化（領域サイズに依存しない頑健な手法）
    _, binary = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # デバッグ: 二値化画像を保存
    if debug_dir is not None:
        import os
        cv2.imwrite(os.path.join(debug_dir, "heart_2_binary.png"), binary)
    
    # 輪郭検出（文字と同じ設定）
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0
    
    # 最大の輪郭（ハート）を探す
    heart_contour = max(contours, key=cv2.contourArea)
    
    # バウンディングボックス
    x, y, w, h = cv2.boundingRect(heart_contour)
    
    # デバッグ: 輪郭情報を出力
    if debug_dir is not None:
        print(f"[DEBUG] ハート輪郭検出: バウンディングボックス=({x}, {y}, {w}, {h}), 面積={cv2.contourArea(heart_contour):.1f}")
    
    # ハート領域のマスクを作成
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [heart_contour], -1, 255, -1)
    
    # ハート領域を抽出
    heart_region = cv2.bitwise_and(binary, mask)
    
    # デバッグ: マスク適用後の画像を保存
    if debug_dir is not None:
        cv2.imwrite(os.path.join(debug_dir, "heart_3_masked.png"), heart_region)
    
    # ハート部分だけを切り出し
    if w > 0 and h > 0:
        heart_region_cropped = heart_region[y:y+h, x:x+w]
        
        # デバッグ: 切り出し後の画像を保存
        if debug_dir is not None:
            cv2.imwrite(os.path.join(debug_dir, "heart_4_cropped.png"), heart_region_cropped)
            print(f"[DEBUG] ハート切り出し後のサイズ: {heart_region_cropped.shape}")
        
        # テンプレートサイズにリサイズ（240x240）
        target_size = 240
        heart_resized = cv2.resize(heart_region_cropped, (target_size, target_size))
        
        # デバッグ: リサイズ後の画像を保存
        if debug_dir is not None:
            cv2.imwrite(os.path.join(debug_dir, "heart_5_resized.png"), heart_resized)
    else:
        return 0.0
    
    # 正規化（文字と同じ処理）
    heart_norm = cv2.normalize(heart_resized, None, 0, 255, cv2.NORM_MINMAX)
    
    # テンプレートと合わせるために反転（文字と同じ処理）
    # 現在の画像は黒背景・白ハートなので、反転が必要
    heart_norm = 255 - heart_norm
    
    # デバッグ: 正規化・反転後の画像を保存
    if debug_dir is not None:
        cv2.imwrite(os.path.join(debug_dir, "heart_6_normalized_inverted.png"), heart_norm)
        print(f"[DEBUG] 反転後のサイズ: {heart_norm.shape}, 値の範囲: [{heart_norm.min()}, {heart_norm.max()}]")
    
    # 前処理済みテンプレートを取得
    preprocessed_templates = get_preprocessed_heart_templates()
    
    # デバッグ: テンプレート数を確認
    if debug_dir is not None:
        print(f"[DEBUG] 比較するハートテンプレート数: {len(preprocessed_templates)}")
    
    # テンプレートマッチング（文字と全く同じアルゴリズム）
    best_angle = 0
    best_score = 0
    
    comparison_count = 0
    for angle, template in preprocessed_templates:
        # テンプレートを入力画像と同じサイズにリサイズ
        template_resized = cv2.resize(template, (heart_norm.shape[1], heart_norm.shape[0]))
        
        # テンプレートを正規化
        template_norm = cv2.normalize(template_resized, None, 0, 255, cv2.NORM_MINMAX)
        
        # 最初のテンプレートを保存してデバッグ
        if debug_dir is not None and comparison_count == 0:
            cv2.imwrite(os.path.join(debug_dir, f"heart_template_sample_{angle}.png"), template_norm)
            print(f"[DEBUG] サンプルハートテンプレート: {angle}°, サイズ={template_norm.shape}, 値=[{template_norm.min()}, {template_norm.max()}]")
        
        # 同じサイズの画像同士の類似度計算（正規化相互相関）
        heart_flat = heart_norm.flatten().astype(np.float32)
        template_flat = template_norm.flatten().astype(np.float32)
        
        # 平均を引いて正規化
        heart_mean = np.mean(heart_flat)
        template_mean = np.mean(template_flat)
        heart_centered = heart_flat - heart_mean
        template_centered = template_flat - template_mean
        
        # 正規化相互相関係数
        numerator = np.dot(heart_centered, template_centered)
        denominator = np.sqrt(np.dot(heart_centered, heart_centered) * np.dot(template_centered, template_centered))
        
        if denominator > 0:
            score = numerator / denominator
        else:
            score = 0
        
        # 最初の数個のスコアをデバッグ出力
        if debug_dir is not None and comparison_count < 3:
            print(f"[DEBUG] ハート{angle}°: スコア={score:.6f}")
        
        comparison_count += 1
        
        if score > best_score:
            best_score = score
            best_angle = angle
    
    # デバッグ: マッチング結果を出力
    if debug_dir is not None:
        print(f"[DEBUG] ハートマッチング結果: 角度={best_angle}°, スコア={best_score:.4f}")
    
    return best_angle


def recognize_character(image_region):
    """
    円内の文字を認識（テンプレートマッチング版、ノイズ対応強化 - 前処理最小化）
    
    Args:
        image_region: 円領域の画像（NumPy配列、RGB）
    
    Returns:
        tuple: (文字 (A-F or Unknown), マッチングスコア (0.0-1.0))
    """
    # テンプレートを読み込み
    templates = load_templates()
    
    if not templates:
        return ('Unknown', 0.0)
    
    # グレースケール変換
    if len(image_region.shape) == 3:
        gray = cv2.cvtColor(image_region, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_region
    
    # 前処理パイプラインを適用（デバッグモードでは各段階を保存）
    debug_dir = _DEBUG_DIR if _DEBUG_MODE else None
    processed = apply_preprocessing_pipeline(gray, _PREPROCESSING_PIPELINE, 
                                            debug_dir=debug_dir, 
                                            filename_prefix="char")
    
    # 円領域の抽出（二値化）
    # 適応的二値化
    binary = cv2.adaptiveThreshold(
        processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 5
    )
    
    # デバッグ: 二値化画像を保存
    if debug_dir is not None:
        import os
        cv2.imwrite(os.path.join(debug_dir, "char_2_binary.png"), binary)
    
    # 輪郭検出
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return ('Unknown', 0.0)
    
    # 最大の輪郭（円）を探す
    circle_contour = max(contours, key=cv2.contourArea)
    
    # バウンディングボックス
    x, y, w, h = cv2.boundingRect(circle_contour)
    
    # デバッグ: 輪郭情報を出力
    if debug_dir is not None:
        print(f"[DEBUG] 輪郭検出: バウンディングボックス=({x}, {y}, {w}, {h}), 面積={cv2.contourArea(circle_contour):.1f}")
    
    # 円内部のマスクを作成
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [circle_contour], -1, 255, -1)
    
    # 文字領域を抽出（白文字）
    char_region = cv2.bitwise_and(binary, mask)
    
    # デバッグ: マスク適用後の画像を保存
    if debug_dir is not None:
        cv2.imwrite(os.path.join(debug_dir, "char_3_masked.png"), char_region)
    
    # 文字部分だけを切り出し
    if w > 0 and h > 0:
        char_region_cropped = char_region[y:y+h, x:x+w]
        
        # デバッグ: 切り出し後の画像を保存
        if debug_dir is not None:
            cv2.imwrite(os.path.join(debug_dir, "char_4_cropped.png"), char_region_cropped)
            print(f"[DEBUG] 切り出し後のサイズ: {char_region_cropped.shape}")
        
        # リサイズしてテンプレートサイズに合わせる（200x200）
        target_size = 200  # テンプレートと同じサイズ
        char_resized = cv2.resize(char_region_cropped, (target_size, target_size))
        
        # デバッグ: リサイズ後の画像を保存
        if debug_dir is not None:
            cv2.imwrite(os.path.join(debug_dir, "char_5_resized.png"), char_resized)
    else:
        return ('Unknown', 0.0)
    
    # 正規化マッチング（前処理済みテンプレートを使用）
    best_match = None
    best_score = 0
    best_angle = 0
    
    # 文字を正規化
    char_norm = cv2.normalize(char_resized, None, 0, 255, cv2.NORM_MINMAX)
    
    # テンプレートと合わせるために反転（テンプレートは白背景・黒円・白文字）
    # 現在の画像は黒背景・白文字なので、反転が必要
    char_norm = 255 - char_norm
    
    # デバッグ: 正規化後の画像を保存
    if debug_dir is not None:
        cv2.imwrite(os.path.join(debug_dir, "char_6_normalized_inverted.png"), char_norm)
        print(f"[DEBUG] 反転後のサイズ: {char_norm.shape}, 値の範囲: [{char_norm.min()}, {char_norm.max()}]")
    
    # 前処理済みテンプレートを取得
    preprocessed_templates = get_preprocessed_templates()
    
    # デバッグ: テンプレート数を確認
    if debug_dir is not None:
        total_templates = sum(len(v) for v in preprocessed_templates.values())
        print(f"[DEBUG] 比較するテンプレート数: {total_templates}")
    
    comparison_count = 0
    for char, template_list in preprocessed_templates.items():
        for angle, template in template_list:
            # テンプレートを入力画像と同じサイズにリサイズ
            template_resized = cv2.resize(template, (char_norm.shape[1], char_norm.shape[0]))
            
            # テンプレートを正規化
            template_norm = cv2.normalize(template_resized, None, 0, 255, cv2.NORM_MINMAX)
            
            # 最初のテンプレートを保存してデバッグ
            if debug_dir is not None and comparison_count == 0:
                cv2.imwrite(os.path.join(debug_dir, f"template_sample_{char}_{angle}.png"), template_norm)
                print(f"[DEBUG] サンプルテンプレート: {char}_{angle}°, サイズ={template_norm.shape}, 値=[{template_norm.min()}, {template_norm.max()}]")
            
            # 同じサイズの画像同士の類似度計算（正規化相互相関）
            # 正規化してから相関係数を計算
            char_flat = char_norm.flatten().astype(np.float32)
            template_flat = template_norm.flatten().astype(np.float32)
            
            # 平均を引いて正規化
            char_mean = np.mean(char_flat)
            template_mean = np.mean(template_flat)
            char_centered = char_flat - char_mean
            template_centered = template_flat - template_mean
            
            # 正規化相互相関係数
            numerator = np.dot(char_centered, template_centered)
            denominator = np.sqrt(np.dot(char_centered, char_centered) * np.dot(template_centered, template_centered))
            
            if denominator > 0:
                score = numerator / denominator
            else:
                score = 0
            
            # 最初の数個のスコアをデバッグ出力
            if debug_dir is not None and comparison_count < 3:
                print(f"[DEBUG] {char}_{angle}°: スコア={score:.6f}")
            
            comparison_count += 1
            
            if score > best_score:
                best_score = score
                best_match = char
                best_angle = angle
    
    # デバッグ: マッチング結果を出力
    if debug_dir is not None:
        print(f"[DEBUG] マッチング結果: 文字={best_match}, スコア={best_score:.4f}, 角度={best_angle}°")
        print(f"[DEBUG] 閾値判定: {best_score:.4f} {'≥' if best_score >= 0.25 else '<'} 0.25")
    
    # スコアが低すぎる場合は Unknown（ノイズ対応で閾値を大幅に下げる）
    if best_score < 0.15:  # 0.25 → 0.15に変更
        return ('Unknown', best_score)
    
    return (best_match, best_score)


def draw_results_on_image(image, results, ground_truth=None):
    """
    画像に解析結果を描画
    
    Args:
        image: 元画像（NumPy配列、RGB）
        results: 解析結果の辞書
        ground_truth: 正解データの辞書（オプション）
    
    Returns:
        描画済み画像（NumPy配列、RGB）
    """
    # 画像をコピー
    output_image = image.copy()
    
    # テキスト描画の準備
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    # 結果を描画
    y_offset = 30
    
    # ハート角度
    text = f"Heart: {results['heart_angle']:.1f}°"
    if ground_truth:
        gt_angle = ground_truth.get('heart_angle', 0)
        error = abs(results['heart_angle'] - gt_angle)
        if error > 180:
            error = 360 - error
        text += f" (GT: {gt_angle:.1f}°, Err: {error:.1f}°)"
        color = (0, 255, 0) if error < 5 else (255, 165, 0) if error < 15 else (255, 0, 0)
    else:
        color = (255, 255, 255)
    
    cv2.putText(output_image, text, (10, y_offset), font, font_scale, color, thickness)
    y_offset += 35
    
    # 右上の文字
    text = f"Top: {results['top_character']} ({results['top_score']:.2f})"
    if ground_truth:
        gt_char = ground_truth.get('top_letter', 'Unknown')
        correct = results['top_character'] == gt_char
        text += f" (GT: {gt_char})"
        color = (0, 255, 0) if correct else (255, 0, 0)
    else:
        color = (255, 255, 255)
    
    cv2.putText(output_image, text, (10, y_offset), font, font_scale, color, thickness)
    y_offset += 35
    
    # 右下の文字
    text = f"Bottom: {results['bottom_character']} ({results['bottom_score']:.2f})"
    if ground_truth:
        gt_char = ground_truth.get('bottom_letter', 'Unknown')
        correct = results['bottom_character'] == gt_char
        text += f" (GT: {gt_char})"
        color = (0, 255, 0) if correct else (255, 0, 0)
    else:
        color = (255, 255, 255)
    
    cv2.putText(output_image, text, (10, y_offset), font, font_scale, color, thickness)
    
    return output_image


def generate_accuracy_plots(results_df, output_dir):
    """
    精度グラフを生成
    
    Args:
        results_df: pandas DataFrame（解析結果）
        output_dir: 出力ディレクトリ
    """
    # 統計情報を計算
    stats = {}
    
    # ハート角度の誤差
    if 'heart_angle_error' in results_df.columns:
        heart_errors = results_df['heart_angle_error'].dropna()
        stats['heart_mean_error'] = heart_errors.mean()
        stats['heart_max_error'] = heart_errors.max()
        stats['heart_std_error'] = heart_errors.std()
    
    # 文字認識精度
    if 'top_correct' in results_df.columns:
        stats['top_accuracy'] = (results_df['top_correct'].sum() / len(results_df)) * 100
    
    if 'bottom_correct' in results_df.columns:
        stats['bottom_accuracy'] = (results_df['bottom_correct'].sum() / len(results_df)) * 100
    
    # 統計情報を保存
    stats_path = os.path.join(output_dir, 'statistics.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write('解析統計（ノイズデータセット）\n')
        f.write('='*50 + '\n\n')
        
        if 'heart_mean_error' in stats:
            f.write(f"ハート角度:\n")
            f.write(f"  平均誤差: {stats['heart_mean_error']:.2f}°\n")
            f.write(f"  最大誤差: {stats['heart_max_error']:.2f}°\n")
            f.write(f"  標準偏差: {stats['heart_std_error']:.2f}°\n\n")
        
        if 'top_accuracy' in stats:
            f.write(f"右上文字認識精度: {stats['top_accuracy']:.2f}%\n")
        
        if 'bottom_accuracy' in stats:
            f.write(f"右下文字認識精度: {stats['bottom_accuracy']:.2f}%\n")
    
    print(f"✓ 統計情報を保存: {stats_path}")
    
    # グラフ生成
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    
    # 日本語フォント設定
    plt.rcParams['font.family'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け対策
    
    # ハート角度誤差のヒストグラム
    if 'heart_angle_error' in results_df.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(results_df['heart_angle_error'].dropna(), bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('角度誤差 (度)', fontsize=12)
        plt.ylabel('頻度', fontsize=12)
        plt.title('ハート角度検出の誤差分布', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'heart_angle_error_histogram.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ グラフ生成: heart_angle_error_histogram.png")
    
    # 文字認識精度の棒グラフ
    if 'top_accuracy' in stats and 'bottom_accuracy' in stats:
        plt.figure(figsize=(8, 6))
        labels = ['右上文字', '右下文字']
        accuracies = [stats['top_accuracy'], stats['bottom_accuracy']]
        colors = ['#2ecc71', '#3498db']
        
        bars = plt.bar(labels, accuracies, color=colors, edgecolor='black', alpha=0.8)
        plt.ylabel('認識精度 (%)', fontsize=12)
        plt.title('文字認識精度', fontsize=14)
        plt.ylim(0, 105)
        plt.grid(True, axis='y', alpha=0.3)
        
        # 値をバーの上に表示
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.savefig(os.path.join(output_dir, 'character_recognition_accuracy.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ グラフ生成: character_recognition_accuracy.png")
    
    # 総合精度のサマリー
    plt.figure(figsize=(10, 6))
    
    # 3つのメトリクスを表示
    metrics = []
    values = []
    colors_list = []
    
    if 'heart_mean_error' in stats:
        metrics.append('ハート角度\n平均誤差')
        values.append(stats['heart_mean_error'])
        colors_list.append('#e74c3c')
    
    if 'top_accuracy' in stats:
        metrics.append('右上文字\n認識精度')
        values.append(stats['top_accuracy'])
        colors_list.append('#2ecc71')
    
    if 'bottom_accuracy' in stats:
        metrics.append('右下文字\n認識精度')
        values.append(stats['bottom_accuracy'])
        colors_list.append('#3498db')
    
    if metrics:
        bars = plt.bar(metrics, values, color=colors_list, edgecolor='black', alpha=0.8)
        plt.ylabel('値', fontsize=12)
        plt.title('総合解析結果サマリー', fontsize=14)
        plt.grid(True, axis='y', alpha=0.3)
        
        # 値をバーの上に表示
        for bar, val, metric in zip(bars, values, metrics):
            height = bar.get_height()
            if 'ハート' in metric:
                label = f'{val:.2f}°'
            else:
                label = f'{val:.1f}%'
            plt.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                    label, ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.savefig(os.path.join(output_dir, 'summary.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ グラフ生成: summary.png")
