"""
前処理手法集
各手法を組み合わせて使用できる
"""

import cv2
import numpy as np


def apply_preprocessing_pipeline(image, pipeline, debug_dir=None, filename_prefix=""):
    """
    前処理パイプラインを適用（各段階を保存可能）
    
    Args:
        image: 入力画像（グレースケール）
        pipeline: 前処理手法のリスト ["median_blur", "clahe", ...]
        debug_dir: デバッグ画像の保存先ディレクトリ（Noneなら保存しない）
        filename_prefix: デバッグ画像のファイル名プレフィックス
    
    Returns:
        処理済み画像
    """
    import os
    
    processed = image.copy()
    
    # 元画像を保存
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        original_path = os.path.join(debug_dir, f"{filename_prefix}_0_original.png")
        cv2.imwrite(original_path, image)
    
    for idx, method in enumerate(pipeline, 1):
        if method in PREPROCESSING_METHODS:
            processed = PREPROCESSING_METHODS[method](processed)
            
            # 各段階の画像を保存
            if debug_dir is not None:
                step_path = os.path.join(debug_dir, f"{filename_prefix}_{idx}_{method}.png")
                cv2.imwrite(step_path, processed)
        else:
            print(f"警告: 未知の前処理手法: {method}")
    
    return processed


# ============================================================
# 前処理手法の定義
# ============================================================

def no_preprocessing(image):
    """前処理なし"""
    return image


def median_blur_3(image):
    """Medianフィルタ (3x3)"""
    return cv2.medianBlur(image, 3)


def median_blur_5(image):
    """Medianフィルタ (5x5)"""
    return cv2.medianBlur(image, 5)


def gaussian_blur_3(image):
    """ガウシアンぼかし (3x3)"""
    return cv2.GaussianBlur(image, (3, 3), 0)


def gaussian_blur_5(image):
    """ガウシアンぼかし (5x5)"""
    return cv2.GaussianBlur(image, (5, 5), 0)


def bilateral_filter(image):
    """バイラテラルフィルタ（エッジ保存ぼかし）"""
    return cv2.bilateralFilter(image, 9, 75, 75)


def clahe(image):
    """CLAHE（適応的ヒストグラム等化）"""
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe_obj.apply(image)


def clahe_strong(image):
    """CLAHE 強め"""
    clahe_obj = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe_obj.apply(image)


def histogram_equalization(image):
    """ヒストグラム等化"""
    return cv2.equalizeHist(image)


def normalize_minmax(image):
    """Min-Max正規化"""
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)


def denoise_nlmeans(image):
    """Non-local Means Denoising（高品質ノイズ除去）"""
    return cv2.fastNlMeansDenoising(image, h=10)


def denoise_nlmeans_fast(image):
    """Non-local Means Denoising（高速版）"""
    return cv2.fastNlMeansDenoising(image, h=5)


def morphology_open(image):
    """モルフォロジー Opening（小さいノイズ除去）"""
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def morphology_close(image):
    """モルフォロジー Closing（隙間埋め）"""
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


def morphology_gradient(image):
    """モルフォロジー Gradient（輪郭強調）"""
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)


def sharpen(image):
    """シャープネス強調"""
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


def unsharp_mask(image):
    """アンシャープマスク"""
    gaussian = cv2.GaussianBlur(image, (5, 5), 0)
    unsharp = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    return np.clip(unsharp, 0, 255).astype(np.uint8)


def edge_enhancement(image):
    """エッジ強調（Laplacian）"""
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    enhanced = cv2.add(image, laplacian)
    return enhanced


def gamma_correction_bright(image):
    """ガンマ補正（明るく）"""
    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def gamma_correction_dark(image):
    """ガンマ補正（暗く）"""
    gamma = 0.8
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def contrast_stretching(image):
    """コントラストストレッチング"""
    min_val = np.percentile(image, 2)
    max_val = np.percentile(image, 98)
    
    if max_val - min_val < 1:
        return image
    
    stretched = ((image - min_val) / (max_val - min_val) * 255)
    return np.clip(stretched, 0, 255).astype(np.uint8)


def background_subtraction(image):
    """背景除去（大きなぼかしで背景推定）"""
    background = cv2.GaussianBlur(image, (51, 51), 0)
    subtracted = cv2.divide(image.astype(float), background.astype(float) + 1e-6) * 255
    return np.clip(subtracted, 0, 255).astype(np.uint8)


def erosion(image):
    """膨張処理"""
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


def dilation(image):
    """収縮処理"""
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# ============================================================
# 前処理手法の辞書
# ============================================================

PREPROCESSING_METHODS = {
    # 基本
    'none': no_preprocessing,
    
    # ノイズ除去
    'median_3': median_blur_3,
    'median_5': median_blur_5,
    'gaussian_3': gaussian_blur_3,
    'gaussian_5': gaussian_blur_5,
    'bilateral': bilateral_filter,
    'nlmeans': denoise_nlmeans,
    'nlmeans_fast': denoise_nlmeans_fast,
    
    # コントラスト調整
    'clahe': clahe,
    'clahe_strong': clahe_strong,
    'hist_eq': histogram_equalization,
    'normalize': normalize_minmax,
    'contrast_stretch': contrast_stretching,
    'gamma_bright': gamma_correction_bright,
    'gamma_dark': gamma_correction_dark,
    
    # シャープネス
    'sharpen': sharpen,
    'unsharp': unsharp_mask,
    'edge_enhance': edge_enhancement,
    
    # モルフォロジー
    'morph_open': morphology_open,
    'morph_close': morphology_close,
    'morph_gradient': morphology_gradient,
    'erosion': erosion,
    'dilation': dilation,
    
    # 背景処理
    'bg_subtract': background_subtraction,
}


# ============================================================
# 推奨パイプライン
# ============================================================

RECOMMENDED_PIPELINES = {
    'default': ['median_3'],
    'aggressive_denoise': ['median_5', 'bilateral', 'clahe'],
    'contrast_enhance': ['clahe', 'normalize'],
    'edge_preserve': ['bilateral', 'clahe', 'sharpen'],
    'no_preprocessing': ['none'],
    'light_denoise': ['median_3', 'clahe'],
    'heavy_process': ['nlmeans_fast', 'clahe', 'unsharp'],
    'simple_normalize': ['normalize'],
    'histogram_based': ['hist_eq', 'contrast_stretch'],
    'background_aware': ['bg_subtract', 'clahe'],
}


def get_available_methods():
    """利用可能な前処理手法のリストを取得"""
    return list(PREPROCESSING_METHODS.keys())


def get_recommended_pipelines():
    """推奨パイプラインのリストを取得"""
    return RECOMMENDED_PIPELINES


if __name__ == '__main__':
    # テスト
    print("利用可能な前処理手法:")
    for method in get_available_methods():
        print(f"  - {method}")
    
    print("\n推奨パイプライン:")
    for name, pipeline in get_recommended_pipelines().items():
        print(f"  - {name}: {pipeline}")
