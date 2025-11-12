"""
ノイズを含むデータセット生成スクリプト
既存のgenerate_dataset.pyにノイズ機能を追加したバージョン
"""

import os
import csv
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import math
import numpy as np


# 設定
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
BACKGROUND_COLOR = (255, 255, 255)

# ハート
HEART_SIZE = 150
HEART_COLOR = (0, 0, 0)
HEART_X = 200
HEART_Y = 400

# 円と文字
CIRCLE_RADIUS = 100
CIRCLE_COLOR = (0, 0, 0)
FONT_SIZE = 80
ALPHABET_CHOICES = ['A', 'B', 'C', 'D', 'E', 'F']

# 右上・右下の円の位置
TOP_X = 650
TOP_Y = 150
BOTTOM_X = 650
BOTTOM_Y = 450

# データセット
NUM_IMAGES = 100
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset_noisy')
RANDOM_SEED = 42

# ノイズ設定
NOISE_CONFIG = {
    'gaussian_noise': {
        'enabled': True,
        'probability': 0.3,  # 30%の画像に適用
        'std_dev': 10  # 標準偏差（0-255）
    },
    'salt_pepper_noise': {
        'enabled': True,
        'probability': 0.2,  # 20%の画像に適用
        'amount': 0.005  # ノイズの量（0.5%のピクセル）
    },
    'blur': {
        'enabled': True,
        'probability': 0.25,  # 25%の画像に適用
        'radius': 1.5  # ぼかし半径
    },
    'brightness': {
        'enabled': True,
        'probability': 0.3,  # 30%の画像に適用
        'range': (0.8, 1.2)  # 明るさの倍率範囲
    },
    'rotation_jitter': {
        'enabled': True,
        'probability': 0.2,  # 20%の画像に適用
        'max_angle': 2  # 最大±2度のズレ
    },
    'position_jitter': {
        'enabled': True,
        'probability': 0.25,  # 25%の画像に適用
        'max_offset': 5  # 最大±5ピクセルのズレ
    },
    'background_noise': {
        'enabled': True,
        'probability': 0.15,  # 15%の画像に適用
        'num_spots': (3, 8),  # ランダムなスポット数
        'spot_size': (5, 15)  # スポットサイズ範囲
    },
    'contrast': {
        'enabled': True,
        'probability': 0.3,  # 30%の画像に適用
        'range': (0.7, 1.3)  # コントラスト倍率範囲（1.0が元のコントラスト）
    },
    'saturation': {
        'enabled': True,
        'probability': 0.2,  # 20%の画像に適用
        'range': (0.8, 1.2)  # 彩度倍率範囲（白黒画像でも微妙な効果）
    },
    'gamma': {
        'enabled': True,
        'probability': 0.25,  # 25%の画像に適用
        'range': (0.7, 1.3)  # ガンマ補正値（1.0が補正なし）
    },
    'shadow': {
        'enabled': True,
        'probability': 0.15,  # 15%の画像に適用
        'intensity': (0.3, 0.7)  # 影の濃さ範囲
    },
    'vignette': {
        'enabled': True,
        'probability': 0.1,  # 10%の画像に適用
        'intensity': (0.2, 0.5)  # ビネット効果の強さ
    }
}


def add_gaussian_noise(img, std_dev=10):
    """ガウシアンノイズを追加"""
    img_array = np.array(img)
    noise = np.random.normal(0, std_dev, img_array.shape)
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)


def add_salt_pepper_noise(img, amount=0.005):
    """塩コショウノイズを追加"""
    img_array = np.array(img)
    
    # Salt (白いノイズ)
    num_salt = int(amount * img_array.size * 0.5)
    coords = [np.random.randint(0, i - 1, num_salt) for i in img_array.shape[:2]]
    img_array[coords[0], coords[1]] = 255
    
    # Pepper (黒いノイズ)
    num_pepper = int(amount * img_array.size * 0.5)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in img_array.shape[:2]]
    img_array[coords[0], coords[1]] = 0
    
    return Image.fromarray(img_array)


def adjust_brightness(img, factor):
    """明るさを調整"""
    img_array = np.array(img)
    adjusted = np.clip(img_array * factor, 0, 255).astype(np.uint8)
    return Image.fromarray(adjusted)


def adjust_contrast(img, factor):
    """コントラストを調整"""
    img_array = np.array(img).astype(np.float32)
    mean = np.mean(img_array)
    adjusted = (img_array - mean) * factor + mean
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return Image.fromarray(adjusted)


def adjust_saturation(img, factor):
    """彩度を調整（グレースケール寄りに）"""
    img_array = np.array(img).astype(np.float32)
    # RGBからグレースケールへの変換ウェイト
    gray = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
    gray_3ch = np.stack([gray, gray, gray], axis=2)
    # 元の画像とグレースケールをブレンド
    adjusted = img_array * factor + gray_3ch * (1 - factor)
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return Image.fromarray(adjusted)


def adjust_gamma(img, gamma):
    """ガンマ補正を適用"""
    img_array = np.array(img).astype(np.float32) / 255.0
    corrected = np.power(img_array, gamma)
    corrected = (corrected * 255).astype(np.uint8)
    return Image.fromarray(corrected)


def add_shadow(img, intensity):
    """影効果を追加（画像の一部を暗くする）"""
    img_array = np.array(img).astype(np.float32)
    h, w = img_array.shape[:2]
    
    # ランダムな位置と形の影を作成
    shadow_type = random.choice(['corner', 'edge', 'spot'])
    
    if shadow_type == 'corner':
        # 角から影
        corner = random.choice(['top_left', 'top_right', 'bottom_left', 'bottom_right'])
        y_grid, x_grid = np.ogrid[:h, :w]
        
        if corner == 'top_left':
            distance = np.sqrt((x_grid / w) ** 2 + (y_grid / h) ** 2)
        elif corner == 'top_right':
            distance = np.sqrt(((w - x_grid) / w) ** 2 + (y_grid / h) ** 2)
        elif corner == 'bottom_left':
            distance = np.sqrt((x_grid / w) ** 2 + ((h - y_grid) / h) ** 2)
        else:  # bottom_right
            distance = np.sqrt(((w - x_grid) / w) ** 2 + ((h - y_grid) / h) ** 2)
        
        shadow_mask = 1 - (distance * intensity)
        
    elif shadow_type == 'edge':
        # 端から影
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        y_grid, x_grid = np.ogrid[:h, :w]
        
        if edge == 'top':
            shadow_mask = 1 - (y_grid / h * intensity)
        elif edge == 'bottom':
            shadow_mask = 1 - ((h - y_grid) / h * intensity)
        elif edge == 'left':
            shadow_mask = 1 - (x_grid / w * intensity)
        else:  # right
            shadow_mask = 1 - ((w - x_grid) / w * intensity)
    
    else:  # spot
        # スポット影
        center_x = random.randint(int(w * 0.3), int(w * 0.7))
        center_y = random.randint(int(h * 0.3), int(h * 0.7))
        y_grid, x_grid = np.ogrid[:h, :w]
        distance = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)
        max_dist = np.sqrt(w ** 2 + h ** 2) / 2
        shadow_mask = 1 - (distance / max_dist * intensity)
    
    shadow_mask = np.clip(shadow_mask, 1 - intensity, 1)
    shadow_mask = np.expand_dims(shadow_mask, axis=2)
    
    shadowed = img_array * shadow_mask
    shadowed = np.clip(shadowed, 0, 255).astype(np.uint8)
    return Image.fromarray(shadowed)


def add_vignette(img, intensity):
    """ビネット効果を追加（周辺を暗くする）"""
    img_array = np.array(img).astype(np.float32)
    h, w = img_array.shape[:2]
    
    # 中心からの距離に基づくマスクを作成
    y_grid, x_grid = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2
    
    distance = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)
    max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
    
    vignette_mask = 1 - (distance / max_distance * intensity)
    vignette_mask = np.clip(vignette_mask, 1 - intensity, 1)
    vignette_mask = np.expand_dims(vignette_mask, axis=2)
    
    vignetted = img_array * vignette_mask
    vignetted = np.clip(vignetted, 0, 255).astype(np.uint8)
    return Image.fromarray(vignetted)


def add_background_spots(draw, num_spots, spot_size_range):
    """背景にランダムなスポットを追加"""
    for _ in range(num_spots):
        x = random.randint(0, IMAGE_WIDTH)
        y = random.randint(0, IMAGE_HEIGHT)
        size = random.randint(*spot_size_range)
        color = random.randint(200, 240)  # 薄いグレー
        draw.ellipse([x - size, y - size, x + size, y + size], fill=(color, color, color))


def draw_heart(draw, center_x, center_y, size, angle_deg, color):
    """ハートを描画"""
    points = []
    
    for t_deg in range(0, 360, 5):
        t = math.radians(t_deg)
        # ハートのパラメトリック方程式
        x = 16 * math.sin(t) ** 3
        y = -(13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t))
        
        # スケーリング
        x = x * size / 20
        y = y * size / 20
        
        # 回転適用
        angle_rad = math.radians(angle_deg)
        rotated_x = x * math.cos(angle_rad) - y * math.sin(angle_rad)
        rotated_y = x * math.sin(angle_rad) + y * math.cos(angle_rad)
        
        # 位置調整
        points.append((center_x + rotated_x, center_y + rotated_y))
    
    draw.polygon(points, fill=color)


def draw_circle_with_letter(center_x, center_y, radius, letter, angle_deg, circle_color):
    """円と回転した文字を描画"""
    # 一時的な画像を作成（透明背景）
    temp_size = radius * 3
    temp_img = Image.new('RGBA', (temp_size, temp_size), (255, 255, 255, 0))
    temp_draw = ImageDraw.Draw(temp_img)
    temp_center = temp_size // 2
    
    # 円を描画
    temp_draw.ellipse(
        [temp_center - radius, temp_center - radius,
         temp_center + radius, temp_center + radius],
        fill=circle_color
    )
    
    # フォント
    try:
        font = ImageFont.truetype("arial.ttf", FONT_SIZE)
    except:
        font = ImageFont.load_default()
    
    # 文字のサイズを取得
    bbox = temp_draw.textbbox((0, 0), letter, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    # 文字を中央に描画（白色）
    text_x = temp_center - text_w // 2
    text_y = temp_center - text_h // 2 - bbox[1]
    temp_draw.text((text_x, text_y), letter, fill=(255, 255, 255), font=font)
    
    # 回転（反時計回りにする必要がある）
    rotated = temp_img.rotate(-angle_deg, expand=False, fillcolor=(255, 255, 255, 0))
    
    return rotated, center_x - temp_size // 2, center_y - temp_size // 2


def apply_noise(img, noise_types_applied):
    """ノイズを適用"""
    
    # ガウシアンノイズ
    if NOISE_CONFIG['gaussian_noise']['enabled'] and random.random() < NOISE_CONFIG['gaussian_noise']['probability']:
        img = add_gaussian_noise(img, NOISE_CONFIG['gaussian_noise']['std_dev'])
        noise_types_applied.append('gaussian')
    
    # 塩コショウノイズ
    if NOISE_CONFIG['salt_pepper_noise']['enabled'] and random.random() < NOISE_CONFIG['salt_pepper_noise']['probability']:
        img = add_salt_pepper_noise(img, NOISE_CONFIG['salt_pepper_noise']['amount'])
        noise_types_applied.append('salt_pepper')
    
    # ぼかし
    if NOISE_CONFIG['blur']['enabled'] and random.random() < NOISE_CONFIG['blur']['probability']:
        img = img.filter(ImageFilter.GaussianBlur(radius=NOISE_CONFIG['blur']['radius']))
        noise_types_applied.append('blur')
    
    # 明るさ調整
    if NOISE_CONFIG['brightness']['enabled'] and random.random() < NOISE_CONFIG['brightness']['probability']:
        factor = random.uniform(*NOISE_CONFIG['brightness']['range'])
        img = adjust_brightness(img, factor)
        noise_types_applied.append(f'brightness_{factor:.2f}')
    
    # コントラスト調整
    if NOISE_CONFIG['contrast']['enabled'] and random.random() < NOISE_CONFIG['contrast']['probability']:
        factor = random.uniform(*NOISE_CONFIG['contrast']['range'])
        img = adjust_contrast(img, factor)
        noise_types_applied.append(f'contrast_{factor:.2f}')
    
    # 彩度調整
    if NOISE_CONFIG['saturation']['enabled'] and random.random() < NOISE_CONFIG['saturation']['probability']:
        factor = random.uniform(*NOISE_CONFIG['saturation']['range'])
        img = adjust_saturation(img, factor)
        noise_types_applied.append(f'saturation_{factor:.2f}')
    
    # ガンマ補正
    if NOISE_CONFIG['gamma']['enabled'] and random.random() < NOISE_CONFIG['gamma']['probability']:
        gamma = random.uniform(*NOISE_CONFIG['gamma']['range'])
        img = adjust_gamma(img, gamma)
        noise_types_applied.append(f'gamma_{gamma:.2f}')
    
    # 影効果
    if NOISE_CONFIG['shadow']['enabled'] and random.random() < NOISE_CONFIG['shadow']['probability']:
        intensity = random.uniform(*NOISE_CONFIG['shadow']['intensity'])
        img = add_shadow(img, intensity)
        noise_types_applied.append(f'shadow_{intensity:.2f}')
    
    # ビネット効果
    if NOISE_CONFIG['vignette']['enabled'] and random.random() < NOISE_CONFIG['vignette']['probability']:
        intensity = random.uniform(*NOISE_CONFIG['vignette']['intensity'])
        img = add_vignette(img, intensity)
        noise_types_applied.append(f'vignette_{intensity:.2f}')
    
    return img


def generate_single_image(filepath, heart_angle, top_letter, top_angle, bottom_letter, bottom_angle):
    """1枚の画像を生成（ノイズ付き）"""
    noise_types_applied = []
    
    # 位置のジッター
    heart_x = HEART_X
    heart_y = HEART_Y
    top_x = TOP_X
    top_y = TOP_Y
    bottom_x = BOTTOM_X
    bottom_y = BOTTOM_Y
    
    if NOISE_CONFIG['position_jitter']['enabled'] and random.random() < NOISE_CONFIG['position_jitter']['probability']:
        max_offset = NOISE_CONFIG['position_jitter']['max_offset']
        heart_x += random.randint(-max_offset, max_offset)
        heart_y += random.randint(-max_offset, max_offset)
        top_x += random.randint(-max_offset, max_offset)
        top_y += random.randint(-max_offset, max_offset)
        bottom_x += random.randint(-max_offset, max_offset)
        bottom_y += random.randint(-max_offset, max_offset)
        noise_types_applied.append('position_jitter')
    
    # 回転のジッター
    heart_angle_actual = heart_angle
    top_angle_actual = top_angle
    bottom_angle_actual = bottom_angle
    
    if NOISE_CONFIG['rotation_jitter']['enabled'] and random.random() < NOISE_CONFIG['rotation_jitter']['probability']:
        max_angle = NOISE_CONFIG['rotation_jitter']['max_angle']
        heart_angle_actual += random.uniform(-max_angle, max_angle)
        top_angle_actual += random.uniform(-max_angle, max_angle)
        bottom_angle_actual += random.uniform(-max_angle, max_angle)
        noise_types_applied.append('rotation_jitter')
    
    # 基本画像
    img = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    
    # 背景ノイズ
    if NOISE_CONFIG['background_noise']['enabled'] and random.random() < NOISE_CONFIG['background_noise']['probability']:
        num_spots = random.randint(*NOISE_CONFIG['background_noise']['num_spots'])
        add_background_spots(draw, num_spots, NOISE_CONFIG['background_noise']['spot_size'])
        noise_types_applied.append('background_spots')
    
    # ハート描画
    draw_heart(draw, heart_x, heart_y, HEART_SIZE, heart_angle_actual, HEART_COLOR)
    
    # 右上の円と文字
    top_img, top_img_x, top_img_y = draw_circle_with_letter(
        top_x, top_y, CIRCLE_RADIUS, top_letter, top_angle_actual, CIRCLE_COLOR
    )
    img.paste(top_img, (top_img_x, top_img_y), top_img)
    
    # 右下の円と文字
    bottom_img, bottom_img_x, bottom_img_y = draw_circle_with_letter(
        bottom_x, bottom_y, CIRCLE_RADIUS, bottom_letter, bottom_angle_actual, CIRCLE_COLOR
    )
    img.paste(bottom_img, (bottom_img_x, bottom_img_y), bottom_img)
    
    # ノイズ適用
    img = apply_noise(img, noise_types_applied)
    
    # 保存
    img.save(filepath)
    
    return noise_types_applied


def main():
    """メイン処理"""
    # ディレクトリ作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # シード設定
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
    
    # データを生成
    dataset = []
    noise_stats = {}
    
    for i in range(NUM_IMAGES):
        # パラメータ生成
        heart_angle = random.randint(0, 359)
        top_letter = random.choice(ALPHABET_CHOICES)
        top_angle = random.randint(0, 359)
        bottom_letter = random.choice(ALPHABET_CHOICES)
        bottom_angle = random.randint(0, 359)
        
        # ファイル名
        filename = f'image_{i:04d}_heart{heart_angle}_top{top_letter}{top_angle}_bottom{bottom_letter}{bottom_angle}.png'
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        print(f'Generating {i+1}/{NUM_IMAGES}: {filename}')
        
        try:
            # 画像生成
            noise_types = generate_single_image(filepath, heart_angle, top_letter, top_angle, bottom_letter, bottom_angle)
            
            # ノイズ統計
            for noise_type in noise_types:
                noise_stats[noise_type] = noise_stats.get(noise_type, 0) + 1
            
            # データ記録
            dataset.append({
                'filename': filename,
                'heart_angle': heart_angle,
                'top_letter': top_letter,
                'top_angle': top_angle,
                'bottom_letter': bottom_letter,
                'bottom_angle': bottom_angle,
                'noise_types': ';'.join(noise_types) if noise_types else 'none'
            })
            
        except Exception as e:
            print(f'ERROR generating {filename}: {e}')
            import traceback
            traceback.print_exc()
    
    # CSVに保存
    csv_path = os.path.join(OUTPUT_DIR, 'labels.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['filename', 'heart_angle', 'top_letter', 'top_angle', 'bottom_letter', 'bottom_angle', 'noise_types']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)
    
    # ノイズ統計を保存
    stats_path = os.path.join(OUTPUT_DIR, 'noise_statistics.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write('ノイズ統計\n')
        f.write('='*50 + '\n\n')
        f.write(f'総画像数: {NUM_IMAGES}\n\n')
        
        for noise_type, count in sorted(noise_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / NUM_IMAGES) * 100
            f.write(f'{noise_type}: {count} ({percentage:.1f}%)\n')
        
        f.write('\n' + '='*50 + '\n')
        f.write('ノイズ設定\n')
        f.write('='*50 + '\n\n')
        
        for noise_name, config in NOISE_CONFIG.items():
            if config['enabled']:
                f.write(f'{noise_name}:\n')
                f.write(f'  確率: {config["probability"]*100:.0f}%\n')
                for key, value in config.items():
                    if key not in ['enabled', 'probability']:
                        f.write(f'  {key}: {value}\n')
                f.write('\n')
    
    print(f'\n完了！')
    print(f'生成: {len(dataset)} 枚')
    print(f'出力: {os.path.abspath(OUTPUT_DIR)}')
    print(f'ラベル: {csv_path}')
    print(f'統計: {stats_path}')
    print(f'\nノイズ統計:')
    for noise_type, count in sorted(noise_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / NUM_IMAGES) * 100
        print(f'  {noise_type}: {count} ({percentage:.1f}%)')


if __name__ == '__main__':
    main()
