"""
データセット生成スクリプト（完全に作り直したバージョン）
シンプルで確実な実装
"""

import os
import csv
import random
from PIL import Image, ImageDraw, ImageFont
import math


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
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset')
RANDOM_SEED = 42


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


def generate_single_image(filepath, heart_angle, top_letter, top_angle, bottom_letter, bottom_angle):
    """1枚の画像を生成"""
    # 基本画像
    img = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    
    # ハート描画
    draw_heart(draw, HEART_X, HEART_Y, HEART_SIZE, heart_angle, HEART_COLOR)
    
    # 右上の円と文字
    top_img, top_x, top_y = draw_circle_with_letter(
        TOP_X, TOP_Y, CIRCLE_RADIUS, top_letter, top_angle, CIRCLE_COLOR
    )
    img.paste(top_img, (top_x, top_y), top_img)
    
    # 右下の円と文字
    bottom_img, bottom_x, bottom_y = draw_circle_with_letter(
        BOTTOM_X, BOTTOM_Y, CIRCLE_RADIUS, bottom_letter, bottom_angle, CIRCLE_COLOR
    )
    img.paste(bottom_img, (bottom_x, bottom_y), bottom_img)
    
    # 保存
    img.save(filepath)


def main():
    """メイン処理"""
    # ディレクトリ作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # シード設定
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
    
    # データを生成
    dataset = []
    
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
            generate_single_image(filepath, heart_angle, top_letter, top_angle, bottom_letter, bottom_angle)
            
            # データ記録
            dataset.append({
                'filename': filename,
                'heart_angle': heart_angle,
                'top_letter': top_letter,
                'top_angle': top_angle,
                'bottom_letter': bottom_letter,
                'bottom_angle': bottom_angle
            })
            
        except Exception as e:
            print(f'ERROR generating {filename}: {e}')
            import traceback
            traceback.print_exc()
    
    # CSVに保存
    csv_path = os.path.join(OUTPUT_DIR, 'labels.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['filename', 'heart_angle', 'top_letter', 'top_angle', 'bottom_letter', 'bottom_angle']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)
    
    print(f'\nCompleted!')
    print(f'Generated: {len(dataset)} images')
    print(f'Output: {OUTPUT_DIR}')
    print(f'Labels: {csv_path}')


if __name__ == '__main__':
    main()
