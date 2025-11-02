"""
テンプレート画像生成スクリプト
A-Fの文字を様々な角度で生成してテンプレートとして保存
"""
from PIL import Image, ImageDraw, ImageFont
import os

def generate_templates(output_dir='analyzer/templates', angles=range(0, 360, 10)):
    """
    各文字（A-F）のテンプレート画像を生成
    
    Args:
        output_dir: テンプレート保存先ディレクトリ
        angles: 生成する角度のリスト（デフォルト: 0-360度、10度刻み）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 設定（データセットと同じ）
    circle_diameter = 200  # データセット: 半径100 × 2
    font_size = 80         # データセットと同じ
    characters = ['A', 'B', 'C', 'D', 'E', 'F']
    
    # フォント読み込み
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    print(f"テンプレート生成中... ({len(characters)} 文字 × {len(angles)} 角度)")
    
    for char in characters:
        for angle in angles:
            # 白背景、黒い円と白文字の画像を作成
            img = Image.new('RGB', (circle_diameter, circle_diameter), 'white')
            draw = ImageDraw.Draw(img)
            
            # 黒い円を描画
            draw.ellipse([0, 0, circle_diameter-1, circle_diameter-1], 
                        fill='black', outline='black')
            
            # 白い文字を描画（中央配置）
            bbox = draw.textbbox((0, 0), char, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (circle_diameter - text_width) / 2 - bbox[0]
            y = (circle_diameter - text_height) / 2 - bbox[1]
            
            draw.text((x, y), char, fill='white', font=font)
            
            # 回転
            if angle != 0:
                img = img.rotate(-angle, expand=False, fillcolor='white')
            
            # 保存
            filename = f"{char}_{angle:03d}.png"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath)
        
        print(f"  ✓ {char}: {len(angles)} テンプレート生成完了")
    
    total = len(characters) * len(angles)
    print(f"\n合計 {total} 個のテンプレート画像を生成しました: {output_dir}")


if __name__ == '__main__':
    generate_templates()
