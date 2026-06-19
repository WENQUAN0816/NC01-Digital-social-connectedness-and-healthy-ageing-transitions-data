"""
将JPG转换为PDF
"""
from PIL import Image

# 打开图片
img = Image.open('C:/Users/Administrator/NC01-Digital-social-connectedness-and-healthy-ageing-transitions-data/manuscript/figures/figure1_mediation.jpg')

# 转换为RGB（如果是RGBA）
if img.mode == 'RGBA':
    img = img.convert('RGB')

# 保存为PDF
img.save('C:/Users/Administrator/NC01-Digital-social-connectedness-and-healthy-ageing-transitions-data/manuscript/figures/figure1_mediation.pdf',
         'PDF', resolution=300.0, quality=100)

print("✓ 机制图已转换为PDF")
