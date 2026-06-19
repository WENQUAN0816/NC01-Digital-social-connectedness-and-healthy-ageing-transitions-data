"""
生成Figure 1: 中介路径图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# 颜色方案
color_exposure = '#2563EB'  # 蓝色
color_mediator = '#16A34A'  # 绿色
color_outcome = '#EA580C'   # 橙色
color_path = '#6B7280'      # 灰色

# 定义节点位置
x_exposure = 1.5
x_mediators = 5
x_outcome = 8.5
y_top = 7
y_middle = 5
y_bottom = 3

# 绘制节点
def draw_box(ax, x, y, width, height, text, color):
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='black',
                          linewidth=2, alpha=0.3)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=12, fontweight='bold', wrap=True)

# 暴露变量
draw_box(ax, x_exposure, y_middle, 2, 1.2,
         'Digital-Social\nConnectedness', color_exposure)

# 中介变量
draw_box(ax, x_mediators, y_top, 2, 1,
         'Cognitive\nFunction', color_mediator)
draw_box(ax, x_mediators, y_bottom, 2, 1,
         'Depression', color_mediator)

# 结果变量
draw_box(ax, x_outcome, y_middle, 2, 1.2,
         'Healthy\nAging', color_outcome)

# 绘制路径箭头
def draw_arrow(ax, x1, y1, x2, y2, label, color, style='->'):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style,
                           color=color, linewidth=2.5,
                           connectionstyle="arc3,rad=0")
    ax.add_patch(arrow)
    # 计算标签位置
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    ax.text(mid_x, mid_y + 0.3, label, ha='center', va='bottom',
            fontsize=10, color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor=color, alpha=0.8))

# 路径1：认知通路
# a1: Digital → Cognition
draw_arrow(ax, x_exposure + 1, y_middle + 0.3,
          x_mediators - 1, y_top - 0.2,
          r'$a_1=0.184$' + '\n(95% CI: 0.178-0.190)',
          '#16A34A')

# b1: Cognition → Healthy Aging
draw_arrow(ax, x_mediators + 1, y_top - 0.2,
          x_outcome - 1, y_middle + 0.3,
          r'$b_1=0.095$' + '\n(95% CI: 0.089-0.101)',
          '#16A34A')

# 路径2：抑郁通路
# a2: Digital → Depression
draw_arrow(ax, x_exposure + 1, y_middle - 0.3,
          x_mediators - 1, y_bottom + 0.2,
          r'$a_2=-0.106$' + '\n(95% CI: -0.112 to -0.100)',
          '#DC2626')

# b2: Depression → Healthy Aging
draw_arrow(ax, x_mediators + 1, y_bottom + 0.2,
          x_outcome - 1, y_middle - 0.3,
          r'$b_2=-0.268$' + '\n(95% CI: -0.274 to -0.262)',
          '#DC2626')

# 直接效应（虚线）
arrow_direct = FancyArrowPatch((x_exposure + 1, y_middle),
                               (x_outcome - 1, y_middle),
                               arrowstyle='->', linestyle='--',
                               color=color_path, linewidth=2,
                               alpha=0.5)
ax.add_patch(arrow_direct)
ax.text((x_exposure + x_outcome) / 2, y_middle - 0.5,
        r"Direct effect: $c'=0.019$" + '\n(95% CI: 0.013-0.025)',
        ha='center', fontsize=10, color=color_path, style='italic')

# 添加中介效应文本框
ax.text(5, 1,
        'Cognitive pathway: 36.2% mediated (95% CI: 33.1-39.5%)\n' +
        'Mental health pathway: 59.6% mediated (95% CI: 55.8-63.7%)\n' +
        'Total mediation: 95.7% (95% CI: 93.5-97.8%)',
        ha='center', va='center', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#FEF3C7',
                 edgecolor='#F59E0B', linewidth=2))

# 标题
ax.text(5, 9.2,
        'Figure 1: Mediation Pathways from Digital-Social Connectedness to Healthy Aging',
        ha='center', fontsize=14, fontweight='bold')

ax.text(5, 8.7,
        'Based on 88,456 observations from CHARLS 2011-2020',
        ha='center', fontsize=10, style='italic', color='#6B7280')

plt.tight_layout()
plt.savefig('C:/Users/Administrator/NC01-Digital-social-connectedness-and-healthy-ageing-transitions-data/manuscript/figures/figure1_mediation.pdf',
            dpi=300, bbox_inches='tight')
print("✓ Figure 1 saved: figure1_mediation.pdf")
plt.close()
