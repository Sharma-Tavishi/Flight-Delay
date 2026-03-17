"""
Poster chart generator — run from the FlightDelay root directory:
    python notebooks/generate_poster_charts.py
Outputs go to assets/
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── 1. PIPELINE DIAGRAM ────────────────────────────────────────────────────────
def make_pipeline():
    fig, ax = plt.subplots(figsize=(22, 9))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 9)
    ax.axis('off')

    ax.text(11, 8.4, 'System Architecture', ha='center', fontsize=24,
            fontweight='bold', color='#1e293b')
    ax.text(11, 7.9, 'End-to-end flight delay prediction pipeline', ha='center',
            fontsize=13, color='#64748b')

    steps = [
        (2.2,  "User Input",      "Natural language\nflight query",      "NL",  '#eff6ff', '#3b82f6', '#1d4ed8'),
        (6.2,  "Claude API",      "Extracts structured\nflight details",  "LLM", '#f5f3ff', '#8b5cf6', '#6d28d9'),
        (10.2, "Feature Engine",  "Weather + route\nhistory enrichment",  "FE",  '#f0fdf4', '#22c55e', '#15803d'),
        (14.2, "LightGBM Model",  "Predicts delay class\n& minutes",      "ML",  '#fff7ed', '#f97316', '#c2410c'),
        (18.2, "Result",          "Delay prediction\n+ explanation",      "OUT", '#fef2f2', '#ef4444', '#b91c1c'),
    ]

    box_w, box_h, box_y = 3.2, 3.8, 3.2

    for x, label, sublabel, icon, bg, border, dark in steps:
        shadow = FancyBboxPatch((x - box_w/2 + 0.08, box_y - 0.08), box_w, box_h,
                                 boxstyle="round,pad=0.15", facecolor='#e2e8f0',
                                 edgecolor='none', zorder=1)
        ax.add_patch(shadow)
        box = FancyBboxPatch((x - box_w/2, box_y), box_w, box_h,
                              boxstyle="round,pad=0.15",
                              facecolor=bg, edgecolor=border, linewidth=2, zorder=2)
        ax.add_patch(box)
        circle = plt.Circle((x, box_y + box_h - 0.7), 0.45, color=dark, zorder=3)
        ax.add_patch(circle)
        ax.text(x, box_y + box_h - 0.7, icon, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white', zorder=4)
        ax.text(x, box_y + box_h/2 + 0.1, label, ha='center', va='center',
                fontsize=13, fontweight='bold', color='#1e293b', zorder=3)
        ax.text(x, box_y + 0.6, sublabel, ha='center', va='center',
                fontsize=10, color='#64748b', linespacing=1.5, zorder=3)

    for x in [3.8, 7.8, 11.8, 15.8]:
        ax.annotate('', xy=(x + 0.3, box_y + box_h/2),
                    xytext=(x - 0.3, box_y + box_h/2),
                    arrowprops=dict(arrowstyle='->', color='#94a3b8',
                                    lw=2, mutation_scale=20), zorder=5)

    for i, (x, *_) in enumerate(steps):
        ax.text(x - box_w/2 + 0.25, box_y + box_h - 0.2, f'Step {i+1}',
                fontsize=8, color='#94a3b8', fontweight='bold', zorder=4)

    ax.text(2.2, 2.7, 'Input: "Will my AA flight\nfrom ORD to JFK be delayed?"',
            ha='center', fontsize=9, color='#3b82f6', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#eff6ff',
                      edgecolor='#3b82f6', alpha=0.8))
    ax.text(18.2, 2.7, 'Output: "Minor Delay\n+24 min estimated"',
            ha='center', fontsize=9, color='#b91c1c', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef2f2',
                      edgecolor='#ef4444', alpha=0.8))

    plt.tight_layout(pad=0.5)
    plt.savefig('assets/pipeline_diagram.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: assets/pipeline_diagram.png")


# ── 2. CLASS DISTRIBUTION ──────────────────────────────────────────────────────
def make_class_distribution():
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.35)

    # Donut
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('white')
    sizes   = [81.7, 12.5, 5.8]
    colors  = ['#16a34a', '#f59e0b', '#ef4444']
    labels  = ['On-time', 'Minor Delay\n(16–59 min)', 'Major Delay\n(60+ min)']
    wedges, _ = ax1.pie(sizes, colors=colors, explode=(0.02, 0.04, 0.06),
                         startangle=140,
                         wedgeprops=dict(width=0.55, linewidth=2, edgecolor='white'))
    ax1.text(0, 0.08, '987K', ha='center', va='center',
             fontsize=22, fontweight='bold', color='#1e293b')
    ax1.text(0, -0.22, 'flights', ha='center', va='center', fontsize=11, color='#64748b')
    ax1.set_title('Flight Delay Class Distribution', fontsize=15,
                   fontweight='bold', color='#1e293b', pad=15)
    legend_items = [mpatches.Patch(color=c, label=f'{l}  —  {s:.1f}%')
                    for c, l, s in zip(colors, labels, sizes)]
    ax1.legend(handles=legend_items, loc='lower center', bbox_to_anchor=(0.5, -0.14),
               fontsize=11, frameon=True, framealpha=0.9, edgecolor='#e2e8f0')

    # Bar chart
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('#f8fafc')
    for spine in ['top', 'right']:
        ax2.spines[spine].set_visible(False)
    ax2.spines['left'].set_color('#e2e8f0')
    ax2.spines['bottom'].set_color('#e2e8f0')

    categories = ['On-time\n(<15 min)', 'Minor Delay\n(16–59 min)', 'Major Delay\n(60+ min)']
    values     = [81.7, 12.5, 5.8]
    y_pos      = [2, 1, 0]
    bars = ax2.barh(y_pos, values, color=['#16a34a', '#f59e0b', '#ef4444'],
                    height=0.55, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}%', va='center', fontsize=13, fontweight='bold', color='#1e293b')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(categories, fontsize=11, color='#374151')
    ax2.set_xlabel('Percentage of Flights (%)', fontsize=11, color='#64748b')
    ax2.set_xlim(0, 100)
    ax2.tick_params(colors='#94a3b8')
    ax2.set_title('Class Breakdown', fontsize=15, fontweight='bold', color='#1e293b', pad=12)
    ax2.grid(axis='x', alpha=0.4, color='#e2e8f0')
    for y, count in zip(y_pos, ['~806K flights', '~123K flights', '~57K flights']):
        ax2.text(1, y - 0.22, count, fontsize=9, color='#94a3b8', style='italic')

    fig.suptitle('Dataset Class Imbalance — Key Modeling Challenge',
                 fontsize=13, color='#64748b', y=0.02, style='italic')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('assets/class_distribution.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: assets/class_distribution.png")


# ── 3. DATA STATS INFOGRAPHIC ──────────────────────────────────────────────────
def make_data_stats():
    fig, ax = plt.subplots(figsize=(20, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.axis('off')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 7)

    ax.text(10, 6.4, 'Dataset & Model at a Glance', ha='center', fontsize=22,
            fontweight='bold', color='#1e293b')
    ax.plot([2, 18], [6.0, 6.0], color='#e2e8f0', lw=1.5)

    stats = [
        ("987,000+", "Flights\nTrained On",      '#3b82f6', '#dbeafe', '#1d4ed8'),
        ("47",       "Months of Data\n2022–2025", '#8b5cf6', '#ede9fe', '#6d28d9'),
        ("51",       "US Airports\nCovered",      '#10b981', '#d1fae5', '#047857'),
        ("14",       "Predictive\nFeatures",      '#f59e0b', '#fef3c7', '#b45309'),
        ("9",        "Airlines\nIncluded",        '#ef4444', '#fee2e2', '#b91c1c'),
        ("3",        "Delay\nClasses",            '#6366f1', '#e0e7ff', '#4338ca'),
    ]

    w, h = 2.8, 3.8
    gap = (20 - len(stats) * w) / (len(stats) + 1)

    for i, (val, label, accent, bg, dark) in enumerate(stats):
        x = gap + i * (w + gap)
        y = 1.0
        shadow = FancyBboxPatch((x + 0.06, y - 0.06), w, h,
                                 boxstyle="round,pad=0.2", facecolor='#f1f5f9',
                                 edgecolor='none', zorder=1)
        ax.add_patch(shadow)
        card = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                               facecolor=bg, edgecolor=accent, linewidth=1.8, zorder=2)
        ax.add_patch(card)
        bar = FancyBboxPatch((x, y + h - 0.35), w, 0.35,
                              boxstyle="round,pad=0.0", facecolor=accent,
                              edgecolor='none', zorder=3)
        ax.add_patch(bar)
        ax.text(x + w/2, y + h*0.52, val, ha='center', va='center',
                fontsize=28, fontweight='bold', color=dark, zorder=4)
        ax.text(x + w/2, y + h*0.2, label, ha='center', va='center',
                fontsize=10.5, color='#374151', linespacing=1.5, zorder=4)

    plt.tight_layout(pad=0.3)
    plt.savefig('assets/data_stats_infographic.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: assets/data_stats_infographic.png")


if __name__ == '__main__':
    make_pipeline()
    make_class_distribution()
    make_data_stats()
    print("All charts generated.")
