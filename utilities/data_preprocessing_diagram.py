import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Get the directory where this script resides
output_dir = os.path.dirname(os.path.abspath(__file__))

# -------------------------
# Figure
# -------------------------
fig, ax = plt.subplots(figsize=(12, 3))
ax.set_xlim(0, 100)
ax.set_ylim(0, 1)
ax.axis('off')

# -------------------------
# Original Features Box
# -------------------------
orig_text = 'Original Features\n78'
orig_x, orig_y = 0, 0.55
orig_w, orig_h = 10, 0.25
ax.add_patch(Rectangle((orig_x, orig_y), orig_w, orig_h, facecolor='#a6cee3', edgecolor='black'))
ax.text(orig_x + orig_w/2, orig_y + orig_h/2, orig_text, ha='center', va='center', fontsize=10)

# -------------------------
# Removed Features Box
# -------------------------
removed_text = "Removed Features\nCleaning + LGBM\n43 features"
removed_x, removed_y = 15, 0.55
removed_w, removed_h = 15, 0.3
ax.add_patch(Rectangle((removed_x, removed_y), removed_w, removed_h, facecolor='#ff7f00', edgecolor='black'))
ax.text(removed_x + removed_w/2, removed_y + removed_h/2, removed_text, ha='center', va='center', fontsize=9, wrap=True)

# -------------------------
# Selected Features Box
# -------------------------
sel_text = 'Selected Features\n35'
sel_x, sel_y = 35, 0.55
sel_w, sel_h = 10, 0.25
ax.add_patch(Rectangle((sel_x, sel_y), sel_w, sel_h, facecolor='#1f78b4', edgecolor='black'))
ax.text(sel_x + sel_w/2, sel_y + sel_h/2, sel_text, ha='center', va='center', fontsize=10)

# -------------------------
# Original Labels Box
# -------------------------
orig_label_text = 'Original Labels\n13 classes'
orig_label_x, orig_label_y = 0, 0.1
orig_label_w, orig_label_h = 15, 0.25
ax.add_patch(Rectangle((orig_label_x, orig_label_y), orig_label_w, orig_label_h, facecolor='#b2df8a', edgecolor='black'))
ax.text(orig_label_x + orig_label_w/2, orig_label_y + orig_label_h/2, orig_label_text, ha='center', va='center', fontsize=10)

# -------------------------
# Binary Labels Box
# -------------------------
bin_label_text = 'Binary Labels\ntrusted / untrusted'
bin_label_x, bin_label_y = 35, 0.1
bin_label_w, bin_label_h = 15, 0.25
ax.add_patch(Rectangle((bin_label_x, bin_label_y), bin_label_w, bin_label_h, facecolor='#33a02c', edgecolor='black'))
ax.text(bin_label_x + bin_label_w/2, bin_label_y + bin_label_h/2, bin_label_text, ha='center', va='center', fontsize=10)

# -------------------------
# Arrows
# -------------------------
# Features flow
ax.annotate('', xy=(orig_x + orig_w, orig_y + orig_h/2), xytext=(removed_x, removed_y + removed_h/2),
            arrowprops=dict(facecolor='black', arrowstyle='<-'))
ax.annotate('', xy=(removed_x + removed_w, removed_y + removed_h/2), xytext=(sel_x, sel_y + sel_h/2),
            arrowprops=dict(facecolor='black', arrowstyle='<-'))

# Labels flow
ax.annotate('', xy=(orig_label_x + orig_label_w, orig_label_y + orig_label_h/2),
            xytext=(bin_label_x, bin_label_y + bin_label_h/2),
            arrowprops=dict(facecolor='black', arrowstyle='<-'))

# -------------------------
# Title
# -------------------------
plt.title('CIC-IDS-2017 Feature & Label Transformation\n(RandomForestRegressor + Binary Label Mapping)', fontsize=12, loc='left')

# Show figure
plt.show()

# Save figure to same folder as this script
output_file = os.path.join(output_dir, 'data_preprocessing_diagram.png')
fig.savefig(output_file, bbox_inches='tight')
print(f"[INFO] Figure saved to: {output_file}")
