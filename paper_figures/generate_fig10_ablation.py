import matplotlib.pyplot as plt
import numpy as np

configs = ['Full\n(all 8)', 'No DW-\nNOMINATE', 'No\nParty ID', 'No NOM+\nNo Party', 'Network\nonly', 'NOMINATE\nonly']
def_auc = [0.887, 0.748, 0.752, 0.897, 0.903, 0.814]
def_f1 =  [0.502, 0.452, 0.229, 0.000, 0.000, 0.438]
coal_f1 = [0.993, 0.968, 0.977, 0.663, 0.663, 1.000]

x = np.arange(len(configs))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 5))

bars1 = ax.bar(x - width, def_auc, width, label='Defection AUC', color='#2166ac', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x, def_f1, width, label='Defection F1', color='#b2182b', edgecolor='white', linewidth=0.5)
bars3 = ax.bar(x + width, coal_f1, width, label='Coalition F1', color='#4dac26', edgecolor='white', linewidth=0.5)

# Add value labels on top of bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0.05:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

ax.set_ylabel('Score', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=9)
ax.set_ylim(0, 1.12)
ax.legend(loc='upper right', frameon=True, fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('Feature Ablation: Impact on Prediction Tasks', fontsize=12, fontweight='bold')

# Horizontal reference line at full model AUC
ax.axhline(y=0.887, color='#2166ac', linestyle='--', alpha=0.3, linewidth=0.8)

plt.tight_layout()
plt.savefig('paper_figures/fig10_ablation.pdf', bbox_inches='tight')
plt.savefig('paper_figures/fig10_ablation.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 10 saved")
