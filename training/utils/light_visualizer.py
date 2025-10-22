import os

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def visualize_statistics(directory: str, stats: "TrainingStatistics"):
    os.makedirs(directory, exist_ok=True)
    
    sns.set_theme(style="whitegrid")

    axes = None
    if len(stats.val_history_steps) != 0:
        axes = sns.lineplot(x=stats.val_history_steps, y=stats.val_loss_history, label=f'Validation Loss (l={stats.val_loss_history[-1]:.3f}/b={min(stats.val_loss_history):.3f})')
        val_loss_min = np.argmin(stats.val_loss_history)
        axes.plot(stats.val_history_steps[val_loss_min], stats.val_loss_history[val_loss_min], 'o', markersize=3)
    if len(stats.train_history_steps) != 0:
        axes = sns.lineplot(x=stats.train_history_steps, y=stats.train_loss_history, label=f'Training Loss (l={stats.train_loss_history[-1]:.3f}/b={min(stats.train_loss_history):.3f})', ax=axes)
        train_loss_min = np.argmin(stats.train_loss_history)
        axes.plot(stats.train_history_steps[train_loss_min], stats.train_loss_history[train_loss_min], 'o', markersize=3)
    if axes is not None:
        axes.set_xlabel('Steps')
        axes.set_ylabel('Loss / Training Loss (log scale)')
        axes.set_yscale('log')
        axes.set_ylim(top=5)
        axes.set_title('Validation Loss and Training Loss over Steps')
        axes.grid(True)
        plt.savefig(os.path.join(directory, 'loss_curve.png'), dpi=500)
        plt.clf()

    if len(stats.lr_history_steps) != 0:
        axes = sns.lineplot(x=stats.lr_history_steps, y=stats.lr_history, label=f'Learning Rate (l={stats.lr_history[-1]:.3}/h={max(stats.lr_history):.3})')
        axes.set_xlabel('Steps')
        axes.set_ylabel('Learning Rate')
        axes.set_title('Learning Rate over Steps')
        axes.grid(True)
        plt.savefig(os.path.join(directory, 'lr_curve.png'), dpi=500)
        plt.clf()

    if len(stats.sparsity_history_steps) != 0:
        axes = sns.lineplot(x=stats.sparsity_history_steps, y=stats.sparsity_history, label=f'Sparsity Ratio (l={stats.sparsity_history[-1]:.3f})')
        axes.set_xlabel('Steps')
        axes.set_ylabel('Sparsity Ratio')
        axes.set_title('Sparsity Ratio over Steps')
        axes.grid(True)
        plt.savefig(os.path.join(directory, 'sparsity_curve.png'), dpi=500)
        plt.clf()

    if len(stats.val_history_steps) != 0 and len(stats.val_ppl_history) != 0:
        axes = sns.lineplot(x=stats.val_history_steps, y=stats.val_ppl_history, label=f'Validation Perplexity (l={stats.val_ppl_history[-1]:.3f}/b={min(stats.val_ppl_history):.3f})')
        val_ppl_min = np.argmin(stats.val_ppl_history)
        axes.plot(stats.val_history_steps[val_ppl_min], stats.val_ppl_history[val_ppl_min], 'o', markersize=3)

        axes.set_xlabel('Steps')
        axes.set_ylabel('Perplexity (log scale)')
        axes.set_yscale('log')
        axes.set_ylim(top=16)
        axes.set_title('Perplexity over Steps')
        axes.grid(True)
        plt.savefig(os.path.join(directory, 'ppl_curve.png'), dpi=500)
        plt.clf()

    for step, stat in zip(stats.real_sparsity_history_steps, stats.real_sparsity_history):
        x = []
        weights = []
        hue = []
        hints = []

        for i in range(3):
            x.append(['weight', 'input', 'output'][i])
            weights.append(0)
            hue.append("zero")
            hints.append('------------------' + ['weight', 'input', 'output'][i] + '------------------')
            for name, value in stat.items():
                x.extend([name + ['_w', '_i', '_o'][i]] * 3)
                weights.extend([value[i], value[i+3], value[i+6]])
                hue.extend(["positive", "zero", "negative"])
                hints.append(f"+{value[i]:.3f} | {value[i + 3]:.3f} | -{value[i + 6]:.3f}")

        plt.figure(figsize=(2 + len(x) // 8, 6))

        sns.histplot(x=x, hue=hue, weights=weights,
            multiple="stack",
            palette="light:m_r",
            edgecolor=".3",
            linewidth=.5,)
        
        zero_patch = plt.gca().patches[0]
        bar_start, bar_width = zero_patch.get_x(), zero_patch.get_width()

        for layer_index, hint in enumerate(hints):
            plt.text(
                x=bar_start + layer_index * bar_width + bar_width / 2, y = 0.05,
                s=str(hint),
                ha='center', va='bottom',
                rotation=90
            )

        plt.xlim(-0.65, max(20.0, len(set(x))) - 0.35)
        plt.xticks(rotation=90)
        plt.xlabel('Layer')
        plt.ylabel('Proportion')
        plt.title(f'Sparsity Distribution at Step {step}')
        plt.legend(title='Activation Type', labels=['Positive', 'Zero', 'Negative'])
        plt.tight_layout()
        plt.savefig(os.path.join(directory, f'sparsity_distribution_step_{step}.png'), dpi=150)
        plt.clf()

    for step, stat in zip(stats.threshold_history_steps, stats.threshold_history):
        x = list(stat.keys())

        thresholds_min = [v[0] for v in stat.values()]
        thresholds_mean = [v[1] for v in stat.values()]
        thresholds_max = [v[2] for v in stat.values()]

        x_indices = np.arange(len(x))
        width = 0.25
        plt.figure(figsize=(2 + len(x) // 4, 6))
        plt.bar(x_indices - width, thresholds_min, width=width, label='Min Threshold', color='tab:blue', alpha=0.7)
        plt.bar(x_indices, thresholds_mean, width=width, label='Mean Threshold', color='tab:orange', alpha=0.7)
        plt.bar(x_indices + width, thresholds_max, width=width, label='Max Threshold', color='tab:green', alpha=0.7)
        plt.xticks(ticks=x_indices, labels=x, rotation=90)
        plt.xlabel('Layer')
        plt.ylabel('Threshold Value')
        plt.title(f'Threshold Statistics at Step {step}')
        plt.legend()
        plt.tight_layout()
        plt.grid(axis='x')
        plt.savefig(os.path.join(directory, f'threshold_statistics_step_{step}.png'), dpi=150)
        plt.clf()
