"""
PulseGuard GTU Data Visualization Engine
Covers GTU Unit 7 (Visualizing data through figure, subplot and its properties,
graphs, plots in Matplotlib, advanced visualizations with Seaborn)
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional

# Set Seaborn theme aesthetics
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({
    "figure.facecolor": "#0F172A",
    "axes.facecolor": "#1E293B",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#E2E8F0",
    "xtick.color": "#94A3B8",
    "ytick.color": "#94A3B8",
    "text.color": "#F8FAFC",
    "grid.color": "#334155"
})

class GTUVisualizer:
    def __init__(self):
        pass

    def plot_matplotlib_subplots(self, df: pd.DataFrame, clt_means: Optional[list] = None) -> plt.Figure:
        """
        GTU Unit 7: Visualizing data through figure, subplot (2x2 grid) and properties:
        - Subplot 1: Categorical Bar Chart (Sentiment Value Counts)
        - Subplot 2: Histogram with KDE (Sentiment Score Dispersion)
        - Subplot 3: Scatter Plot (Comment Word Count vs Toxicity Score)
        - Subplot 4: Central Limit Theorem Bell Curve (Sampling Distribution of Means)
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("GTU Unit 7: Multi-Feature Statistical Subplots & Distributions", fontsize=16, fontweight="bold", color="#38BDF8")

        # Subplot 1: Bar Chart (Sentiment Counts)
        ax1 = axes[0, 0]
        if not df.empty and "sentiment" in df.columns:
            counts = df["sentiment"].value_counts()
            colors = ["#10B981" if x == "POSITIVE" else "#EF4444" if x == "NEGATIVE" else "#3B82F6" for x in counts.index]
            ax1.bar(counts.index, counts.values, color=colors, edgecolor="#0F172A", alpha=0.85)
            ax1.set_title("1. Sentiment Polarity Frequencies", fontweight="bold")
            ax1.set_ylabel("Count")
            for i, v in enumerate(counts.values):
                ax1.text(i, v + 0.5, str(v), ha='center', fontweight='bold', color="#F8FAFC")
        else:
            ax1.text(0.5, 0.5, "No Data Available", ha='center', va='center')

        # Subplot 2: Histogram & KDE (Unit 3 & 7)
        ax2 = axes[0, 1]
        if not df.empty and "sentiment_score" in df.columns:
            sns.histplot(df["sentiment_score"], kde=True, ax=ax2, color="#F59E0B", bins=15, alpha=0.6)
            ax2.set_title("2. Sentiment Score Continuous Distribution", fontweight="bold")
            ax2.set_xlabel("Confidence / Polarity Score")
        else:
            ax2.text(0.5, 0.5, "No Data Available", ha='center', va='center')

        # Subplot 3: Scatter Plot (Bivariate Analysis)
        ax3 = axes[1, 0]
        if not df.empty and "word_count" in df.columns and "toxicity_score" in df.columns:
            scatter_colors = ["#EF4444" if t else "#3B82F6" for t in df["is_toxic"]]
            ax3.scatter(df["word_count"], df["toxicity_score"], c=scatter_colors, alpha=0.7, edgecolors="none", s=45)
            ax3.set_title("3. Word Count vs Toxicity Score (Red=Toxic)", fontweight="bold")
            ax3.set_xlabel("Comment Length (Words)")
            ax3.set_ylabel("Toxicity Score [0, 1]")
        else:
            ax3.text(0.5, 0.5, "No Data Available", ha='center', va='center')

        # Subplot 4: Central Limit Theorem Simulation Curve (Unit 5 & 7)
        ax4 = axes[1, 1]
        if clt_means and len(clt_means) > 10:
            sns.histplot(clt_means, kde=True, ax=ax4, color="#38BDF8", bins=20, stat="density", alpha=0.6)
            ax4.set_title("4. Central Limit Theorem: Sampling Mean Normal Curve", fontweight="bold")
            ax4.set_xlabel("Sample Mean (n=30)")
            ax4.set_ylabel("Probability Density")
        else:
            # Synthetic demonstration if no stream clt
            demo_clt = np.random.normal(loc=0.5, scale=0.08, size=500)
            sns.histplot(demo_clt, kde=True, ax=ax4, color="#38BDF8", bins=20, stat="density", alpha=0.6)
            ax4.set_title("4. Central Limit Theorem: Standard Normal Convergence", fontweight="bold")
            ax4.set_xlabel("Sample Mean (n=30)")

        plt.tight_layout()
        return fig

    def plot_seaborn_correlation_heatmap(self, df: pd.DataFrame) -> plt.Figure:
        """
        GTU Unit 7: Advanced Seaborn Visualization - Correlation Heatmap:
        Examines multivariate linear relationships among numeric variables.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        numeric_cols = ["sentiment_score", "toxicity_score", "is_toxic", "word_count", "char_length"]
        available_cols = [c for c in numeric_cols if c in df.columns]

        if len(available_cols) >= 2:
            corr_matrix = df[available_cols].corr()
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".2f",
                cmap="vlag",
                ax=ax,
                cbar_kws={'label': 'Pearson Correlation'},
                linewidths=0.5,
                linecolor="#0F172A"
            )
            ax.set_title("Seaborn Heatmap: Multivariate Feature Correlation (Unit 7)", fontsize=13, fontweight="bold", color="#38BDF8")
        else:
            ax.text(0.5, 0.5, "Need numeric features for heatmap", ha='center', va='center', color="#F8FAFC")

        plt.tight_layout()
        return fig

    def plot_platform_comparison_barplot(self, platform_df: pd.DataFrame) -> plt.Figure:
        """Plot comparative negative velocity across YouTube, Twitter, Reddit, and Telegram."""
        fig, ax = plt.subplots(figsize=(9, 4.5))
        if not platform_df.empty and "platform" in platform_df.columns:
            sns.barplot(
                data=platform_df,
                x="platform",
                y="negative_ratio",
                hue="platform",
                palette="rocket",
                ax=ax,
                legend=False
            )
            ax.axhline(0.25, color="#F59E0B", linestyle="--", label="Warning Threshold (25%)")
            ax.axhline(0.40, color="#EF4444", linestyle="--", label="Crisis Threshold (40%)")
            ax.set_title("Comparative Platform Negative Velocity (Crisis Risk)", fontsize=13, fontweight="bold", color="#38BDF8")
            ax.set_ylabel("Negative Ratio")
            ax.set_ylim(0, 1.0)
            ax.legend()
        plt.tight_layout()
        return fig
