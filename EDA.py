# exploratory data analysis: descriptive statistics, correlations, multicollinearity (VIF).

from config_importation import *


def describe(df):
    table = df[FEATURES + [TARGET]].describe().T.round(3)
    table.to_csv(TABLE_DIR / "descriptive_statistics.csv")
    print(table)


def plot_target_distribution(df):
    y = df[TARGET]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y, bins=range(int(y.min()) - 1, int(y.max()) + 2), edgecolor="black", align="left")
    ax.set_xlabel("Goal difference (team_a - team_b)")
    ax.set_ylabel("Number of matches")
    ax.set_title("Distribution of goal difference")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "target_distribution.png", dpi=150)
    plt.close(fig)


def plot_correlation(df):
    corr = df[FEATURES + [TARGET]].corr()
    corr.round(3).to_csv(TABLE_DIR / "correlation_matrix.csv")
    print("\nCorrelation with goal_difference:")
    print(corr[TARGET].drop(TARGET).sort_values(ascending=False).round(3))

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correlation_matrix.png", dpi=150)
    plt.close(fig)


def compute_vif(df):
    """VIF above roughly 5-10 means a variable is largely explained by the others."""
    X = sm.add_constant(df[FEATURES])
    vif = pd.DataFrame({
        "variable": FEATURES,
        "VIF": [variance_inflation_factor(X.values, i + 1) for i in range(len(FEATURES))],
    }).round(2)
    vif.to_csv(TABLE_DIR / "vif.csv", index=False)
    print("\nVariance Inflation Factors:")
    print(vif.to_string(index=False))
    return vif


def run_eda(df):
    # Automatically create the output subfolders inside Assignment3_Data_Science_Report
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    describe(df)
    plot_target_distribution(df)
    plot_correlation(df)
    return compute_vif(df)


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_CSV)

    # Running the full exploratory analysis
    run_eda(df)