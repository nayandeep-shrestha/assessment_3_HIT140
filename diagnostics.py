from config_importation import *
from modelling import split_data


def plot_diagnostics(y_test, y_pred, ols_full):
    residuals = y_test - y_pred
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].scatter(y_test, y_pred, alpha=0.7)
    lims = [min(y_test.min(), y_pred.min()) - 0.5, max(y_test.max(), y_pred.max()) + 0.5]
    axes[0, 0].plot(lims, lims, "r--")
    axes[0, 0].set_xlabel("Actual goal difference")
    axes[0, 0].set_ylabel("Predicted goal difference")
    axes[0, 0].set_title("Predicted vs Actual (test set)")

    axes[0, 1].scatter(y_pred, residuals, alpha=0.7)
    axes[0, 1].axhline(0, color="red", linestyle="--")
    axes[0, 1].set_xlabel("Predicted goal difference")
    axes[0, 1].set_ylabel("Residual")
    axes[0, 1].set_title("Residuals vs Predicted (test set)")

    axes[1, 0].hist(ols_full.resid, bins=15, edgecolor="black")
    axes[1, 0].set_xlabel("Residual")
    axes[1, 0].set_title("Residual distribution (all matches)")

    sm.qqplot(ols_full.resid, line="45", fit=True, ax=axes[1, 1])
    axes[1, 1].set_title("Normal Q-Q plot of residuals")

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "diagnostics.png", dpi=150)
    plt.close(fig)
    print(f"Saved diagnostic plots to {FIG_DIR / 'diagnostics.png'}")


def assumption_tests(df, ols_full):
    X = sm.add_constant(df[FEATURES])
    _, shapiro_p = stats.shapiro(ols_full.resid)
    _, bp_p, _, _ = het_breuschpagan(ols_full.resid, X)
    dw = sm.stats.durbin_watson(ols_full.resid)

    print("\nAssumption checks:")
    print(f"Normality of residuals (Shapiro-Wilk): p = {shapiro_p:.3f} "
          f"-> {'looks normal' if shapiro_p > 0.05 else 'departs from normal'}")
    print(f"Constant variance (Breusch-Pagan):    p = {bp_p:.3f} "
          f"-> {'no evidence of heteroscedasticity' if bp_p > 0.05 else 'heteroscedasticity detected'}")
    print(f"Durbin-Watson: {dw:.2f} (about 2 = no autocorrelation)")
    return {"Shapiro p": shapiro_p, "Breusch-Pagan p": bp_p, "Durbin-Watson": dw}


def largest_errors(df, ols_full, n=8):
    out = df[["match_id", "team_a", "team_b", "stage", TARGET]].copy()
    out["predicted"] = ols_full.fittedvalues.round(2)
    out["error"] = (out[TARGET] - out["predicted"]).round(2)
    out = out.reindex(out["error"].abs().sort_values(ascending=False).index).head(n)
    
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLE_DIR / "largest_errors.csv", index=False)
    print("\nMatches the model got most wrong:")
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset
    df = pd.read_csv(PROCESSED_CSV)

    # 2. Test-set predictions for out-of-sample diagnostic plots
    X_train, X_test, y_train, y_test = split_data(df)
    lr = LinearRegression().fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    # 3. Fit full OLS model for assumption testing and residual diagnostics
    X_full = sm.add_constant(df[FEATURES])
    ols_full = sm.OLS(df[TARGET], X_full).fit()

    # 4. Generate diagnostic figures, statistical assumption tests, and worst residuals
    plot_diagnostics(y_test, y_pred, ols_full)
    assumption_tests(df, ols_full)
    largest_errors(df, ols_full)