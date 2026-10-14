from config_importation import *


def split_data(df):
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def fit_train_ols(X_train, y_train):
    """statsmodels OLS on the training data (full regression table with p-values)."""
    ols = sm.OLS(y_train, sm.add_constant(X_train)).fit()
    print("--- OLS Training Summary ---")
    print(ols.summary())
    return ols


def evaluate_test(X_train, X_test, y_train, y_test):
    """Fit on the training matches, then score on matches the model has never seen."""
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    baseline = np.full(len(y_test), y_train.mean())  # 'always guess the average' model
    metrics = {
        "Test R2": r2_score(y_test, y_pred),
        "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "Test MAE": mean_absolute_error(y_test, y_pred),
        "Baseline RMSE": np.sqrt(mean_squared_error(y_test, baseline)),
    }
    return model, y_pred, metrics


def cross_validate(X, y):
    """Repeated 5-fold cross-validation: steadier than one small test set."""
    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
    r2 = cross_val_score(LinearRegression(), X, y, cv=cv, scoring="r2")
    rmse = -cross_val_score(LinearRegression(), X, y, cv=cv, scoring="neg_root_mean_squared_error")
    cv_results = {"CV mean R2": r2.mean(), "CV R2 std": r2.std(), "CV mean RMSE": rmse.mean()}
    print("\n--- Repeated 5-Fold Cross-Validation ---")
    for k, v in cv_results.items():
        print(f"{k}: {v:.4f}")
    return cv_results


def fit_final_model(df):
    """Fit on all 104 matches and build the coefficient table for the report."""
    X, y = df[FEATURES], df[TARGET]
    lr = LinearRegression().fit(X, y)
    lr_std = LinearRegression().fit((X - X.mean()) / X.std(), y)  # standardised coefficients
    ols = sm.OLS(y, sm.add_constant(X)).fit()
    conf = ols.conf_int()

    table = pd.DataFrame({
        "coefficient": lr.coef_,
        "std_coefficient": lr_std.coef_,
        "p_value": ols.pvalues[FEATURES].values,
        "CI_low": conf.loc[FEATURES, 0].values,
        "CI_high": conf.loc[FEATURES, 1].values,
    }, index=FEATURES).round(4)
    table = table.sort_values("std_coefficient", key=abs, ascending=False)
    
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_DIR / "coefficients.csv")

    print("\nFinal model (all matches):")
    print(table)
    print(f"Intercept: {lr.intercept_:.4f}")
    print(f"R2: {ols.rsquared:.3f} | Adjusted R2: {ols.rsquared_adj:.3f} | F-test p: {ols.f_pvalue:.2e}")
    return lr, ols, table


def comparison_table(y_all, y_train, y_test, y_pred):
    """Side-by-side test scores: our regression model vs a baseline that always guesses the
    average goal difference of the training matches."""
    baseline_pred = np.full(len(y_test), y_train.mean())
    value_range = y_all.max() - y_all.min()  # used to normalise the RMSE

    def scores(pred):
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        return {
            "MAE": mean_absolute_error(y_test, pred),
            "MSE": mse,
            "RMSE": rmse,
            "RMSE (Normalised)": rmse / value_range,
            "R^2": r2_score(y_test, pred),
        }

    table = pd.DataFrame({"linear_regression": scores(y_pred), "baseline": scores(baseline_pred)}).round(3)
    table.index.name = "metric"
    
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_DIR / "model_vs_baseline.csv")
    print("\nModel vs baseline (test set):")
    print(table)
    return table


if __name__ == "__main__":
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED_CSV)

    # 1. Train / test split
    X_train, X_test, y_train, y_test = split_data(df)

    # 2. Fit and inspect training OLS summary
    ols_train = fit_train_ols(X_train, y_train)

    # 3. Evaluate model predictions on the unseen test set
    model, y_pred, metrics = evaluate_test(X_train, X_test, y_train, y_test)

    # 4. Generate comparison table against the baseline
    comp_tab = comparison_table(df[TARGET], y_train, y_test, y_pred)

    # 5. Run Repeated K-Fold cross validation
    cv_res = cross_validate(df[FEATURES], df[TARGET])

    # 6. Fit final model on all 104 matches for coefficient reporting
    final_lr, final_ols, coef_table = fit_final_model(df)