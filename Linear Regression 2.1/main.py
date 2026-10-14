from config_importation import *
import download_data
import data_loading
import data_cleaning
import feature_engineering
import EDA
import modelling
import diagnostics


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 1. Get and load the raw data ===")
    download_data.download_history()
    raw_matches = data_loading.load_wc_matches()
    history = data_loading.load_history()

    print("\n=== 2. Clean the raw match list ===")
    matches = data_cleaning.clean_matches(raw_matches)
    data_cleaning.check_against_history(matches, history)

    print("\n=== 3. Calculate the 8 explanatory variables ===")
    cutoff = matches["date"].min()                      # day the tournament started
    teams = sorted(set(matches["team_a"]) | set(matches["team_b"]))
    assert len(teams) == N_TEAMS, f"Expected {N_TEAMS} teams, found {len(teams)}"
    team_table = feature_engineering.build_team_table(history, teams, cutoff)
    team_table.round(3).to_csv(TABLE_DIR / "team_values.csv")
    df = feature_engineering.build_features(matches, team_table)
    feature_engineering.save_dataset(df)

    print("\n=== 4. Exploratory analysis ===")
    EDA.run_eda(df)

    print("\n=== 5. Modelling ===")
    X_train, X_test, y_train, y_test = modelling.split_data(df)
    print(f"Training matches: {len(X_train)} | Test matches: {len(X_test)}")
    modelling.fit_train_ols(X_train, y_train)
    _, y_pred, test_metrics = modelling.evaluate_test(X_train, X_test, y_train, y_test)
    modelling.comparison_table(df[TARGET], y_train, y_test, y_pred)
    cv_metrics = modelling.cross_validate(df[FEATURES], df[TARGET])
    _, ols_full, _ = modelling.fit_final_model(df)

    print("\n=== 6. Diagnostics ===")
    diagnostics.plot_diagnostics(y_test, y_pred, ols_full)
    assumptions = diagnostics.assumption_tests(df, ols_full)
    diagnostics.largest_errors(df, ols_full)

    metrics = {**test_metrics, **cv_metrics,
               "Full-data R2": ols_full.rsquared, "Full-data adj R2": ols_full.rsquared_adj, **assumptions}
    pd.Series(metrics, name="value").round(3).to_csv(TABLE_DIR / "metrics.csv")

    print("\n=== Test and cross-validation performance ===")
    for name, value in {**test_metrics, **cv_metrics}.items():
        print(f"{name:15s}: {value:.3f}")
    print(f"\nFigures saved in {FIG_DIR}\nTables saved in {TABLE_DIR}")


if __name__ == "__main__":
    main()