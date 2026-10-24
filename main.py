import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# LIBRARY DEMAND ANALYSIS
# Topics: Time Series Analysis + Machine Learning
# --------------------------------------------------

CSV_FILE = "library_data.csv"


def load_data():
    """Load and validate the CSV dataset."""
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"ERROR: {CSV_FILE} was not found.")
        print("Keep library_data.csv in the same folder as main.py.")
        return None

    required_columns = ["Month", "Books_Issued"]
    for column in required_columns:
        if column not in df.columns:
            print(f"ERROR: CSV must contain the column '{column}'.")
            return None

    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    df["Books_Issued"] = pd.to_numeric(
        df["Books_Issued"], errors="coerce"
    )

    df = df.dropna(subset=["Month", "Books_Issued"])
    df = df.sort_values("Month")
    df = df.drop_duplicates(subset=["Month"], keep="last")
    df["Books_Issued"] = df["Books_Issued"].clip(lower=0)

    if len(df) < 12:
        print("Please provide at least 12 months of data.")
        return None

    return df.reset_index(drop=True)


def show_summary(df):
    """Print basic statistics about the borrowing data."""
    print("\n========== DATA SUMMARY ==========")
    print("Number of months:", len(df))
    print("First month:", df["Month"].min().strftime("%B %Y"))
    print("Last month:", df["Month"].max().strftime("%B %Y"))
    print("Total books issued:", int(df["Books_Issued"].sum()))
    print("Average per month:", round(df["Books_Issued"].mean(), 2))
    print("Lowest monthly count:", int(df["Books_Issued"].min()))
    print("Highest monthly count:", int(df["Books_Issued"].max()))

    busiest = df.loc[df["Books_Issued"].idxmax()]
    quietest = df.loc[df["Books_Issued"].idxmin()]
    print("Busiest month:", busiest["Month"].strftime("%B %Y"))
    print("Quietest month:", quietest["Month"].strftime("%B %Y"))

    print("\nFirst 10 records:")
    print(df.head(10).to_string(index=False))


# --------------------------------------------------
# TOPIC 1: TIME SERIES ANALYSIS
# --------------------------------------------------

def time_series_analysis(df):
    print("\n========== TIME SERIES ANALYSIS ==========")

    # Ensure each month is present. Missing months are filled with zero.
    monthly = df.set_index("Month")["Books_Issued"].asfreq("MS")
    monthly = monthly.fillna(0)

    # Three-month moving average smooths short-term changes.
    monthly_average = monthly.rolling(window=3).mean()

    print("\nMonthly borrowing data:")
    print(monthly.to_string())

    print("\nThree-month moving average (latest values):")
    print(monthly_average.tail(12).round(2).to_string())

    # Graph 1: original time series and moving average
    plt.figure(figsize=(11, 5))
    plt.plot(monthly.index, monthly.values, marker="o",
             label="Monthly books issued")
    plt.plot(monthly_average.index, monthly_average.values,
             linewidth=2, label="3-month moving average")
    plt.title("Library Borrowing Trend Over Time")
    plt.xlabel("Month")
    plt.ylabel("Books issued")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("time_series_trend.png", dpi=150)
    plt.show()

    # Graph 2: average borrowing by calendar month
    df_for_season = df.copy()
    df_for_season["Calendar_Month"] = df_for_season["Month"].dt.month
    month_average = df_for_season.groupby("Calendar_Month")[
        "Books_Issued"
    ].mean()

    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

    plt.figure(figsize=(10, 5))
    plt.bar(month_names, [month_average.get(i, 0) for i in range(1, 13)])
    plt.title("Average Book Borrowing by Calendar Month")
    plt.xlabel("Month")
    plt.ylabel("Average books issued")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("monthly_seasonality.png", dpi=150)
    plt.show()

    print("\nTime Series Analysis completed.")
    return monthly


# --------------------------------------------------
# TOPIC 2: MACHINE LEARNING
# --------------------------------------------------

def make_features(df):
    """Create simple time-based features for the ML model."""
    result = df.copy()
    result["Month_Number"] = range(len(result))
    result["Calendar_Month"] = result["Month"].dt.month
    result["Year"] = result["Month"].dt.year
    result["Previous_Month"] = result["Books_Issued"].shift(1)
    result["Previous_2_Months"] = result["Books_Issued"].shift(2)
    result["Previous_3_Months"] = result["Books_Issued"].shift(3)
    return result.dropna().reset_index(drop=True)


def machine_learning_prediction(df):
    print("\n========== MACHINE LEARNING ==========")

    featured = make_features(df)

    feature_columns = [
        "Month_Number", "Calendar_Month", "Year",
        "Previous_Month", "Previous_2_Months", "Previous_3_Months"
    ]

    X = featured[feature_columns]
    y = featured["Books_Issued"]

    # Time-based split: older records train the model,
    # newer records test it. Do not shuffle time series data.
    split = int(len(featured) * 0.8)
    if split < 5 or len(featured) - split < 2:
        print("Not enough records for a reliable train/test split.")
        return

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=42, max_depth=5
        )
    }

    best_model = None
    best_name = ""
    best_mae = float("inf")

    print("\nMODEL COMPARISON")
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions) if len(y_test) >= 2 else 0

        print(f"\n{name}")
        print("MAE :", round(mae, 2))
        print("RMSE:", round(rmse, 2))
        print("R2  :", round(r2, 3))

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_name = name

    print("\nSelected model based on test MAE:", best_name)

    # Forecast the next 6 months one month at a time.
    history = df[["Month", "Books_Issued"]].copy()
    future_rows = []

    for step in range(6):
        next_date = history["Month"].iloc[-1] + pd.offsets.MonthBegin(1)
        month_number = len(history)
        previous_1 = float(history["Books_Issued"].iloc[-1])
        previous_2 = float(history["Books_Issued"].iloc[-2])
        previous_3 = float(history["Books_Issued"].iloc[-3])

        next_features = pd.DataFrame([{
            "Month_Number": month_number,
            "Calendar_Month": next_date.month,
            "Year": next_date.year,
            "Previous_Month": previous_1,
            "Previous_2_Months": previous_2,
            "Previous_3_Months": previous_3
        }])[feature_columns]

        predicted_value = float(best_model.predict(next_features)[0])
        predicted_value = max(0, round(predicted_value))

        future_rows.append({
            "Month": next_date,
            "Predicted_Books": predicted_value
        })

        # Use the prediction as history for the next forecast.
        history = pd.concat([
            history,
            pd.DataFrame([{
                "Month": next_date,
                "Books_Issued": predicted_value
            }])
        ], ignore_index=True)

    forecast = pd.DataFrame(future_rows)

    print("\n========== NEXT 6 MONTHS FORECAST ==========")
    print(forecast.to_string(index=False))

    forecast.to_csv("library_predictions.csv", index=False)
    print("\nPredictions saved to library_predictions.csv")

    # Graph 3: actual data and future predictions
    plt.figure(figsize=(11, 5))
    plt.plot(df["Month"], df["Books_Issued"], marker="o",
             label="Historical actual data")
    plt.plot(forecast["Month"], forecast["Predicted_Books"],
             marker="x", linestyle="--", label="ML forecast")
    plt.title("Library Book Demand: Actual vs Predicted")
    plt.xlabel("Month")
    plt.ylabel("Books issued")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("machine_learning_forecast.png", dpi=150)
    plt.show()


def main():
    print("LIBRARY BOOK DEMAND ANALYSIS")
    print("Topics: Time Series Analysis and Machine Learning only")

    df = load_data()
    if df is None:
        return

    show_summary(df)
    time_series_analysis(df)
    machine_learning_prediction(df)

    print("\nProject completed.")
    print("Generated files:")
    print("- time_series_trend.png")
    print("- monthly_seasonality.png")
    print("- machine_learning_forecast.png")
    print("- library_predictions.csv")


if __name__ == "__main__":
    main()
