import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

def fit_accuracy_curve(file_path, save_path="fitted_acc_data.csv"):
    """
    Reads accuracy data from an Excel file, fits a monotonic PCHIP curve,
    and saves the interpolated results.
    """
    # Load data
    df = pd.read_excel(file_path)

    # Define SNR range
    snr_values = np.array([0, 3, 5, 7, 10])  # Given SNR values
    fitted_snr = np.linspace(0, 10, 100)  # Continuous SNR range

    # Dictionary to store fitted results
    fitted_data = {
        "Model": [],
        "Coding Rate": [],
        "SNR": [],
        "Accuracy": []
    }

    # Iterate over rows
    for index, row in df.iterrows():
        model = row.iloc[0]  # Model name
        coding_rate = row.iloc[1]  # Channel coding rate
        accuracy = row.iloc[2:].values  # Accuracy at given SNRs

        # Ensure accuracy is strictly increasing using PCHIP
        interpolator = PchipInterpolator(snr_values, accuracy)
        fitted_accuracy = interpolator(fitted_snr)

        # Store results
        for snr, acc in zip(fitted_snr, fitted_accuracy):
            fitted_data["Model"].append(model)
            fitted_data["Coding Rate"].append(coding_rate)
            fitted_data["SNR"].append(snr)
            fitted_data["Accuracy"].append(acc)

    # Save results to CSV
    fitted_df = pd.DataFrame(fitted_data)
    fitted_df.to_csv(save_path, index=False)

    # Plot fitted accuracy curves
    plt.figure(figsize=(8, 5))
    for model_name in df.iloc[:, 0].unique():
        subset = fitted_df[fitted_df["Model"] == model_name]
        plt.plot(subset["SNR"], subset["Accuracy"], label=model_name)

    plt.xlabel("SNR")
    plt.ylabel("Accuracy")
    plt.title("Fitted Accuracy Curves")
    plt.legend()
    plt.grid()
    plt.show()


# Run the function with your data file
fit_accuracy_curve("acc_data.xlsx")
