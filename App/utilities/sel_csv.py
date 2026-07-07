import pandas as pd
import random

def sample_csv_reservoir(input_path, output_path, n=20000, **kwargs):
    reservoir = []
    total = 0

    print("Inizio lettura...")

    for chunk in pd.read_csv(input_path, chunksize=5000, **kwargs):
        for _, row in chunk.iterrows():
            total += 1

            if total % 50000 == 0:
                print(f"Letti {total} record...")

            if len(reservoir) < n:
                reservoir.append(row)
            else:
                j = random.randint(0, total - 1)
                if j < n:
                    reservoir[j] = row

    print(f"Totale righe lette: {total}")

    df_sample = pd.DataFrame(reservoir)
    df_sample.to_csv(output_path, index=False)

    print(f"File salvato: {output_path}")
    return df_sample


if __name__ == "__main__":
    sample_csv_reservoir(
        r"C:/Users/elyon/OneDrive/Desktop/Cosmo-Edu_Lab/data/GaiaSource.csv",
        r"C:/Users/elyon/OneDrive/Desktop/Cosmo-Edu_Lab/data/Gaia_20000.csv",
        n=20000,
        quotechar="'",
        skipinitialspace=True
    )
