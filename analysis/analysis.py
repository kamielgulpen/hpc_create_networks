import pandas as pd
import data_lake as dl


df = pd.read_parquet(dl.ROOT / "analysis_tables" / "full_table.parquet")

print(df.columns)
print(df["mean_final_adoption_fraction"].describe())