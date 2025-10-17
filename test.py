from drivecycle.io import load_cycle
df = load_cycle("data/final_drive_cycle.csv")
print(df.head())
print(f"\nRows loaded: {len(df)}")