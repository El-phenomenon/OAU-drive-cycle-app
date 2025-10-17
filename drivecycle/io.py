# drivecycle/io.py
import pandas as pd

def load_cycle(path_or_buffer):
    """
    Load the OAU drive cycle CSV and extract time and representative speed.

    Parameters
    ----------
    path_or_buffer : str or file-like
        Path to the CSV file (e.g., 'data/final_drive cycle.csv')

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame with columns: 'time_s' and 'speed_m_s'
    """
    df = pd.read_csv(path_or_buffer)
    # Normalize column names (remove spaces, lowercase)
    df.columns = [c.strip().lower() for c in df.columns]

    if 'time_s' not in df.columns or 'rep_speed_mps' not in df.columns:
        raise ValueError("CSV must contain 'Time_s' and 'Rep_Speed_mps' columns.")

    df_out = df[['time_s', 'rep_speed_mps']].copy()
    df_out = df_out.rename(columns={'rep_speed_mps': 'speed_m_s'})
    return df_out