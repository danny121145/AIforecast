import pandas as pd

def load_and_prepare(file_path, uid=None):
    df = pd.read_csv(file_path)
    df['Sales Date'] = pd.to_datetime(df['Sales Date'], dayfirst=True)
    all_uids = df['UID'].unique().tolist()

    if uid:
        df = df[df['UID'] == uid]
        df = df.groupby(pd.Grouper(key='Sales Date', freq='ME')).agg({'Amount (USD)': 'sum'}).reset_index()
        df = df.rename(columns={"Sales Date": "ds", "Amount (USD)": "y"})

    return df, all_uids
