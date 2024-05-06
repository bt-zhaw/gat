import pandas as pd


def construct_port_scan_label(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df.sort_values(by=['ip_source', 'timestamp'], inplace=True)
    time_window = '10S'

    def diversity_index(x):
        if not isinstance(x, pd.Series):
            x = pd.Series(x)
        return x.nunique() / len(x) if len(x) > 0 else 0

    df.set_index('timestamp', inplace=True)
    results = df.groupby('ip_source')['destination_port_label'].rolling(
        window=time_window).apply(diversity_index, raw=False)
    df['diversity_index'] = results.values
    df.reset_index(inplace=True)
    df.drop(columns=['timestamp'], inplace=True)
    df['diversity_index'] = df['diversity_index'].fillna(0)
    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def label_converter(df):
    df = construct_port_scan_label(df)
    return df
