﻿import hashlib

import pandas as pd
import torch


def ip_encoder(df, column_name):
    def encode_ip(ip_address):
        if ':' in ip_address:
            # IPv6 address
            parts = [int(x, 16) if x else 0 for x in ip_address.split(':')]
            parts += [0] * (8 - len(parts))
            max_value = 65535  # Max value for a segment in IPv6 (16 bits)
        elif '.' in ip_address:
            # IPv4 address
            parts = [int(x) for x in ip_address.split('.')] + [0] * 4
            max_value = 255    # Max value for a byte in IPv4
        else:
            raise ValueError("Invalid IP address format")

        # Normalize the parts
        normalized_parts = [x / max_value for x in parts]
        return normalized_parts

    df_expanded = pd.DataFrame(df[column_name].apply(encode_ip).tolist(),
                               index=df.index,
                               columns=[f"{column_name}_part{i}" for i in range(1, 9)])
    df = df.drop(column_name, axis=1).join(df_expanded)
    return df

def string_encoder(df, column_name):
    def get_hash(x):
        if pd.notna(x):
            return int(hashlib.sha256(x.encode()).hexdigest(), 16)
        else:
            return 0
    
    hashed_values = df[column_name].apply(get_hash)
    tensor = torch.tensor(hashed_values.values.astype(float))
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    df[f'{column_name}_normalized'] = normalized_tensor.numpy()
    df = df.drop(column_name, axis=1)
    df = df.drop(f'{column_name}_hashed', axis=1, errors='ignore')
    return df

def number_normalizer(df, column_name):
    df[f'{column_name}_int'] = df[column_name].apply(
        lambda x: int(x) if pd.notna(x) else None)
    tensor = torch.tensor(df[f'{column_name}_int'].dropna().values.astype(float))
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    normalized_series = pd.Series(
        normalized_tensor.numpy(), index=df[df[column_name].notna()].index)
    df[f'{column_name}_normalized'] = normalized_series
    df = df.drop([column_name, f'{column_name}_int'], axis=1)
    return df

def boolean_string_to_int(df, column = None):
    mapping = {'False': 0, 'True': 1}
    if column:
        df[column] = df[column].map(mapping).astype(int)
    else:
        df = df.map(mapping).astype(int)
    return df
