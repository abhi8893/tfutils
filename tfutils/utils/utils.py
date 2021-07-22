

def get_dataframe_cols(df, cols):
    '''Get dataframe columns which are present in the pandas dataframe'''
    
    cols = df.columns[df.columns.isin(cols)]
    
    return df[cols]