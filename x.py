import duckdb
data_dir = "/GenSIvePFS/users/lutianyu/lf/dataset/sequence/uniref_dplm/train/*.parquet"
output = "/GenSIvePFS/users/lutianyu/lf/dataset/sequence/uniref_dplm/uniref50_dplm_train_full.parquet"

duckdb.sql(f"""
    COPY (
        SELECT * 
        FROM read_parquet('{data_dir}')
        USING SAMPLE 100%  -- 全局 shuffle
    ) TO '{output}' (FORMAT PARQUET);
""")