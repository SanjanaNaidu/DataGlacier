import pandas as pd
import dask.dataframe as dd
import time
import os
import yaml
import modin.pandas as mpd

# Ensure Ray is initialized
import ray
ray.init(ignore_reinit_error=True)

# Define the file path
file_path = '../Data Glacier Week6/car_prices.csv'

# Read with Pandas
start_time = time.time()
pandas_df = pd.read_csv(file_path)
print(f"Pandas took {time.time() - start_time} seconds.")

# Read with Dask
start_time = time.time()
dask_df = dd.read_csv(file_path, dtype={'condition': 'int64',
                                        'mmr': 'int64',
                                        'odometer': 'int64',
                                        'sellingprice': 'int64'}).compute()
print(f"Dask took {time.time() - start_time} seconds.")

# Read with Modin and Ray
start_time = time.time()
modin_ray_df = mpd.read_csv(file_path)
print(f"Modin with Ray took {time.time() - start_time} seconds.")

# Column name validation and cleanup
pandas_df.columns = [col.replace(' ', '_').replace('|', '_').replace('-', '_') for col in pandas_df.columns]




# Load the YAML schema
schema_file_path = r'C:\Users\sanja\Desktop\Data Glacier Week6\schema.yml'
with open(schema_file_path, 'r') as f:
    # Your code here


    schema = yaml.safe_load(f)

# Validate the number of columns and column names
assert len(schema['columns']) == len(pandas_df.columns), "Column count mismatch"
assert all(col in pandas_df.columns for col in schema['columns']), "Column names mismatch"
# Write the DataFrame to a new gzipped pipe-separated file
pandas_df.to_csv('output_file.gz', sep='|', index=False, compression='gzip')
summary = {
    "Total rows": len(pandas_df),
    "Total columns": len(pandas_df.columns),
    "File size": os.path.getsize('output_file.gz')
}

print(summary)
