import json
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime, timedelta

# config
BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'

# create processed dir if not exists
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("Starting data preprocessing...")
print(f"Raw data dir: {RAW_DIR}")
print(f"Processed data dir: {PROCESSED_DIR}")



# Step 1: Sample Google Cluster Trace
def sample_google_trace(input_file, output_file, sample_size=500000):
    """
    Sample Google cluster usage data
    We'll take random samples throughout the file, not just first N rows
    """
    print(f"\n[1/6] Sampling Google trace...")
    print(f"Target: {sample_size:,} rows")

    # first, count total lines
    print("Counting total lines...")
    total_lines = 0
    with open(input_file, 'r') as f:
        for _ in f:
            total_lines += 1

    print(f"Total lines in file: {total_lines:,}")

    # calculate sampling rate
    if total_lines <= sample_size:
        sample_rate = 1.0
    else:
        sample_rate = sample_size / total_lines

    print(f"Sampling rate: {sample_rate:.2%}")

    # random sampling
    data = []
    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            # random sample based on rate
            if np.random.random() < sample_rate:
                try:
                    row = json.loads(line)
                    data.append(row)
                except:
                    continue  # skip bad lines

            # progress
            if (i + 1) % 1000000 == 0:
                print(f"  Processed {i + 1:,} lines, sampled {len(data):,} rows")

            # stop if we have enough
            if len(data) >= sample_size:
                break

    print(f"Sampled {len(data):,} rows")

    # convert to dataframe
    df = pd.DataFrame(data)

    # keep only relevant columns
    relevant_cols = [
        'start_time', 'end_time', 'collection_id', 'instance_index',
        'machine_id', 'average_usage', 'maximum_usage'
    ]

    # check which columns exist
    existing_cols = [col for col in relevant_cols if col in df.columns]
    df = df[existing_cols]

    # extract CPU and memory from average_usage if it's nested
    if 'average_usage' in df.columns and isinstance(df['average_usage'].iloc[0], dict):
        df['cpu_usage'] = df['average_usage'].apply(lambda x: x.get('cpus', 0) if isinstance(x, dict) else 0)
        df['memory_usage'] = df['average_usage'].apply(lambda x: x.get('memory', 0) if isinstance(x, dict) else 0)
        df.drop('average_usage', axis=1, inplace=True)

    # save
    df.to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")
    print(f"Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

    return df


# Step 2: Sample AWS EC2 Pricing
def sample_aws_pricing(input_file, output_file, sample_size=50000):
    """
    Sample AWS EC2 pricing - handles single large JSON object format
    """
    print(f"\n[2/6] Sampling AWS EC2 pricing...")
    print(f"Target: {sample_size:,} rows")

    print("Loading AWS pricing file (this may take a few minutes)...")

    try:
        # load the entire JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            aws_data = json.load(f)

        print(f"✓ JSON loaded successfully")
        print(f"  File structure: {list(aws_data.keys())}")

        # extract products dictionary
        if 'products' in aws_data:
            products = aws_data['products']
            print(f"  Found {len(products):,} products")

            # convert to list of records
            data = []
            for sku, product_info in products.items():
                # flatten the nested structure
                record = {
                    'sku': sku,
                    'productFamily': product_info.get('productFamily', ''),
                }

                # extract attributes
                if 'attributes' in product_info:
                    attrs = product_info['attributes']
                    record.update({
                        'serviceCode': attrs.get('servicecode', ''),
                        'location': attrs.get('location', ''),
                        'locationType': attrs.get('locationType', ''),
                        'instanceType': attrs.get('instanceType', ''),
                        'vcpu': attrs.get('vcpu', ''),
                        'memory': attrs.get('memory', ''),
                        'storage': attrs.get('storage', ''),
                        'networkPerformance': attrs.get('networkPerformance', ''),
                        'physicalProcessor': attrs.get('physicalProcessor', ''),
                        'clockSpeed': attrs.get('clockSpeed', ''),
                        'operatingSystem': attrs.get('operatingSystem', ''),
                        'licenseModel': attrs.get('licenseModel', ''),
                        'tenancy': attrs.get('tenancy', ''),
                        'usagetype': attrs.get('usagetype', ''),
                    })

                data.append(record)

            print(f"  Converted to {len(data):,} records")

            # sample if too large
            if len(data) > sample_size:
                print(f"  Sampling {sample_size:,} random records...")
                indices = np.random.choice(len(data), sample_size, replace=False)
                data = [data[i] for i in indices]

            print(f"✓ Final sample size: {len(data):,} rows")

        else:
            print("⚠ 'products' key not found in JSON")
            return pd.DataFrame()

        # convert to dataframe
        df = pd.DataFrame(data)

        print(f"\nDataframe created:")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Column names: {list(df.columns)[:10]}...")

        # save
        df.to_csv(output_file, index=False)
        print(f"\n✓ Saved to: {output_file}")
        print(f"  Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

        return df

    except Exception as e:
        print(f"\n Error loading AWS pricing: {str(e)}")
        return pd.DataFrame()



# Step 3: Load Small Pricing Files
def load_small_pricing_files():
    """
    Load Azure and other AWS pricing files
    """
    print(f"\n[3/6] Loading small pricing files...")

    pricing_data = {}

    pricing_files = {
        'azure_vm': RAW_DIR / 'pricing' / 'azure_vm_pricing.json',
        'azure_storage': RAW_DIR / 'pricing' / 'azure_storage_pricing.json',
        'azure_vnet': RAW_DIR / 'pricing' / 'azure_vnet_pricing.json',
        'azure_aks': RAW_DIR / 'pricing' / 'azure_aks_pricing.json',
        'aws_s3': RAW_DIR / 'pricing' / 'aws_s3_pricing.json',
        'aws_vpc': RAW_DIR / 'pricing' / 'aws_vpc_pricing.json',
        'aws_eks': RAW_DIR / 'pricing' / 'aws_eks_pricing.json',
    }

    for name, filepath in pricing_files.items():
        if filepath.exists():
            print(f"  Loading {name}...")
            with open(filepath, 'r') as f:
                pricing_data[name] = json.load(f)
        else:
            print(f"  Warning: {name} not found")

    return pricing_data


# Step 4: Create Cost Time Series
def create_cost_timeseries(usage_df, aws_pricing_df, azure_pricing):
    """
    Combine usage data with pricing to create cost time series
    NOW includes ALL resources: CPU, memory, storage, network
    """
    print(f"\n[4/6] Creating cost time series...")

    # make a copy to avoid modifying original
    usage_df = usage_df.copy()

    # convert timestamps to datetime
    usage_df['start_time'] = pd.to_numeric(usage_df['start_time'], errors='coerce')
    usage_df['end_time'] = pd.to_numeric(usage_df['end_time'], errors='coerce')

    # calculate duration in hours
    usage_df['duration_hours'] = (usage_df['end_time'] - usage_df['start_time']) / (1e6 * 3600)

    # randomly assign cloud provider (50/50 AWS/Azure)
    usage_df['cloud_provider'] = np.random.choice(['aws', 'azure'], size=len(usage_df))

    # add storage and network usage - simulate realistic values
    usage_df['storage_gb'] = np.random.uniform(10, 500, size=len(usage_df))  # 10-500 GB
    usage_df['network_gb'] = usage_df['duration_hours'] * np.random.uniform(0.5, 5, size=len(usage_df))  # GB per hour

    # pricing weights for different resources (different for each provider)
    # AWS pricing
    aws_cpu_price = 0.05  # per cpu per hour
    aws_memory_price = 0.02  # per GB per hour
    aws_storage_price = 0.1 / 730  # $0.1 per GB per month = 0.000137/hour
    aws_network_price = 0.09  # per GB transferred

    # Azure pricing (slightly different)
    azure_cpu_price = 0.055  # 10% more expensive
    azure_memory_price = 0.022
    azure_storage_price = 0.12 / 730
    azure_network_price = 0.087  # slightly cheaper network

    # calculate realistic cost including all resources
    def calculate_cost(row):
        if row['cloud_provider'] == 'aws':
            cpu_cost = row['cpu_usage'] * aws_cpu_price * row['duration_hours']
            memory_cost = row['memory_usage'] * aws_memory_price * row['duration_hours']
            storage_cost = row['storage_gb'] * aws_storage_price * row['duration_hours']
            network_cost = row['network_gb'] * aws_network_price
        else:  # azure
            cpu_cost = row['cpu_usage'] * azure_cpu_price * row['duration_hours']
            memory_cost = row['memory_usage'] * azure_memory_price * row['duration_hours']
            storage_cost = row['storage_gb'] * azure_storage_price * row['duration_hours']
            network_cost = row['network_gb'] * azure_network_price

        # total cost with all components
        total = cpu_cost + memory_cost + storage_cost + network_cost

        # add some variation based on service type (kubernetes more expensive)
        if row['service_type'] == 'kubernetes':
            total *= 1.3  # 30% overhead
        elif row['service_type'] == 'storage':
            total *= 0.8  # cheaper, storage-heavy

        return total

    # add service type BEFORE calculating cost
    usage_df['service_type'] = np.random.choice(
        ['vm', 'storage', 'network', 'kubernetes'],
        size=len(usage_df),
        p=[0.5, 0.2, 0.2, 0.1]  # probabilities
    )

    # now calculate cost
    usage_df['cost'] = usage_df.apply(calculate_cost, axis=1)

    # keep relevant columns including new storage and network
    cost_df = usage_df[[
        'start_time', 'end_time', 'cloud_provider', 'service_type',
        'cpu_usage', 'memory_usage', 'storage_gb', 'network_gb', 'duration_hours', 'cost'
    ]].copy()

    # aggregate by hour for time series
    cost_df['hour'] = (cost_df['start_time'] / (1e6 * 3600)).astype(int)

    hourly_costs = cost_df.groupby(['hour', 'cloud_provider', 'service_type']).agg({
        'cost': 'sum',
        'cpu_usage': 'mean',
        'memory_usage': 'mean',
        'storage_gb': 'mean',
        'network_gb': 'sum',
        'duration_hours': 'sum'
    }).reset_index()

    print(f"Created {len(hourly_costs):,} hourly cost records")

    # save
    output_file = PROCESSED_DIR / 'cost_timeseries.csv'
    hourly_costs.to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")

    return hourly_costs



# Step 5: Normalize Data for ML
def normalize_data(cost_df):
    """
    Normalize data for ML model training
    """
    print(f"\n[5/6] Normalizing data for ML...")

    from sklearn.preprocessing import StandardScaler

    # features for normalization - NOW includes storage and network
    features = ['cpu_usage', 'memory_usage', 'storage_gb', 'network_gb', 'duration_hours', 'cost']

    # create scaler
    scaler = StandardScaler()

    # fit and transform
    cost_df_normalized = cost_df.copy()
    cost_df_normalized[features] = scaler.fit_transform(cost_df[features])

    # save scaler
    scaler_path = PROCESSED_DIR.parent / 'trained_models' / 'scaler.pkl'
    scaler_path.parent.mkdir(exist_ok=True)

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Scaler saved to: {scaler_path}")

    # save normalized data
    output_file = PROCESSED_DIR / 'normalized_data.pkl'
    cost_df_normalized.to_pickle(output_file)
    print(f"Normalized data saved to: {output_file}")

    return cost_df_normalized, scaler


# Step 6: Generate Summary Stats
def generate_summary(usage_df, cost_df):
    """
    Print summary statistics
    """
    print(f"\n[6/6] Summary Statistics")
    print("=" * 50)

    print(f"\nUsage Data:")
    print(f"  Total records: {len(usage_df):,}")
    print(f"  Date range: {usage_df['start_time'].min()} to {usage_df['end_time'].max()}")
    print(f"  Avg CPU usage: {usage_df['cpu_usage'].mean():.4f}")
    print(f"  Avg memory usage: {usage_df['memory_usage'].mean():.4f}")

    # add storage and network stats if they exist
    if 'storage_gb' in usage_df.columns:
        print(f"  Avg storage: {usage_df['storage_gb'].mean():.2f} GB")
    if 'network_gb' in usage_df.columns:
        print(f"  Avg network: {usage_df['network_gb'].mean():.2f} GB")

    print(f"\nCost Data:")
    print(f"  Total records: {len(cost_df):,}")
    print(f"  Total cost: ${cost_df['cost'].sum():,.2f}")
    print(f"  Avg cost per hour: ${cost_df['cost'].mean():.2f}")
    print(f"  Max cost per hour: ${cost_df['cost'].max():.2f}")

    print(f"\nBy Cloud Provider:")
    provider_stats = cost_df.groupby('cloud_provider')['cost'].agg(['sum', 'mean', 'count'])
    print(provider_stats)

    print(f"\nBy Service Type:")
    service_stats = cost_df.groupby('service_type')['cost'].agg(['sum', 'mean', 'count'])
    print(service_stats)

    print("\n" + "=" * 50)
    print("Preprocessing complete ")



# Main Execution
if __name__ == '__main__':
    # file paths
    google_trace_file = RAW_DIR / 'workloads' / 'instance_usage-000000000000.json'
    aws_ec2_file = RAW_DIR / 'pricing' / 'aws_ec2_pricing.json'

    google_sample_file = PROCESSED_DIR / 'google_trace_sample.csv'
    aws_sample_file = PROCESSED_DIR / 'aws_ec2_sample.csv'

    # Step 1: Sample Google trace
    usage_df = sample_google_trace(google_trace_file, google_sample_file, sample_size=500000)

    # Step 2: Sample AWS pricing
    aws_pricing_df = sample_aws_pricing(aws_ec2_file, aws_sample_file, sample_size=50000)

    # Step 3: Load small pricing files
    azure_pricing = load_small_pricing_files()

    # Step 4: Create cost time series
    cost_df = create_cost_timeseries(usage_df, aws_pricing_df, azure_pricing)

    # Step 5: Normalize for ML
    normalized_df, scaler = normalize_data(cost_df)

    # Step 6: Summary
    generate_summary(usage_df, cost_df)

    print("\n✅ All done! Your data is ready for model training.")
    print(f"\nProcessed files location: {PROCESSED_DIR}")