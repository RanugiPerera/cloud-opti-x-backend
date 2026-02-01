"""
Download Cloud Pricing Data from Azure and AWS APIs
This script fetches pricing for VMs, Storage, Networking, and Kubernetes
"""

import requests
import json
import os
from datetime import datetime

# create output directory
OUTPUT_DIR = 'data/raw/pricing'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("CLOUD PRICING DATA DOWNLOADER")
print("=" * 60)

# AZURE PRICING API
def fetch_azure_prices(service_name, output_filename):
    """Fetch Azure pricing for a specific service"""
    
    print(f"\n[Azure] Downloading {service_name} pricing...")
    
    url = "https://prices.azure.com/api/retail/prices"
    all_items = []
    
    params = {
        "$filter": f"serviceName eq '{service_name}' and priceType eq 'Consumption'",
        "currencyCode": "USD"
    }
    
    # handle pagination
    page = 1
    while url:
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            items = data.get('Items', [])
            all_items.extend(items)
            
            print(f"  Page {page}: {len(items)} items (Total: {len(all_items)})")
            
            url = data.get('NextPageLink')  # get next page
            params = {}  # clear params for pagination
            page += 1
            
            # limit to first 5 pages to avoid huge files
            if page > 5:
                print(f"  Limiting to 5 pages for manageable file size")
                break
                
        except requests.exceptions.RequestException as e:
            print(f"  Error: {e}")
            break
    
    # save to json
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_items, f, indent=2)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"  ✓ Saved {len(all_items)} items to {output_filename} ({file_size:.2f} MB)")
    
    return len(all_items)


# AWS PRICING API
def fetch_aws_prices(service_code, output_filename):
    """Fetch AWS pricing for a specific service"""
    
    print(f"\n[AWS] Downloading {service_code} pricing...")
    
    try:
        # first get the index
        index_url = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/index.json"
        print(f"  Fetching AWS pricing index...")
        response = requests.get(index_url, timeout=30)
        response.raise_for_status()
        index = response.json()
        
        # get the offer file url
        if service_code not in index.get('offers', {}):
            print(f"  ✗ Service {service_code} not found in AWS index")
            return 0
        
        offer_path = index['offers'][service_code]['currentVersionUrl']
        offer_url = f"https://pricing.us-east-1.amazonaws.com{offer_path}"
        
        print(f"  Downloading pricing data (this may take 30-60 seconds)...")
        response = requests.get(offer_url, timeout=120)
        response.raise_for_status()
        
        pricing_data = response.json()
        
        # save full data
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pricing_data, f, indent=2)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        
        # count products
        num_products = len(pricing_data.get('products', {}))
        print(f"  ✓ Saved {num_products} products to {output_filename} ({file_size:.2f} MB)")
        
        return num_products
        
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Error: {e}")
        return 0


# MAIN EXECUTION
def main():
    start_time = datetime.now()
    
    # azure services to download
    azure_services = [
        ('Virtual Machines', 'azure_vm_pricing.json'),
        ('Storage', 'azure_storage_pricing.json'),
        ('Virtual Network', 'azure_vnet_pricing.json'),
        ('Azure Kubernetes Service (AKS)', 'azure_aks_pricing.json')
    ]
    
    # aws services to download
    aws_services = [
        ('AmazonEC2', 'aws_ec2_pricing.json'),
        ('AmazonS3', 'aws_s3_pricing.json'),
        ('AmazonVPC', 'aws_vpc_pricing.json'),
        ('AmazonEKS', 'aws_eks_pricing.json')
    ]
    
    print("\n" + "=" * 60)
    print("DOWNLOADING AZURE PRICING DATA")
    print("=" * 60)
    
    azure_total = 0
    for service_name, filename in azure_services:
        count = fetch_azure_prices(service_name, filename)
        azure_total += count
    
    print("\n" + "=" * 60)
    print("DOWNLOADING AWS PRICING DATA")
    print("=" * 60)
    
    aws_total = 0
    for service_code, filename in aws_services:
        count = fetch_aws_prices(service_code, filename)
        aws_total += count
    
    # summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE!")
    print("=" * 60)
    print(f"Azure items downloaded: {azure_total}")
    print(f"AWS products downloaded: {aws_total}")
    print(f"Total time: {duration:.1f} seconds")
    print(f"Files saved in: {os.path.abspath(OUTPUT_DIR)}")
    
    # list downloaded files
    print("\nDownloaded files:")
    for filename in os.listdir(OUTPUT_DIR):
        filepath = os.path.join(OUTPUT_DIR, filename)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  - {filename} ({size_mb:.2f} MB)")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
