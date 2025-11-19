import os
import urllib.request
import urllib.error
from datetime import datetime
import argparse

def download_proteome(proteome_id, date=None):
    # URL to fetch the proteome
    url = f"https://rest.uniprot.org/uniprotkb/stream?download=true&format=fasta&query=%28%28proteome%3A{proteome_id}%29+AND+reviewed%3Dtrue%29"
    
    # Directory to save the proteome file
    proteomes_dir = os.getenv("PROTEOMES_DIR")
    if not proteomes_dir:
        raise EnvironmentError("PROTEOMES_DIR environment variable is not set.")
    
    # Create directory if it doesn't exist
    if not os.path.exists(proteomes_dir):
        os.makedirs(proteomes_dir)
    
    # Fetch the proteome data
    try:
        with urllib.request.urlopen(url) as response:
            if response.status != 200:
                raise Exception(f"Failed to download proteome: {response.status}")
            data = response.read().decode('utf-8')
            
            # Parse organism name from FASTA file header (first line)
            first_line = data.split('\n')[0]
            organism = first_line.split("OS=")[1].split("OX=")[0].strip().replace(' ', '_').lower()
    except urllib.error.URLError as e:
        raise Exception(f"Failed to download proteome: {e.reason}")
    
    # Use the provided date or the current date
    current_date = date if date else datetime.now().strftime("%Y_%m_%d")
    
    # Construct file name
    file_name = f"{organism}_uniprotkb_proteome_{proteome_id}_{current_date}.fasta"
    file_path = os.path.join(proteomes_dir, file_name)
    
    # Save the proteome data to the file
    with open(file_path, "w") as f:
        f.write(data)
    
    print(f"Proteome saved to: {file_path}")

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Download a reference genome from UniProt by proteome ID.")
    parser.add_argument("proteome_id", type=str, help="The UniProt proteome ID (e.g., UP000000589)")
    parser.add_argument("--date", type=str, help="Optional date for file naming (format: YYYY_MM_DD). Defaults to current date.")
        
    # Parse arguments
    args = parser.parse_args()
    
    # Download the proteome using the provided ID
    download_proteome(args.proteome_id)
