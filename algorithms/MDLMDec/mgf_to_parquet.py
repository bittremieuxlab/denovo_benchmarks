import sys
import argparse
import os
import pandas as  pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
import shutil
from tqdm import tqdm
from glob import glob

# Function to read mgf file and save positions of all spectra
def gather_file_md(filepath, typ=None):
    if typ==None:
        typ = filepath.split('.')[-1].strip().lower()

    with open(filepath) as f:
        _ = f.read()
        end = f.tell()
        f.seek(0)
        
        pos = f.tell()
        pos_prev = f.tell()
        spectra = {}
        spec_ticker = 0
        while pos!=end:

            line = f.readline().strip()
            pos = f.tell()
            
            if typ=='mgf':
                if line == 'BEGIN IONS':
                    # If the last spectrum had no peaks (or no charge), delete it
                    if spec_ticker in spectra:
                        del(spectra[spec_ticker])

                    spectra[spec_ticker] = {}
                elif line.split('=')[0] == 'TITLE':
                    spectra[spec_ticker]['title'] = line.strip().split('=')[-1]
                elif line.split('=')[0] == 'SCANS':
                    scan = int(line.split('=')[-1])
                    spectra[spec_ticker]['scan'] = scan
                elif line.split('=')[0] == 'RTINSECONDS':
                    rt = float(line.split('=')[-1])
                    spectra[spec_ticker]['rt'] = rt
                elif line.split('=')[0] == 'CHARGE':
                    charge = int(line.split('=')[-1].replace('+',''))
                    spectra[spec_ticker]['charge'] = charge
                elif line.split('=')[0] == 'PEPMASS':
                    mass = float(line.split('=')[-1].split()[0])
                    spectra[spec_ticker]['mass'] = mass
                elif line.split('=')[0] == 'SEQUENCE':
                    seq = line.split('=')[-1]
                    spectra[spec_ticker]['sequence'] = seq
                elif len(line.split('.')) == 3:
                    peak_ticker = 0
                    spectra[spec_ticker]['pos'] = pos_prev
                    while line != 'END IONS':
                        peak_ticker += 1
                        line = f.readline().strip()
                    spectra[spec_ticker]['nmpks'] = peak_ticker
                    
                    if (
                        'charge' in spectra[spec_ticker] and
                        'mass' in spectra[spec_ticker]
                    ):
                        spec_ticker += 1
                
                pos_prev = pos
            
            else:
                NotImplementedError("File type not implemented yet.")
    
    # If the very last spectrum had no peaks, delete it
    tick = max(list(spectra.keys()))
    if (
        'pos' not in spectra[tick] or
        'charge' not in spectra[tick]
    ):
        del(spectra[tick])
    return spectra

def main():
    # Parse arguments to get data directory
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, help="Input directory containing .mgf files"
    )
    parser.add_argument(
        "--divisions", default=1, help="Number of parquet files to create (sharding)"
    )
    args = parser.parse_args()
    
    # Get rid of any parquet files/directory that preceded this run
    if os.path.isdir('parquet'):
        shutil.rmtree("parquet")
    elif os.path.exists('parquet'):
        os.remove("parquet")
    os.mkdir("parquet")

    # Will first write an all-in-one parquet
    parquet_file = "parquet/full.parquet"
    schema_defined = False
    
    # Run all files in input directory
    files = glob(os.path.join(args.input, "*"))
    pbar = tqdm(files)
    counter = 0
    for file in pbar:
        
        name = file.split('.')[0]
        mem = psutil.virtual_memory()[2]
        pbar.set_description('%s (%.1f)'%(name, mem))
        
        # Will write one row at a time
        rows = {
            'file': [],
            'index': [],
            'precursor_charge': [],
            'precursor_mass': [],
            'mz_array': [],
            'intensity_array': [],
        }
        # Read all the data in an mgf file
        md = gather_file_md(file)
        counter += len(md)
        for a, dic in md.items():
            
            mzs = []
            Abs = []
            with open(file) as f:
                # First read the spectrum
                f.seek(md[a]['pos'])
                for i in range(md[a]['nmpks']):
                    mz, ab = f.readline().strip().split()
                    mz = np.float32(mz)
                    ab = np.float32(ab)
                    mzs.append(mz)
                    Abs.append(ab)
                # Place rest of the information into rows
                rows['precursor_charge'].append(np.int32(dic['charge']))
                rows['precursor_mass'].append(np.float32(dic['mass'])) # MassIVEKB mgfs are strange: they list the mass and not the m/z
                rows['file'].append(file.split('/')[-1].split('.')[0])
                rows['index'].append(a)
                if 'title' in dic:
                    if 'title' not in rows:
                        rows['title'] = []
                    rows['title'].append(dic['title'])
                if 'scan' in dic:
                    if 'scan_number' not in rows:
                        rows['scan_number'] = []
                    rows['scan_number'].append(dic['scan'])
                if 'rt' in dic:
                    if 'retention_time' not in rows:
                        rows['retention_time'] = []
                    rows['retention_time'].append(dic['rt'])
                rows['mz_array'].append(mzs)
                rows['intensity_array'].append(Abs)
                if 'sequence' in dic:
                    if 'sequence' not in rows:
                        rows['sequence'] = []
                    rows['sequence'].append(dic['sequence'])
        
        # Write all the information from an MGF to the parquet file
        table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
        
        if not schema_defined:
            writer = pq.ParquetWriter(parquet_file, table.schema, compression='snappy')
            schema_defined = True

        writer.write_table(table)
    
    # Close the file after all mgfs have been read
    if schema_defined:
        writer.close()

    # Shard the full parquet into {divisions} separate files
    df = pd.read_parquet("parquet/full.parquet")
    division_size = len(df) // int(args.divisions)
    for i in range(int(args.divisions)):
        start = i*division_size
        if i == int(args.divisions)-1:
            end=999999999999999
        else:
            end = (i+1)*division_size
        partition = df.iloc[start:end]
        partition.to_parquet(f"parquet/partition_{i}.parquet")
    os.remove("parquet/full.parquet")
    open("parquet/size.tsv", "w").write(f'all\t{counter}')

if __name__ == '__main__':
    main()
