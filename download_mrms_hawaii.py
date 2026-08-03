"""
Download MRMS (Multi-Radar/Multi-Sensor) 1-hour QPE data for Hawaii domain.

MRMS provides gridded, multi-sensor precipitation estimates combining 
radar, satellite, and gauge data. This script downloads 1-hour QPE 
accumulation data for a specified time period over Hawaii.
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
import requests
import s3fs
import xarray as xr
import numpy as np
from tqdm import tqdm
import warnings
import gzip
import shutil

# Import local configuration
import config


class MRMSHawaiiDownloader:
    """Download MRMS QPE data for Hawaii domain."""
    
    # Hawaii bounding box (approx)
    HAWAII_BOUNDS = {
        'lat_min': 18.0,
        'lat_max': 23.0,
        'lon_min': -161.0,
        'lon_max': -154.0
    }
    
    def __init__(self, output_dir=None, verbose=False):
        """
        Initialize MRMS downloader.
        
        Args:
            output_dir (Path or str): Output directory for downloaded files
            verbose (bool): Enable verbose output
        """
        self.verbose = verbose
        self.output_dir = Path(output_dir) if output_dir else config.get_output_dir() / "mrms_hawaii"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # MRMS data sources
        self.sources = {
            'aws': 'noaa-mrms-pds',  # AWS S3 bucket (has dedicated HAWAII directory)
            'iowa': 'https://mtarchive.geol.iastate.edu',  # Iowa State archive
            'ncei': 'https://www.ncei.noaa.gov/data/mrms',  # NCEI archive
        }
        
        # Use Hawaii-specific path on AWS
        self.aws_region = 'HAWAII'  # Changed from CONUS to HAWAII
        
        # Initialize S3 filesystem for AWS access
        try:
            self.s3_fs = s3fs.S3FileSystem(anon=True)
            self.has_s3 = True
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not initialize S3 access: {e}")
            self.has_s3 = False
        
        # HTTP session for other sources
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'MRMS-QPE-Downloader/1.0'})
    
    def list_available_products(self):
        """
        List available MRMS products in the Hawaii bucket.
        
        Returns:
            list: Available product names
        """
        if not self.has_s3:
            print("S3 access not available")
            return []
        
        try:
            hawaii_path = f"{self.sources['aws']}/{self.aws_region}"
            products = self.s3_fs.ls(hawaii_path)
            product_names = [p.split('/')[-1] for p in products]
            
            print(f"\nAvailable MRMS products in {self.aws_region}:")
            for prod in product_names:
                print(f"  - {prod}")
            print()
            
            return product_names
        except Exception as e:
            print(f"Error listing products: {e}")
            return []
    
    def list_available_dates(self, product='MultiSensor_QPE_01H_Pass2', limit=10):
        """
        List available dates for a specific product.
        
        Args:
            product (str): MRMS product name
            limit (int): Maximum number of dates to show
            
        Returns:
            list: Available dates
        """
        if not self.has_s3:
            print("S3 access not available")
            return []
        
        try:
            product_path = f"{self.sources['aws']}/{self.aws_region}/{product}_00.00"
            dates = self.s3_fs.ls(product_path)
            date_dirs = [d.split('/')[-1] for d in dates]
            
            print(f"\nAvailable dates for {product} (showing last {limit}):")
            for date_dir in sorted(date_dirs)[-limit:]:
                print(f"  - {date_dir}")
            print()
            
            return sorted(date_dirs)
        except Exception as e:
            print(f"Error listing dates: {e}")
            return []
    
    def cleanup_temp_files(self):
        """
        Clean up any temporary files (.idx, temp_*.grib2, temp_*.grib2.gz) in output directory.
        
        This is called automatically after downloading, but can be run manually if needed.
        """
        patterns = ['*.idx', 'temp_*.grib2', 'temp_*.grib2.gz']
        total_cleaned = 0
        
        for pattern in patterns:
            files = list(self.output_dir.glob(pattern))
            for f in files:
                try:
                    f.unlink()
                    total_cleaned += 1
                    if self.verbose:
                        print(f"  Removed: {f.name}")
                except Exception as e:
                    if self.verbose:
                        print(f"  Warning: Could not delete {f.name}: {e}")
        
        if total_cleaned > 0:
            print(f"✓ Cleaned up {total_cleaned} temporary files")
        elif self.verbose:
            print("No temporary files to clean up")
        
        return total_cleaned
    
    def get_mrms_filename(self, dt, product='MultiSensor_QPE_01H_Pass2'):
        """
        Generate MRMS filename for a given datetime.
        
        Args:
            dt (datetime): Datetime for the file
            product (str): MRMS product name
            
        Returns:
            str: MRMS filename
        """
        # MRMS naming convention: MRMS_Product_00.00_YYYYMMDD-HHMMSS.grib2.gz
        # Files in Hawaii bucket are gzip compressed
        timestamp = dt.strftime('%Y%m%d-%H%M%S')
        return f"MRMS_{product}_00.00_{timestamp}.grib2.gz"
    
    def get_aws_path(self, dt, product='MultiSensor_QPE_01H_Pass2'):
        """
        Get AWS S3 path for MRMS Hawaii file.
        
        Args:
            dt (datetime): Datetime for the file
            product (str): MRMS product name
            
        Returns:
            str: S3 path
        """
        # AWS structure for Hawaii: noaa-mrms-pds/HAWAII/Product_00.00/YYYYMMDD/filename
        # Product name in path includes _00.00 suffix
        date_str = dt.strftime('%Y%m%d')
        filename = self.get_mrms_filename(dt, product)
        
        # Primary path: HAWAII/Product_00.00/YYYYMMDD/
        product_with_suffix = f"{product}_00.00"
        return f"{self.sources['aws']}/{self.aws_region}/{product_with_suffix}/{date_str}/{filename}"
    
    def get_iowa_url(self, dt, product='MultiSensor_QPE_01H_Pass2'):
        """
        Get Iowa State archive URL for MRMS file.
        
        Args:
            dt (datetime): Datetime for the file
            product (str): MRMS product name
            
        Returns:
            str: Iowa State URL
        """
        # Iowa structure varies - this is a common pattern
        year = dt.strftime('%Y')
        month = dt.strftime('%m')
        day = dt.strftime('%d')
        filename = self.get_mrms_filename(dt, product)
        return f"{self.sources['iowa']}/{year}/{month}/{day}/mrms/ncep/{product}/{filename}"
    
    def check_file_availability(self, dt, product='MultiSensor_QPE_01H_Pass2'):
        """
        Check if MRMS file is available from any source.
        
        Args:
            dt (datetime): Datetime to check
            product (str): MRMS product name
            
        Returns:
            dict: Source and path/URL if available
        """
        # Try AWS first (fastest)
        if self.has_s3:
            try:
                # Try primary Hawaii path
                s3_path = self.get_aws_path(dt, product)
                if self.s3_fs.exists(s3_path):
                    file_info = self.s3_fs.info(s3_path)
                    if self.verbose:
                        print(f"  Found: {s3_path}")
                    return {
                        'source': 'aws',
                        'path': s3_path,
                        'size_mb': file_info.get('size', 0) / (1024**2),
                        'available': True
                    }
                else:
                    if self.verbose:
                        print(f"  Not found: {s3_path}")
                        
            except Exception as e:
                if self.verbose:
                    print(f"  AWS check failed: {e}")
        
        # Try Iowa State archive
        try:
            iowa_url = self.get_iowa_url(dt, product)
            response = self.session.head(iowa_url, timeout=5)
            if response.status_code == 200:
                size_mb = int(response.headers.get('content-length', 0)) / (1024**2)
                return {
                    'source': 'iowa',
                    'url': iowa_url,
                    'size_mb': size_mb,
                    'available': True
                }
        except Exception as e:
            if self.verbose:
                print(f"  Iowa State check failed: {e}")
        
        return {'available': False}
    
    def download_file(self, dt, product='MultiSensor_QPE_01H_Pass2', subset_hawaii=True):
        """
        Download MRMS file for a specific datetime.
        
        Args:
            dt (datetime): Datetime to download
            product (str): MRMS product name
            subset_hawaii (bool): Subset to Hawaii domain only
            
        Returns:
            Path: Path to downloaded file, or None if failed
        """
        # Check availability
        availability = self.check_file_availability(dt, product)
        
        if not availability['available']:
            if self.verbose:
                print(f"  ❌ File not available for {dt}")
            return None
        
        # Generate output filename
        output_filename = f"mrms_qpe_1hr_hawaii_{dt.strftime('%Y%m%d_%H%M')}.nc"
        output_path = self.output_dir / output_filename
        
        # Skip if already exists
        if output_path.exists():
            if self.verbose:
                print(f"  ⏭ Already exists: {output_filename}")
            return output_path
        
        try:
            if availability['source'] == 'aws':
                # Download from AWS S3
                if self.verbose:
                    print(f"  ⬇ Downloading from AWS S3 ({availability['size_mb']:.1f} MB)...")
                
                # Download the gzipped file first
                temp_gz = self.output_dir / f"temp_{dt.strftime('%Y%m%d_%H%M')}.grib2.gz"
                temp_grib = self.output_dir / f"temp_{dt.strftime('%Y%m%d_%H%M')}.grib2"
                
                try:
                    # Download from S3
                    with self.s3_fs.open(availability['path'], 'rb') as f_in:
                        with open(temp_gz, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # Decompress gzip
                    if self.verbose:
                        print(f"  📦 Decompressing...")
                    with gzip.open(temp_gz, 'rb') as f_in:
                        with open(temp_grib, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # Load with xarray
                    ds = xr.open_dataset(temp_grib, engine='cfgrib')
                    
                    if subset_hawaii:
                        # Subset to Hawaii domain (though Hawaii bucket is already subset)
                        ds = self._subset_to_hawaii(ds)
                    
                    # Save as NetCDF
                    ds.to_netcdf(output_path)
                    ds.close()
                    
                finally:
                    # Clean up temp files including .idx files created by cfgrib
                    if temp_gz.exists():
                        temp_gz.unlink()
                    if temp_grib.exists():
                        temp_grib.unlink()
                    # Clean up index file if it exists
                    temp_idx = temp_grib.with_suffix('.grib2.idx')
                    if temp_idx.exists():
                        temp_idx.unlink()
                
            elif availability['source'] == 'iowa':
                # Download from Iowa State
                if self.verbose:
                    print(f"  ⬇ Downloading from Iowa State ({availability['size_mb']:.1f} MB)...")
                
                response = self.session.get(availability['url'], stream=True, timeout=30)
                response.raise_for_status()
                
                # Save to temporary GRIB2 file
                temp_grib = self.output_dir / f"temp_{dt.strftime('%Y%m%d_%H%M')}.grib2"
                
                with open(temp_grib, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Load with xarray and process
                ds = xr.open_dataset(temp_grib, engine='cfgrib')
                
                if subset_hawaii:
                    ds = self._subset_to_hawaii(ds)
                
                # Save as NetCDF
                ds.to_netcdf(output_path)
                ds.close()
                
                # Clean up temp files including .idx file created by cfgrib
                temp_grib.unlink()
                temp_idx = temp_grib.with_suffix('.grib2.idx')
                if temp_idx.exists():
                    temp_idx.unlink()
            
            if self.verbose:
                size_mb = output_path.stat().st_size / (1024**2)
                print(f"  ✓ Saved: {output_filename} ({size_mb:.1f} MB)")
            
            return output_path
            
        except Exception as e:
            print(f"  ❌ Error downloading {dt}: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return None
    
    def _subset_to_hawaii(self, ds):
        """
        Subset xarray dataset to Hawaii domain.
        
        Args:
            ds (xarray.Dataset): Input dataset
            
        Returns:
            xarray.Dataset: Subset dataset
        """
        try:
            # Get latitude and longitude
            if 'latitude' in ds.coords:
                lat = ds.latitude
                lon = ds.longitude
            elif 'lat' in ds.coords:
                lat = ds.lat
                lon = ds.lon
            else:
                # Try to extract from coordinates
                if 'y' in ds.coords and 'x' in ds.coords:
                    # May need projection info - skip subsetting for now
                    warnings.warn("Could not identify lat/lon coordinates for subsetting")
                    return ds
            
            # Convert longitude to -180 to 180 if needed
            if lon.max() > 180:
                lon = ((lon + 180) % 360) - 180
            
            # Create mask for Hawaii region
            lat_mask = (lat >= self.HAWAII_BOUNDS['lat_min']) & (lat <= self.HAWAII_BOUNDS['lat_max'])
            lon_mask = (lon >= self.HAWAII_BOUNDS['lon_min']) & (lon <= self.HAWAII_BOUNDS['lon_max'])
            
            # Apply mask
            mask = lat_mask & lon_mask
            
            # Subset the dataset
            if mask.any():
                ds_subset = ds.where(mask, drop=True)
                return ds_subset
            else:
                warnings.warn("No data points found within Hawaii bounds")
                return ds
                
        except Exception as e:
            warnings.warn(f"Could not subset to Hawaii: {e}")
            return ds
    
    def download_time_range(self, start_time, end_time, interval_minutes=60, 
                           product='MultiSensor_QPE_01H_Pass2', subset_hawaii=True):
        """
        Download MRMS data for a time range.
        
        Args:
            start_time (datetime): Start of time range
            end_time (datetime): End of time range
            interval_minutes (int): Time interval between files (default: 60 for hourly)
            product (str): MRMS product name
            subset_hawaii (bool): Subset to Hawaii domain
            
        Returns:
            list: Paths to downloaded files
        """
        print(f"\n{'='*70}")
        print(f"Downloading MRMS {product}")
        print(f"Time range: {start_time} to {end_time}")
        print(f"Interval: {interval_minutes} minutes")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*70}\n")
        
        # Generate list of datetimes
        current_time = start_time
        datetimes = []
        while current_time <= end_time:
            datetimes.append(current_time)
            current_time += timedelta(minutes=interval_minutes)
        
        print(f"Total files to download: {len(datetimes)}\n")
        
        # Download each file
        downloaded_files = []
        failed_files = []
        
        for dt in tqdm(datetimes, desc="Downloading MRMS files"):
            result = self.download_file(dt, product=product, subset_hawaii=subset_hawaii)
            if result:
                downloaded_files.append(result)
            else:
                failed_files.append(dt)
        
        # Summary
        print(f"\n{'='*70}")
        print("DOWNLOAD SUMMARY")
        print(f"{'='*70}\n")
        print(f"Successfully downloaded: {len(downloaded_files)} files")
        print(f"Failed: {len(failed_files)} files")
        
        if failed_files and self.verbose:
            print(f"\nFailed downloads:")
            for dt in failed_files[:10]:  # Show first 10
                print(f"  - {dt}")
            if len(failed_files) > 10:
                print(f"  ... and {len(failed_files) - 10} more")
        
        # Clean up any temporary files
        print()
        self.cleanup_temp_files()
        
        print(f"\n{'='*70}\n")
        
        return downloaded_files


def main():
    """Main entry point for the MRMS download script."""
    parser = argparse.ArgumentParser(
        description="Download MRMS 1-hour QPE data for Hawaii domain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available products in Hawaii bucket
  python download_mrms_hawaii.py --list-products
  
  # List available dates for a product
  python download_mrms_hawaii.py --list-dates
  
  # Download for a specific day (March 2, 2026)
  python download_mrms_hawaii.py --start "2026-03-02 00:00" --end "2026-03-02 23:59"
  
  # Download for multiple days
  python download_mrms_hawaii.py --start "2026-03-01" --end "2026-03-07"
  
  # Download with custom output directory
  python download_mrms_hawaii.py --start "2026-03-02" --end "2026-03-02" --output ../data/mrms
  
  # Download without subsetting to Hawaii (full CONUS)
  python download_mrms_hawaii.py --start "2026-03-02" --end "2026-03-02" --no-subset
        """
    )
    
    parser.add_argument(
        '--start',
        type=str,
        required=False,
        help='Start datetime in YYYY-MM-DD or "YYYY-MM-DD HH:MM" format'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        required=False,
        help='End datetime in YYYY-MM-DD or "YYYY-MM-DD HH:MM" format'
    )
    
    parser.add_argument(
        '--list-products',
        action='store_true',
        help='List available MRMS products in Hawaii bucket and exit'
    )
    
    parser.add_argument(
        '--list-dates',
        action='store_true',
        help='List available dates for the specified product and exit'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Time interval in minutes between files (default: 60 for hourly)'
    )
    
    parser.add_argument(
        '--product',
        type=str,
        default='MultiSensor_QPE_01H_Pass2',
        choices=['MultiSensor_QPE_01H_Pass2', 'MultiSensor_QPE_01H_Pass1'],
        help='MRMS product (default: MultiSensor_QPE_01H_Pass2 - quality controlled)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: data/mrms_hawaii/)'
    )
    
    parser.add_argument(
        '--no-subset',
        action='store_true',
        help='Download full CONUS data without subsetting to Hawaii'
    )
    
    # Initialize downloader
    downloader = MRMSHawaiiDownloader(
        output_dir=args.output,
        verbose=args.verbose
    )
    
    # Handle list operations
    if args.list_products:
        downloader.list_available_products()
        sys.exit(0)
    
    if args.list_dates:
        downloader.list_available_dates(product=args.product)
        sys.exit(0)
    
    # Validate required arguments for download
    if not args.start or not args.end:
        print("Error: --start and --end are required for downloading data")
        print("Use --list-products or --list-dates to explore available data")
        sys.exit(1)
    
    #   action='store_true',
        help='Download full CONUS data without subsetting to Hawaii'
    
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Parse datetimes
    try:
        # Try full datetime format first
        try:
            start_time = datetime.strptime(args.start, '%Y-%m-%d %H:%M')
        except ValueError:
            # Fall back to date only
            start_time = datetime.strptime(args.start, '%Y-%m-%d')
        
        try:
            end_time = datetime.strptime(args.end, '%Y-%m-%d %H:%M')
        except ValueError:
            end_time = datetime.strptime(args.end, '%Y-%m-%d')
            # Set to end of day
            end_time = end_time.replace(hour=23, minute=59)
            
    except ValueError as e:
        print(f"Error: Invalid date format. Use YYYY-MM-DD or 'YYYY-MM-DD HH:MM'")
        print(f"Details: {e}")
        sys.exit(1)
    
    # Validate time range
    if end_time < start_time:
        print("Error: End time must be after start time")
        sys.exit(1)
    
    # Download data
    downloaded_files = downloader.download_time_range(
        start_time=start_time,
        end_time=end_time,
        interval_minutes=args.interval,
        product=args.product,
        subset_hawaii=not args.no_subset  # Note: Hawaii bucket already subset, but option kept for flexibility
    )
    
    if downloaded_files:
        print(f"✓ Download complete! Files saved to: {downloader.output_dir}")
    else:
        print("❌ No files were successfully downloaded")
        sys.exit(1)


if __name__ == '__main__':
    main()
