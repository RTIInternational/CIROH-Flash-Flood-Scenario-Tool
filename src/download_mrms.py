"""
Download MRMS (Multi-Radar/Multi-Sensor) 1-hour QPE data for any region domain.

MRMS provides gridded, multi-sensor precipitation estimates combining
radar, satellite, and gauge data. This script downloads 1-hour QPE
accumulation data for a specified time period and domain (e.g. Hawaii, CONUS, Alaska).
"""

import argparse
import sys
import os
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

from region_config import load_region_config, PROJECT_ROOT


class MRMSDownloader:
    """Download MRMS QPE data from AWS S3 or Iowa State archive."""

    def __init__(self, region_name="hawaii", output_dir=None, verbose=False):
        """
        Initialize MRMS downloader.

        Args:
            region_name (str): Region profile name from regions/<name>/config.json
            output_dir (Path or str): Output directory for downloaded files
            verbose (bool): Enable verbose output
        """
        self.verbose = verbose
        self.region = load_region_config(region_name)
        
        if output_dir:
            self.output_dir = Path(output_dir)
        elif self.region.mrms_dir:
            self.output_dir = self.region.mrms_dir
        else:
            self.output_dir = PROJECT_ROOT / "precip_data" / "MRMS"
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bounds = self.region.bounds
        self.aws_region = self.region.mrms_meta.get("aws_region", "HAWAII")

        # MRMS data sources
        self.sources = {
            "aws": "noaa-mrms-pds",
            "iowa": "https://mtarchive.geol.iastate.edu",
            "ncei": "https://www.ncei.noaa.gov/data/mrms",
        }

        # Initialize S3 filesystem for AWS access
        try:
            self.s3_fs = s3fs.S3FileSystem(anon=True)
            self.has_s3 = True
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not initialize S3 access: {e}")
            self.has_s3 = False

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "MRMS-QPE-Downloader/1.0"})

    def list_available_products(self):
        """List available MRMS products in the AWS bucket for this region."""
        if not self.has_s3:
            print("S3 access not available")
            return []

        try:
            reg_path = f"{self.sources['aws']}/{self.aws_region}"
            products = self.s3_fs.ls(reg_path)
            product_names = [p.split("/")[-1] for p in products]

            print(f"\nAvailable MRMS products in {self.aws_region}:")
            for prod in product_names:
                print(f"  - {prod}")
            print()
            return product_names
        except Exception as e:
            print(f"Error listing products: {e}")
            return []

    def list_available_dates(self, product="MultiSensor_QPE_01H_Pass2", limit=10):
        """List available dates for a specific product."""
        if not self.has_s3:
            print("S3 access not available")
            return []

        try:
            product_path = f"{self.sources['aws']}/{self.aws_region}/{product}_00.00"
            dates = self.s3_fs.ls(product_path)
            date_dirs = [d.split("/")[-1] for d in dates]

            print(f"\nAvailable dates for {product} (showing last {limit}):")
            for date_dir in sorted(date_dirs)[-limit:]:
                print(f"  - {date_dir}")
            print()
            return sorted(date_dirs)
        except Exception as e:
            print(f"Error listing dates: {e}")
            return []

    def cleanup_temp_files(self):
        """Clean up temporary files (.idx, temp_*.grib2, temp_*.grib2.gz)."""
        patterns = ["*.idx", "temp_*.grib2", "temp_*.grib2.gz"]
        total_cleaned = 0

        for pattern in patterns:
            files = list(self.output_dir.glob(pattern))
            for f in files:
                try:
                    f.unlink()
                    total_cleaned += 1
                except Exception:
                    pass

        if total_cleaned > 0 and self.verbose:
            print(f"✓ Cleaned up {total_cleaned} temporary files")
        return total_cleaned

    def get_mrms_filename(self, dt, product="MultiSensor_QPE_01H_Pass2"):
        timestamp = dt.strftime("%Y%m%d-%H%M%S")
        return f"MRMS_{product}_00.00_{timestamp}.grib2.gz"

    def get_aws_path(self, dt, product="MultiSensor_QPE_01H_Pass2"):
        date_str = dt.strftime("%Y%m%d")
        filename = self.get_mrms_filename(dt, product)
        product_with_suffix = f"{product}_00.00"
        return f"{self.sources['aws']}/{self.aws_region}/{product_with_suffix}/{date_str}/{filename}"

    def get_iowa_url(self, dt, product="MultiSensor_QPE_01H_Pass2"):
        year = dt.strftime("%Y")
        month = dt.strftime("%m")
        day = dt.strftime("%d")
        filename = self.get_mrms_filename(dt, product)
        return f"{self.sources['iowa']}/{year}/{month}/{day}/mrms/ncep/{product}/{filename}"

    def check_file_availability(self, dt, product="MultiSensor_QPE_01H_Pass2"):
        if self.has_s3:
            try:
                s3_path = self.get_aws_path(dt, product)
                if self.s3_fs.exists(s3_path):
                    file_info = self.s3_fs.info(s3_path)
                    return {
                        "source": "aws",
                        "path": s3_path,
                        "size_mb": file_info.get("size", 0) / (1024**2),
                        "available": True,
                    }
            except Exception:
                pass

        try:
            iowa_url = self.get_iowa_url(dt, product)
            response = self.session.head(iowa_url, timeout=5)
            if response.status_code == 200:
                size_mb = int(response.headers.get("content-length", 0)) / (1024**2)
                return {
                    "source": "iowa",
                    "url": iowa_url,
                    "size_mb": size_mb,
                    "available": True,
                }
        except Exception:
            pass

        return {"available": False}

    def _subset_to_bounds(self, ds):
        """Subset xarray dataset to domain bounds."""
        try:
            if "latitude" in ds.coords:
                lat = ds.latitude
                lon = ds.longitude
            elif "lat" in ds.coords:
                lat = ds.lat
                lon = ds.lon
            else:
                return ds

            if lon.max() > 180:
                lon = ((lon + 180) % 360) - 180

            lat_mask = (lat >= self.bounds["lat_min"]) & (lat <= self.bounds["lat_max"])
            lon_mask = (lon >= self.bounds["lon_min"]) & (lon <= self.bounds["lon_max"])
            mask = lat_mask & lon_mask

            if mask.any():
                return ds.where(mask, drop=True)
            return ds
        except Exception:
            return ds

    def download_file(self, dt, product="MultiSensor_QPE_01H_Pass2", subset=True):
        """Download MRMS file for a specific datetime."""
        availability = self.check_file_availability(dt, product)
        if not availability["available"]:
            if self.verbose:
                print(f"  ❌ File not available for {dt}")
            return None

        reg_prefix = self.region.mrms_meta.get("file_prefix", f"mrms_qpe_1hr_{self.region.region_id}_")
        output_filename = f"{reg_prefix}{dt.strftime('%Y%m%d_%H%M')}.nc"
        output_path = self.output_dir / output_filename

        if output_path.exists():
            if self.verbose:
                print(f"  ⏭ Already exists: {output_filename}")
            return output_path

        try:
            if availability["source"] == "aws":
                temp_gz = self.output_dir / f"temp_{dt.strftime('%Y%m%d_%H%M')}.grib2.gz"
                temp_grib = self.output_dir / f"temp_{dt.strftime('%Y%m%d_%H%M')}.grib2"
                try:
                    with self.s3_fs.open(availability["path"], "rb") as f_in:
                        with open(temp_gz, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    with gzip.open(temp_gz, "rb") as f_in:
                        with open(temp_grib, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)

                    ds = xr.open_dataset(temp_grib, engine="cfgrib")
                    if subset:
                        ds = self._subset_to_bounds(ds)
                    ds.to_netcdf(output_path)
                    ds.close()
                finally:
                    if temp_gz.exists():
                        temp_gz.unlink()
                    if temp_grib.exists():
                        temp_grib.unlink()
                    temp_idx = temp_grib.with_suffix(".grib2.idx")
                    if temp_idx.exists():
                        temp_idx.unlink()

            elif availability["source"] == "iowa":
                response = self.session.get(availability["url"], stream=True, timeout=30)
                response.raise_for_status()
                temp_grib = self.output_dir / f"temp_{dt.strftime('%Y%m%d_%H%M')}.grib2"
                with open(temp_grib, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                ds = xr.open_dataset(temp_grib, engine="cfgrib")
                if subset:
                    ds = self._subset_to_bounds(ds)
                ds.to_netcdf(output_path)
                ds.close()

                temp_grib.unlink()
                temp_idx = temp_grib.with_suffix(".grib2.idx")
                if temp_idx.exists():
                    temp_idx.unlink()

            if self.verbose:
                size_mb = output_path.stat().st_size / (1024**2)
                print(f"  ✓ Saved: {output_filename} ({size_mb:.1f} MB)")
            return output_path

        except Exception as e:
            print(f"  ❌ Error downloading {dt}: {e}")
            return None

    def download_range(self, start_date, end_date, product="MultiSensor_QPE_01H_Pass2"):
        """Download MRMS files for a date range."""
        if isinstance(start_date, str):
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            start_dt = start_date

        if isinstance(end_date, str):
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23)
        else:
            end_dt = end_date

        current_dt = start_dt
        dates_to_download = []
        while current_dt <= end_dt:
            dates_to_download.append(current_dt)
            current_dt += timedelta(hours=1)

        downloaded_files = []
        print(f"Downloading {len(dates_to_download)} files for {self.region.name} ({self.aws_region})...")
        for dt in tqdm(dates_to_download, desc="MRMS Progress"):
            fpath = self.download_file(dt, product=product)
            if fpath:
                downloaded_files.append(fpath)

        self.cleanup_temp_files()
        return downloaded_files


# Backward compatibility alias
MRMSHawaiiDownloader = MRMSDownloader


def main():
    parser = argparse.ArgumentParser(description="Download MRMS 1-hour QPE data")
    parser.add_argument("--region", default=os.environ.get("ACTIVE_REGION", "hawaii"), help="Region profile name")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", help="Output directory path (optional)")
    parser.add_argument("--product", default="MultiSensor_QPE_01H_Pass2", help="MRMS product name")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    downloader = MRMSDownloader(region_name=args.region, output_dir=args.output, verbose=args.verbose)
    downloader.download_range(args.start, args.end, product=args.product)


if __name__ == "__main__":
    main()
