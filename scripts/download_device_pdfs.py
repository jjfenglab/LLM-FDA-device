#!/usr/bin/env python3

import json
import asyncio
import aiohttp
import ssl
import random
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import logging
import pandas as pd


from utils.pdf_utils import count_pdf_pages

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FDADeviceDownloader:
    """Download FDA device PDF summaries directly from FDA website."""
    
    def __init__(
        self,
        request_delay_min: float = 0.2,
        request_delay_max: float = 0.5,
        max_workers: int = 1  # Keep single-threaded to be respectful
    ):
        self.request_delay_min = request_delay_min
        self.request_delay_max = request_delay_max
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def get_device_type_from_number(device_number: str) -> str:
        """Determine device type from device number format."""
        if device_number.startswith(('K', 'k')):
            return '510k'
        elif device_number.startswith(('DEN', 'den')):
            return 'de_novo'
        elif device_number.startswith(('P', 'p')):
            return 'pma'
        else:
            return 'unknown'

    @staticmethod
    def get_current_year() -> int:
        """Get current year's last two digits."""
        return datetime.now().year % 100

    @staticmethod
    def construct_510k_summary_link(k_number: str) -> str:
        """Construct 510(k) summary URL."""
        year_part = int(k_number[1:3])
        
        if year_part == 0 or year_part > FDADeviceDownloader.get_current_year():
            pdf_part = "pdf"
        else:
            pdf_part = f"pdf{year_part}"
        
        return f"https://www.accessdata.fda.gov/cdrh_docs/{pdf_part}/{k_number}.pdf"

    @staticmethod
    def construct_de_novo_summary_link(d_number: str) -> str:
        """Construct De Novo summary URL."""
        return f"https://www.accessdata.fda.gov/cdrh_docs/reviews/{d_number}.pdf"

    def get_device_url(self, device_number: str) -> str:
        """Get PDF URL for device number."""
        device_type = self.get_device_type_from_number(device_number)
        
        if device_type == '510k':
            return self.construct_510k_summary_link(device_number)
        elif device_type == 'de_novo':
            return self.construct_de_novo_summary_link(device_number)
        else:
            return None

def check_missing_pdfs(device_numbers, pdf_directory):
    """
    Check which device PDFs are missing locally
    
    Args:
        device_numbers (list): List of device numbers
        pdf_directory (Path): Directory where PDFs should be stored
    
    Returns:
        tuple: (missing_devices, existing_count)
    """
    missing_devices = []
    existing_count = 0
    
    for device_number in device_numbers:
        pdf_path = pdf_directory / f"{device_number}.pdf"
        if not pdf_path.exists() or pdf_path.stat().st_size == 0:
            missing_devices.append(device_number)
        else:
            existing_count += 1
    
    return missing_devices, existing_count

async def download_device_pdf(session: aiohttp.ClientSession, device_number: str, pdf_directory: Path, downloader: FDADeviceDownloader) -> dict:
    """
    Download a single device PDF from FDA website
    
    Args:
        session: aiohttp ClientSession
        device_number (str): Device number (K or DEN number)
        pdf_directory (Path): Local directory to save PDF
        downloader: FDADeviceDownloader instance
    
    Returns:
        dict: Result of download attempt
    """
    local_path = pdf_directory / f"{device_number}.pdf"
    
    try:
        url = downloader.get_device_url(device_number)
        if not url:
            return {
                'device_number': device_number,
                'status': 'unsupported',
                'error': 'Unsupported device type'
            }
        
        # Add browser-like headers to avoid rate limiting
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as response:
            # Check for redirects that might indicate rate limiting
            if response.status == 302:
                location = response.headers.get('Location', '')
                if 'abuse-detection' in location or 'apology' in location:
                    return {
                        'device_number': device_number,
                        'status': 'rate_limited',
                        'error': 'Rate limited by FDA server'
                    }
            
            if response.status == 404:
                return {
                    'device_number': device_number,
                    'status': 'not_found',
                    'error': 'PDF not available (404)'
                }
            
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    f.write(chunk)
            
            # Check if file was downloaded and has content
            if local_path.exists() and local_path.stat().st_size > 0:
                return {
                    'device_number': device_number,
                    'status': 'success',
                    'size': local_path.stat().st_size,
                    'url': url
                }
            else:
                return {
                    'device_number': device_number,
                    'status': 'error',
                    'error': 'Downloaded file is empty'
                }
                
    except Exception as e:
        return {
            'device_number': device_number,
            'status': 'error',
            'error': str(e)
        }

async def download_missing_device_pdfs(device_numbers_file, pdf_directory, max_workers=1):
    """
    Main function to download missing device PDFs from FDA website
    
    Args:
        device_numbers_file (str): Path to JSON file with device numbers
        pdf_directory (str): Directory to store PDF files
        max_workers (int): Maximum concurrent downloads (recommended: 1 to be respectful)
    """
    
    # Setup
    pdf_dir = Path(pdf_directory)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    # Read device numbers
    try:
        with open(device_numbers_file, 'r') as f:
            data = json.load(f)
            device_numbers = data.get('device_numbers', [])
    except Exception as e:
        logger.error(f"Failed to read device numbers file: {e}")
        return []
    
    if not device_numbers:
        logger.error("No device numbers found in input file")
        return []
    
    print(f"Found {len(device_numbers)} device numbers")
    
    # Check which PDFs are missing
    missing_devices, existing_count = check_missing_pdfs(device_numbers, pdf_dir)
    
    print(f"PDFs already downloaded: {existing_count}")
    print(f"PDFs missing: {len(missing_devices)}")
    
    if not missing_devices:
        print("✅ All device PDFs are already downloaded!")
        return []
    
    print(f"Starting download of {len(missing_devices)} missing PDFs...")
    print(f"Target directory: {pdf_dir}")
    print(f"Using single-threaded downloads with delays to be respectful to FDA servers")
    print("-" * 50)
    
    # Initialize downloader
    downloader = FDADeviceDownloader(
        request_delay_min=0.2,
        request_delay_max=0.5,
        max_workers=max_workers
    )
    
    all_results = []
    
    # Create session with settings for rate limiting avoidance
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    connector = aiohttp.TCPConnector(
        limit=1,  # Only 1 connection at a time
        limit_per_host=1,  # Only 1 connection per host
        keepalive_timeout=30,
        ssl=ssl_context  # Use custom SSL context that doesn't verify certificates
    )
    
    async with aiohttp.ClientSession(connector=connector) as session:
        # Download with progress bar
        with tqdm(total=len(missing_devices), desc="Downloading PDFs", unit="file") as pbar:
            
            for i, device_number in enumerate(missing_devices):
                # Download single device
                result = await download_device_pdf(session, device_number, pdf_dir, downloader)
                all_results.append(result)
                
                # Update progress bar
                pbar.update(1)
                
                # Update progress bar with status
                if result['status'] == 'success':
                    size_mb = result.get('size', 0) / (1024 * 1024)
                    pbar.set_postfix({
                        'status': 'downloaded', 
                        'device': device_number[:8],
                        'size': f'{size_mb:.1f}MB'
                    })
                elif result['status'] == 'not_found':
                    pbar.set_postfix({
                        'status': 'not_found', 
                        'device': device_number[:8]
                    })
                elif result['status'] == 'rate_limited':
                    pbar.set_postfix({
                        'status': 'rate_limited', 
                        'device': device_number[:8]
                    })
                    # Add extra delay after rate limiting
                    extra_delay = random.uniform(10, 20)
                    await asyncio.sleep(extra_delay)
                else:
                    pbar.set_postfix({
                        'status': 'error', 
                        'device': device_number[:8]
                    })
                
                # Respectful delay between requests
                if i < len(missing_devices) - 1:  # Don't delay after last request
                    delay = random.uniform(downloader.request_delay_min, downloader.request_delay_max)
                    await asyncio.sleep(delay)
    
    # Calculate statistics
    successful_downloads = sum(1 for r in all_results if r['status'] == 'success')
    not_found_count = sum(1 for r in all_results if r['status'] == 'not_found')
    error_count = sum(1 for r in all_results if r['status'] == 'error')
    rate_limited_count = sum(1 for r in all_results if r['status'] == 'rate_limited')
    
    # Calculate total size downloaded
    total_size = sum(r.get('size', 0) for r in all_results if r['status'] == 'success')
    total_size_mb = total_size / (1024 * 1024)
    
    print("\n" + "=" * 50)
    print("DOWNLOAD COMPLETE")
    print(f"Total PDFs attempted: {len(missing_devices)}")
    print(f"Successfully downloaded: {successful_downloads}")
    print(f"Not found (404): {not_found_count}")
    print(f"Rate limited: {rate_limited_count}")
    print(f"Download errors: {error_count}")
    print(f"Total size downloaded: {total_size_mb:.1f} MB")
    print(f"Final status: {existing_count + successful_downloads}/{len(device_numbers)} PDFs available")
    
    # Show not found devices if any
    if not_found_count > 0:
        print(f"\n⚠️  {not_found_count} PDFs not found on FDA website:")
        not_found_devices = [r['device_number'] for r in all_results if r['status'] == 'not_found']
        for device in not_found_devices[:10]:  # Show first 10
            print(f"   - {device}")
        if len(not_found_devices) > 10:
            print(f"   ... and {len(not_found_devices) - 10} more")
    
    # Show rate limited devices if any
    if rate_limited_count > 0:
        print(f"\n🚫 {rate_limited_count} requests were rate limited:")
        rate_limited_devices = [r['device_number'] for r in all_results if r['status'] == 'rate_limited']
        for device in rate_limited_devices[:10]:  # Show first 10
            print(f"   - {device}")
        if len(rate_limited_devices) > 10:
            print(f"   ... and {len(rate_limited_devices) - 10} more")
    
    # Show errors if any
    if error_count > 0:
        print(f"\n❌ {error_count} download errors:")
        error_devices = [(r['device_number'], r['error']) for r in all_results if r['status'] == 'error']
        for device, error in error_devices[:5]:  # Show first 5
            print(f"   - {device}: {error}")
        if len(error_devices) > 5:
            print(f"   ... and {len(error_devices) - 5} more")
    
    return all_results

def export_results_to_csv(device_numbers_file, csv_filename, pdf_directory):
    """Export all device PDF information to CSV file.
    
    Args:
        device_numbers_file (str): Path to JSON file with device numbers
        csv_filename (str): Output CSV filename
        pdf_directory (str): Directory where PDFs are stored
    """
    pdf_dir = Path(pdf_directory)
    downloader = FDADeviceDownloader()
    
    # Read device numbers from input file
    try:
        with open(device_numbers_file, 'r') as f:
            data = json.load(f)
            device_numbers = data.get('device_numbers', [])
    except Exception as e:
        logger.error(f"Failed to read device numbers file: {e}")
        return
    
    if not device_numbers:
        logger.error("No device numbers found in input file")
        return
    
    print(f"Processing {len(device_numbers)} device numbers for CSV export...")
    
    # Prepare data for CSV
    csv_data = []
    
    for device_number in device_numbers:
        pdf_path = pdf_dir / f"{device_number}.pdf"
        
        # Check if PDF exists and get basic info
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            # PDF exists - get file size and page count
            file_size = pdf_path.stat().st_size
            page_count = count_pdf_pages(str(pdf_path))
            status = 'existing'
            url = downloader.get_device_url(device_number) or ''
            error_message = ''
        else:
            # PDF doesn't exist
            file_size = -1
            page_count = -1
            status = 'not_found'
            url = downloader.get_device_url(device_number) or ''
            error_message = 'PDF file not found locally'
        
        csv_row = {
            'device_number': device_number,
            'status': status,
            'file_size_bytes': file_size,
            'page_count': page_count,
            'url': url,
            'error_message': error_message
        }
        csv_data.append(csv_row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_filename, index=False)
    
    # Print summary
    existing_count = len([row for row in csv_data if row['status'] == 'existing'])
    missing_count = len([row for row in csv_data if row['status'] == 'not_found'])
    
    print(f"Results exported to CSV: {csv_filename}")
    print(f"Total entries: {len(csv_data)}")
    print(f"PDFs found locally: {existing_count}")
    print(f"PDFs missing: {missing_count}")

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Download FDA device PDF summaries directly from FDA website",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-i", "--input",
        dest="device_numbers_file",
        default="data/aiml_device_numbers_071025.json",
        help="JSON file containing device numbers to download"
    )
    
    parser.add_argument(
        "-o", "--output",
        dest="pdf_directory", 
        default="data/raw/device_summaries",
        help="Directory to store downloaded PDF files"
    )
    
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=1,
        help="Maximum concurrent downloads (recommended: 1 to be respectful to FDA servers)"
    )
    
    parser.add_argument(
        "-c", "--csv",
        type=str,
        default="scripts/output/table1_device_summaries.csv",
        help="CSV file to save download results with page counts"
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.device_numbers_file).exists():
        print(f"Error: Device numbers file '{args.device_numbers_file}' not found.")
        return
    
    print(f"Input file: {args.device_numbers_file}")
    print(f"Output directory: {args.pdf_directory}")
    print(f"Max workers: {args.workers}")
    print("-" * 50)
    
    # Run the download
    try:
        results = asyncio.run(download_missing_device_pdfs(
            device_numbers_file=args.device_numbers_file,
            pdf_directory=args.pdf_directory,
            max_workers=args.workers
        ))
        
        # Export results to CSV
        export_results_to_csv(args.device_numbers_file, args.csv, args.pdf_directory)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")

if __name__ == "__main__":
    print("Device PDF Downloader from FDA Website")
    print("=" * 50)
    main() 