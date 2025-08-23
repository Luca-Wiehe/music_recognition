from tqdm import tqdm
import sys

def create_tqdm_bar(iterable, desc):
    return tqdm(enumerate(iterable), 
                total=len(iterable), 
                desc=desc,
                ncols=100,
                ascii=True,  # Use ASCII characters for better compatibility
                file=sys.stdout,  # Explicitly use stdout
                leave=False,  # Don't leave progress bar when done
                dynamic_ncols=True,  # Auto-adjust width
                position=0,  # Use position 0 to avoid multiple bars
                miniters=1,  # Update every iteration
                mininterval=0.1)  # Update at least every 0.1 seconds
