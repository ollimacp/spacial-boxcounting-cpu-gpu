import os
import glob
from .api import boxcount_from_file


def batch_boxcount(input_folder, mode='spatial', hilbert=False, file_pattern='*.*'):
    """Process all files in a directory using boxcounting.

    Parameters:
        input_folder (str): Path to the folder containing files.
        mode (str): 'spatial' or 'single'.
        hilbert (bool): If True, apply Hilbert transform.
        file_pattern (str): Pattern to match files, default '*.*'.

    Returns:
        dict: Mapping from filename to boxcount result.
    """
    results = {}
    search_path = os.path.join(input_folder, file_pattern)
    file_list = glob.glob(search_path)

    for filepath in file_list:
        try:
            result = boxcount_from_file(filepath, mode=mode, hilbert=hilbert)
            results[os.path.basename(filepath)] = result
        except Exception as e:
            results[os.path.basename(filepath)] = f'Error: {e}'
    return results


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python -m spacial_boxcounting.batch <input_folder>')
    else:
        folder = sys.argv[1]
        res = batch_boxcount(folder)
        for fname, result in res.items():
            print(f'{fname}: {result}')
