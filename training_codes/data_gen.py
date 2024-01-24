import os
from PIL import Image
import shutil

def copy_large_images(src_folder, dst_folder, min_size=256):
    """
    Copies images with both width and height greater than min_size from src_folder to dst_folder.

    :param src_folder: Path to the source folder.
    :param dst_folder: Path to the destination folder.
    :param min_size: Minimum size of image dimensions.
    """
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    cnt=0
    for filename in os.listdir(src_folder):
        file_path = os.path.join(src_folder, filename)

        try:
            with Image.open(file_path) as img:
                width, height = img.size

                if width >= min_size and height >= min_size:
                    shutil.copy(file_path, os.path.join(dst_folder, filename))
                    cnt=cnt+1

                    if cnt % 1000==0:
                        print('Successfully delt with %d imgs'%cnt)

        except IOError:
            pass
    print('Selected %d imgs from origin %d imgs'%(cnt,len(os.listdir(src_folder))))

# Example usage
source_folder = 'path/to/your/source/folder'
destination_folder = 'path/to/your/destination/folder'
copy_large_images(source_folder, destination_folder)
