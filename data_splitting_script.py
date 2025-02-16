import os
import shutil
import random

def split_data(source_dir, dest_dir1, dest_dir2, split_ratio=0.5):
    if not os.path.exists(dest_dir1):
        os.makedirs(dest_dir1)
    if not os.path.exists(dest_dir2):
        os.makedirs(dest_dir2)

    for category in os.listdir(source_dir):
        category_path = os.path.join(source_dir, category)
        if os.path.isdir(category_path):
            files = os.listdir(category_path)
            random.shuffle(files)
            split_point = int(len(files) * split_ratio)
            files1 = files[:split_point]
            files2 = files[split_point:]

            category_dest_dir1 = os.path.join(dest_dir1, category)
            category_dest_dir2 = os.path.join(dest_dir2, category)
            if not os.path.exists(category_dest_dir1):
                os.makedirs(category_dest_dir1)
            if not os.path.exists(category_dest_dir2):
                os.makedirs(category_dest_dir2)

            for file in files1:
                shutil.copy(os.path.join(category_path, file), os.path.join(category_dest_dir1, file))
            for file in files2:
                shutil.copy(os.path.join(category_path, file), os.path.join(category_dest_dir2, file))

source_dir = '/Users/laksshaysehrawat/Desktop/test/data/train'
dest_dir1 = '/Users/laksshaysehrawat/Desktop/test/data/client1/train'
dest_dir2 = '/Users/laksshaysehrawat/Desktop/test/data/client2/train'
split_data(source_dir, dest_dir1, dest_dir2)

source_dir = '/Users/laksshaysehrawat/Desktop/test/data/valid'
dest_dir1 = '/Users/laksshaysehrawat/Desktop/test/data/client1/valid'
dest_dir2 = '/Users/laksshaysehrawat/Desktop/test/data/client2/valid'
split_data(source_dir, dest_dir1, dest_dir2)

source_dir = '/Users/laksshaysehrawat/Desktop/test/data/test'
dest_dir1 = '/Users/laksshaysehrawat/Desktop/test/data/client1/test'
dest_dir2 = '/Users/laksshaysehrawat/Desktop/test/data/client2/test'
split_data(source_dir, dest_dir1, dest_dir2)