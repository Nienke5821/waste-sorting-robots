import os
import shutil
import winsound


def reduce_classes(source_dir,target_dir):
    """Adjusts label files, these files are already in yolo format and wil stay in this format when applying this function.
       The function generates new label files containing only the selected classes, and combines them into one class  
       
    Args:
        source_dir (str): Path to the directory which contains the original label files `.txt`.
        target_dir (str): Path to the directory where the new label files will be saved `.txt`.

    """ 
    # Check if target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Classes to keep
    bottle = ["0", "1" , "2", "3", "4" , "5", "6", "7", "15", "16" , "17", "18", "19" , "20", "21", "23" , "24"]

    # Loop over each .txt file in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith(".txt"):
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, filename)

            with open(source_path, "r") as src, open(target_path, "w") as tgt:
                for line in src:
                    # Split after first space -> in line with yolo-format of label files
                    parts = line.split(maxsplit=1)  
                    if parts and parts[0] in bottle:
                        parts[0] = "0"  # Bottle class = 0
                    if parts and parts[0] not in bottle:
                        continue  
                    tgt.write(" ".join(parts))

    all_files = os.listdir(target_dir)

    # Store empty file names
    empty_files = []

    # Remove empty files if those are created as result of removing the class ≠ bottle
    n_files = 0
    for file in all_files:
        n_files += 1
        file_path = os.path.join(target_dir, file)
        if os.path.isfile(file_path) and os.path.getsize(file_path) == 0:
            empty_files.append(file)
    print(n_files)

    # Report all information about empty files
    n_empty_files = 0
    if empty_files:
        print("Empty files found:")
        for file in empty_files:
            n_empty_files += 1
            file_path = os.path.join(target_dir, file)
            os.remove(file_path)
            print(f"Deleted: {file}")    
        print("Number of removed empty files:", n_empty_files)
    else:
        print("No empty files found.")


# Define source and target directories
source_dir_test = "C:/...../Warp-D/test/labels"
target_dir_test = "C:/...../Warp-D_filter/test/labels"
reduce_classes(source_dir_test, target_dir_test)

# Make sound if done
winsound.MessageBeep(winsound.MB_OK)

