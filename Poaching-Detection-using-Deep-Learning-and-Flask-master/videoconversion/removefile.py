import os

# Prompt user for directory path
dir_path = input("Enter directory path: ")

# Check if directory exists
if not os.path.isdir(dir_path):
    print(f"{dir_path} is not a valid directory path.")
    exit()

# Get list of files in directory
files = os.listdir(dir_path)

# Delete each file
for file in files:
    file_path = os.path.join(dir_path, file)
    if os.path.isfile(file_path):
        os.remove(file_path)

print(f"All files in {dir_path} have been deleted.")
