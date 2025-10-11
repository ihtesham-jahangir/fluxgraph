import os

def save_tree(startpath=".", outfile="directory_tree.txt", skip_folders=None):
    if skip_folders is None:
        skip_folders = {"myenv", ".git", "__pycache__"}
 # folders to skip

    with open(outfile, "w", encoding="utf-8") as f:
        for root, dirs, files in os.walk(startpath):
            # Skip any folder in the skip list
            dirs[:] = [d for d in dirs if d not in skip_folders]

            level = root.replace(startpath, "").count(os.sep)
            indent = " " * 4 * level
            f.write(f"{indent}{os.path.basename(root)}/\n")

            subindent = " " * 4 * (level + 1)
            for file in files:
                f.write(f"{subindent}{file}\n")

save_tree(".")
