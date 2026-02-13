import os
import sys
import codecs

# Set the standard output to use UTF-8 encoding
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

def generate_structure(root_dir, ignore_dirs, ignore_files, ignore_extensions):
    """
    Generates a string representing the directory structure, ignoring specified items.
    """
    structure = []
    for root, dirs, files in os.walk(root_dir, topdown=True):
        # Filter directories to ignore
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        level = root.replace(root_dir, '').count(os.sep)
        indent = '│   ' * (level)
        
        # Add the current directory to the structure
        structure.append(f"{indent}├───{os.path.basename(root)}/")

        sub_indent = '│   ' * (level + 1)
        
        # Filter and add files
        for f in sorted(files):
            if f not in ignore_files and not any(f.endswith(ext) for ext in ignore_extensions):
                structure.append(f"{sub_indent}├───{f}")
                
    return "\n".join(structure)

if __name__ == "__main__":
    # Adjusted path to point to the root from inside preprocessing_tools/
    target_dir = '../vChat' 
    
    # Define what to ignore
    dirs_to_ignore = {'.git', '__pycache__', 'node_modules', 'dist', 'build', '.venv', 'venv', 'data/prompts', 'data/responses'}
    
    files_to_ignore = {'.DS_Store', 'package-lock.json', 'yarn.lock'}
    
    extensions_to_ignore = {'.pyc', '.jpg', '.png', '.mp4', '.toon', '.vtt', '.json', '.wav', '.pk1', '.txt'}

    # Generate the structure
    try:
        print("File structure for vChat:")
        print(generate_structure(target_dir, dirs_to_ignore, files_to_ignore, extensions_to_ignore))
    except FileNotFoundError:
        print(f"Error: Directory not found.")
