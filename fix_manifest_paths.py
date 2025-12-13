import os


def fix_imglist(file_path):
    # Create a backup just in case
    backup_path = file_path + ".bak"
    if not os.path.exists(backup_path):
        with open(file_path, 'r') as original:
            with open(backup_path, 'w') as backup:
                backup.write(original.read())
        print(f"Backup created: {backup_path}")

    # Read the original lines
    with open(file_path, 'r') as f:
        lines = f.readlines()

    fixed_lines = []
    prefix_to_remove = "images_largescale"

    count = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()

        # We assume the last part is the label, everything else is the path
        label = parts[-1]
        path = " ".join(parts[:-1])

        # Check and remove prefix (handling both / and \)
        if path.startswith(prefix_to_remove):
            # Remove the prefix
            path = path[len(prefix_to_remove):]

            # Remove leading slashes if they remain (e.g., \imagenet_ln -> imagenet_ln)
            if path.startswith(os.sep) or path.startswith("/") or path.startswith("\\"):
                path = path[1:]

            count += 1

        # Reconstruct the line
        fixed_lines.append(f"{path} {label}\n")

    # Write back to the original file
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)

    print(f"Fixed {count} paths in {file_path}")


if __name__ == "__main__":
    # You can add more files to this list if needed
    files_to_fix = ["data/benchmark_imglist/imagenet_ln/nuisance_stratified_manifest.txt"]

    for f in files_to_fix:
        if os.path.exists(f):
            fix_imglist(f)
        else:
            print(f"File not found: {f}")