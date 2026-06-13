import os

# repo root = two levels up from this file (xtransfer/paths.py)
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# HHAR raw data lives under Data/ (symlinked locally; downloadable, see README)
data_root = os.path.join(_REPO, 'Data')
# few-shot meta split files (.pkl) shipped in-repo
target_meta_path = os.path.join(_REPO, 'dataloader', 'target_loader', 'filelists')


def get_target_paths(data_name):
    print(os.getcwd())
    data_path = os.path.join(data_root, data_name)
    meta_path = os.path.join(target_meta_path, data_name)
    meta_file_path = None
    for path in os.listdir(meta_path):
        file_path = os.path.join(meta_path, path)
        if os.path.isfile(file_path) and file_path.endswith('.pkl') and path[:-4].lower() == data_name.lower():
            meta_file_path = os.path.join(meta_path, path)
            break
    return data_path, meta_file_path
