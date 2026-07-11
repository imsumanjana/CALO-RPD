"""Archive arbitrary reproducibility artifacts."""
from pathlib import Path
import zipfile
def create_bundle(paths,destination):
    dest=Path(destination)
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as zf:
        for p in map(Path,paths):zf.write(p,p.name)
    return dest
