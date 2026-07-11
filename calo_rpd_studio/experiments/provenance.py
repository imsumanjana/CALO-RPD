"""Machine and software provenance capture."""
from __future__ import annotations
import importlib.metadata as md,platform,subprocess
from pathlib import Path
import psutil
PACKAGES=('numpy','scipy','pandas','matplotlib','PyQt6','torch','PYPOWER','PyYAML')
def _git_commit():
    try:return subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True,cwd=Path.cwd()).strip()
    except Exception:return ''
def collect_provenance():
    versions={}
    for p in PACKAGES:
        try:versions[p]=md.version(p)
        except md.PackageNotFoundError:versions[p]='not-installed'
    return {'software_version':'1.0.3','git_commit':_git_commit(),'python_version':platform.python_version(),'platform':platform.platform(),'processor':platform.processor(),'cpu_count':psutil.cpu_count(logical=True),'memory_bytes':psutil.virtual_memory().total,'dependencies':versions}
