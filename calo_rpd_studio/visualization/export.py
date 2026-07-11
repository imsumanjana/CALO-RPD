"""Figure export wrapper."""
def export_figure(figure,path,dpi=300,transparent=False,tight=True):figure.savefig(path,dpi=dpi,transparent=transparent,bbox_inches='tight' if tight else None);return path
