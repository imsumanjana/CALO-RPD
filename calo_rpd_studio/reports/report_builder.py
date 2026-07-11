"""Plain-text scientific report assembly."""
from pathlib import Path
class ReportBuilder:
    def __init__(self,title):self.title=title;self.sections=[]
    def add_section(self,heading,content):self.sections.append((heading,content))
    def render(self):return self.title+'\n'+'='*len(self.title)+'\n\n'+'\n\n'.join(f'{h}\n{"-"*len(h)}\n{c}' for h,c in self.sections)
    def save(self,path):Path(path).write_text(self.render(),encoding='utf-8');return Path(path)
