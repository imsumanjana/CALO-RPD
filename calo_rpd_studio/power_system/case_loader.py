"""Load validated MATPOWER/PYPOWER-compatible cases."""
from __future__ import annotations
import importlib, json
from pathlib import Path
from .case_model import PowerSystemCase

class CaseLoader:
    STANDARD={'case30':'pypower.case30','case57':'pypower.case57','case118':'pypower.case118'}
    @classmethod
    def load(cls,source:str|Path)->PowerSystemCase:
        name=str(source)
        if name in cls.STANDARD:
            try:
                module=importlib.import_module(cls.STANDARD[name]); data=getattr(module,name)()
            except ModuleNotFoundError as exc:
                raise RuntimeError('PYPOWER is required to load bundled IEEE benchmark cases.') from exc
            return PowerSystemCase.from_dict(data,name=name)
        path=Path(source)
        if not path.exists(): raise FileNotFoundError(path)
        if path.suffix.lower()=='.json': return PowerSystemCase.from_dict(json.loads(path.read_text(encoding='utf-8')),name=path.stem)
        raise ValueError('Custom cases must use the documented JSON MATPOWER-compatible format.')
    @classmethod
    def available_cases(cls): return tuple(cls.STANDARD)
