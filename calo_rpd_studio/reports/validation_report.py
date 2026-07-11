"""Validation report formatter."""
import json
from .report_builder import ReportBuilder
def build_validation_report(record):b=ReportBuilder('CALO-RPD Validation Report');b.add_section('Independent validation',json.dumps(record,indent=2));return b
