"""Validate a standard case and cross-check its base power flow."""
import argparse
from calo_rpd_studio.power_system.case_loader import CaseLoader
from calo_rpd_studio.power_system.case_validation import validate_case
from calo_rpd_studio.power_system.ac_power_flow import run_ac_power_flow
from calo_rpd_studio.power_system.independent_validator import validate_against_pypower

def main():
    p=argparse.ArgumentParser();p.add_argument("--case",default="case30",choices=["case30","case57","case118"]);a=p.parse_args();case=CaseLoader.load(a.case);report=validate_case(case);print(f"Case structure valid: {report.valid}");pf=run_ac_power_flow(case);print(f"Power flow converged: {pf.converged}; loss={pf.total_loss_mw:.10g} MW; Q-limit rounds={pf.q_limit_rounds}");cross=validate_against_pypower(case,pf);print(cross);return 0 if report.valid and pf.converged and (not cross.available or cross.passed) else 1
if __name__=="__main__":raise SystemExit(main())
