# CALO-RPD Studio 1.0.1

This maintenance release corrects two issues observed on current Windows environments:

1. PYPOWER cross-validation compatibility with NumPy 2.4. The validator now imports only the required PYPOWER modules and the dependency range constrains NumPy below 2.4 for complete PYPOWER compatibility.
2. Theme contrast. The application now applies deterministic light/dark Qt palettes and explicit text colors for all major widgets so text remains readable independently of the Windows system palette.

After updating an existing editable installation, run:

```powershell
python -m pip install --upgrade "numpy<2.4"
python -m pip install -e .
python main.py
```
