"""Execute the notebook top to bottom using the active Python environment."""
from pathlib import Path
import json
import os
import sys
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(ROOT / "tmp/mpl")
os.environ["JUPYTER_RUNTIME_DIR"] = str(ROOT / "tmp/jupyter_runtime")
os.environ["IPYTHONDIR"] = str(ROOT / "tmp/ipython")
kernel_root = ROOT / "tmp/jupyter"
kernel_dir = kernel_root / "kernels/hse-project"
kernel_dir.mkdir(parents=True, exist_ok=True)
(kernel_dir / "kernel.json").write_text(json.dumps({"argv":[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"],"display_name":"HSE project","language":"python"}))
os.environ["JUPYTER_PATH"] = str(kernel_root)
notebook = nbformat.read(ROOT / "EDA.ipynb", as_version=4)
client = NotebookClient(notebook, timeout=2400, kernel_name="hse-project", resources={"metadata":{"path":str(ROOT)}})
client.execute()
nbformat.validate(notebook)
nbformat.write(notebook, ROOT / "EDA.ipynb")
# Temporary rendered preview for QA, not an extra deliverable.
html, _ = HTMLExporter(template_name="lab").from_notebook_node(notebook)
(ROOT / "tmp/EDA.html").write_text(html)
print(f"Executed {sum(c.cell_type=='code' for c in notebook.cells)} code cells without errors", flush=True)
