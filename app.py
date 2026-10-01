"""Local test harness for the Football Team Creator QUBO solver.

Mirrors how the QCentroid platform calls qcentroid.py::run(), but runs on your
machine against the local fixture in input.json.

Static by design: it never needs editing to try a different scenario — change
the "data"/"solver_params" values in input.json instead. The platform does not
read this file or input.json; it always supplies its own input_data per job.

Usage:
    python app.py
"""

input_file_name = "input.json"

import json

with open(input_file_name) as f:
    input_file = json.load(f)

data = input_file["data"]
extra_arguments = input_file.get("extra_arguments", {})
solver_params = input_file.get("solver_params", {})

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

from qcentroid import run

result = run(data, solver_params, extra_arguments)

print(json.dumps(result, indent=2))
