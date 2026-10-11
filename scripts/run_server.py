"""Dev server runner for TerrainLens.

Translates CLI arguments (e.g. `--port 3000` from npm/supervisor) into Streamlit
configuration flags (`--server.port=3000`) and guarantees the repository root is
on PYTHONPATH.
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    args = sys.argv[1:]
    port = "3000"
    host = "0.0.0.0"

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--port" and i + 1 < len(args):
            port = str(args[i + 1])
            i += 2
        elif arg.startswith("--port="):
            port = str(arg.split("=", 1)[1])
            i += 1
        elif arg == "--host" and i + 1 < len(args):
            host = str(args[i + 1])
            i += 2
        elif arg.startswith("--host="):
            host = str(arg.split("=", 1)[1])
            i += 1
        else:
            i += 1

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{repo_root}:{existing_pythonpath}" if existing_pythonpath else repo_root
    )

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        os.path.join(repo_root, "app", "dashboard.py"),
        f"--server.port={port}",
        f"--server.address={host}",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--browser.gatherUsageStats=false",
    ]

    # Replace current process with streamlit process
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
