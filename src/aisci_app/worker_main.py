from __future__ import annotations

import sys

from aisci_core.runner import JobRunner


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: python -m aisci_app.worker_main <job_id>")
        return 2
    JobRunner().run_job(args[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

