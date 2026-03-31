from __future__ import annotations

import sys

from aisci_core.runner import JobRunner
from aisci_core.models import JobStatus


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: python -m aisci_app.worker_main <job_id>")
        return 2
    status = JobRunner().run_job(args[0])
    return 0 if status == JobStatus.SUCCEEDED else 1


if __name__ == "__main__":
    raise SystemExit(main())
