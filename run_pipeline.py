"""Run the full pipeline: python run_pipeline.py [--snapshot FILE | --synthetic] [--quick]

Console output is also written to logs/pipeline.log.
"""
import sys
import traceback
from pathlib import Path

from options_engine.pipeline import main


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


if __name__ == "__main__":
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    with open(log_dir / "pipeline.log", "w", encoding="utf-8") as fh:
        sys.stdout = _Tee(sys.__stdout__, fh)
        sys.stderr = _Tee(sys.__stderr__, fh)
        code = 0
        try:
            main()
            print("PIPELINE_OK")
        except Exception:
            traceback.print_exc()
            print("PIPELINE_FAILED")
            code = 1
        finally:
            sys.stdout.flush()
            sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    sys.exit(code)
