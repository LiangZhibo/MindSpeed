"""Track process identity across delayed device-driver cleanup reports."""
from pathlib import Path


def start_time(pid):
    try:
        # Fields following the final ')' begin with field 3 (state).
        return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


class OwnershipTracker:
    def __init__(self, reader=start_time):
        self.reader = reader
        self.known = {}

    def foreign(self, active, currently_owned):
        for pid in currently_owned:
            self.known.setdefault(pid, self.reader(pid))
        foreign = set()
        for pid in active-currently_owned:
            if pid not in self.known:
                foreign.add(pid)
                continue
            now = self.reader(pid)
            # A driver may still report a PID after /proc and docker top stop
            # listing it. A reused live PID must have the same start tick.
            if now is not None and now != self.known[pid]:
                foreign.add(pid)
        return foreign
