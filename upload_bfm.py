#!/usr/bin/env python3
"""TUI for uploading paired BFM CSVs (real_imag + mag_phase) to Google Drive via rclone.

Sessions are paired by their trailing YYYYMMDD_HHMMSS timestamp, so selecting one
row always uploads both versions together.

    python upload_bfm.py                      # interactive picker
    python upload_bfm.py --dry-run            # show what rclone would do
    python upload_bfm.py --remote gdrive --dest "HAR/BFM"
"""
import argparse
import curses
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime

RI_DIR = "bfm_real_imag_csv"
MP_DIR = "bfm_mag_phase_csv"
TS_RE = re.compile(r"_(\d{8}_\d{6})\.csv$")
PREFIX_RE = re.compile(r"^bfm_(?:ri_|mp_)?data_")


@dataclass
class Session:
    ts: str
    ri: str | None = None
    mp: str | None = None
    ri_done: bool = False
    mp_done: bool = False
    selected: bool = False
    size: int = 0
    labels: set = field(default_factory=set)

    @property
    def complete(self):
        return self.ri is not None and self.mp is not None

    @property
    def uploaded(self):
        return (self.ri is None or self.ri_done) and (self.mp is None or self.mp_done)

    @property
    def label(self):
        # Labels can differ when one side was renamed; show all so it's noticed.
        return " | ".join(sorted(self.labels))

    @property
    def when(self):
        return datetime.strptime(self.ts, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")


def _label(name, ts):
    return PREFIX_RE.sub("", name)[: -len(f"_{ts}.csv")]


def scan_sessions(root="."):
    """Group CSVs from both dirs by timestamp; newest first."""
    sessions = {}
    for d, side in ((RI_DIR, "ri"), (MP_DIR, "mp")):
        path = os.path.join(root, d)
        if not os.path.isdir(path):
            continue
        for name in os.listdir(path):
            m = TS_RE.search(name)
            if not m:
                continue
            ts = m.group(1)
            s = sessions.setdefault(ts, Session(ts))
            if getattr(s, side) is not None:
                # Two files share a timestamp in one dir; keep both visible as separate rows.
                s = sessions.setdefault(f"{ts}#{name}", Session(ts))
            setattr(s, side, name)
            s.labels.add(_label(name, ts))
            s.size += os.path.getsize(os.path.join(path, name))
    return sorted(sessions.values(), key=lambda s: s.ts, reverse=True)


def remote_files(dest):
    """Filenames already in an rclone destination; empty if it doesn't exist yet."""
    r = subprocess.run(["rclone", "lsf", "--files-only", dest],
                       capture_output=True, text=True)
    return set(r.stdout.splitlines()) if r.returncode == 0 else set()


def mark_uploaded(sessions, ri_dest, mp_dest):
    ri_have, mp_have = remote_files(ri_dest), remote_files(mp_dest)
    for s in sessions:
        s.ri_done = s.ri in ri_have
        s.mp_done = s.mp in mp_have


def _fmt_size(n):
    for unit in ("B", "K", "M", "G"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.0f}T"


# ---------------------------------------------------------------- TUI

HELP = ("↑↓/jk move  space select  a all(visible)  n all-not-uploaded  c clear  "
        "/ filter  enter upload  q quit")


def _prompt(stdscr, row, text):
    curses.echo()
    curses.curs_set(1)
    stdscr.move(row, 0)
    stdscr.clrtoeol()
    stdscr.addstr(row, 0, text)
    s = stdscr.getstr(row, len(text), 200).decode(errors="ignore")
    curses.noecho()
    curses.curs_set(0)
    return s


def pick(stdscr, sessions, header):
    curses.curs_set(0)
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)   # uploaded
    curses.init_pair(2, curses.COLOR_YELLOW, -1)  # incomplete pair
    curses.init_pair(3, curses.COLOR_CYAN, -1)    # selected
    cur, top, flt = 0, 0, ""

    while True:
        view = [s for s in sessions if flt.lower() in (s.label + s.ts).lower()]
        cur = max(0, min(cur, len(view) - 1))
        h, w = stdscr.getmaxyx()
        body = max(1, h - 5)
        top = min(max(top, cur - body + 1), cur)

        stdscr.erase()
        nsel = sum(s.selected for s in sessions)
        stdscr.addnstr(0, 0, header, w - 1, curses.A_BOLD)
        stdscr.addnstr(1, 0, f"{len(view)} shown  {nsel} selected"
                       + (f"  filter: {flt!r}" if flt else ""), w - 1)
        stdscr.addnstr(2, 0, f"    {'timestamp':19}  RI MP  {'size':>5}  status    label",
                       w - 1, curses.A_UNDERLINE)
        for i, s in enumerate(view[top:top + body]):
            status = "uploaded" if s.uploaded else ("partial" if s.ri_done or s.mp_done else "")
            line = (f"[{'x' if s.selected else ' '}] {s.when}  "
                    f"{'✓' if s.ri else '-':2} {'✓' if s.mp else '-':2} "
                    f"{_fmt_size(s.size):>5}  {status:8}  {s.label}")
            attr = curses.color_pair(3) if s.selected else 0
            if not s.complete:
                attr |= curses.color_pair(2)
            elif s.uploaded:
                attr |= curses.color_pair(1)
            if top + i == cur:
                attr |= curses.A_REVERSE
            stdscr.addnstr(3 + i, 0, line, w - 1, attr)
        stdscr.addnstr(h - 2, 0, "yellow = only one version exists   green = already on Drive",
                       w - 1, curses.A_DIM)
        stdscr.addnstr(h - 1, 0, HELP, w - 1, curses.A_DIM)
        stdscr.refresh()

        k = stdscr.getch()
        if k in (ord("q"), 27):
            return []
        if k in (curses.KEY_DOWN, ord("j")):
            cur += 1
        elif k in (curses.KEY_UP, ord("k")):
            cur -= 1
        elif k == curses.KEY_NPAGE:
            cur += body
        elif k == curses.KEY_PPAGE:
            cur -= body
        elif k in (curses.KEY_HOME, ord("g")):
            cur = 0
        elif k in (curses.KEY_END, ord("G")):
            cur = len(view) - 1
        elif k == ord(" ") and view:
            view[cur].selected = not view[cur].selected
            cur += 1
        elif k == ord("a"):
            target = not all(s.selected for s in view)
            for s in view:
                s.selected = target
        elif k == ord("n"):
            for s in view:
                s.selected = not s.uploaded
        elif k == ord("c"):
            for s in sessions:
                s.selected = False
        elif k == ord("/"):
            flt = _prompt(stdscr, h - 1, "filter: ")
            cur = top = 0
        elif k in (curses.KEY_ENTER, 10, 13):
            chosen = [s for s in sessions if s.selected]
            if chosen:
                return chosen


# ---------------------------------------------------------------- upload

def rclone_copy(src_dir, names, dest, extra):
    if not names:
        return 0
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(names) + "\n")
        listfile = f.name
    try:
        # --files-from-raw: names contain spaces/parentheses, so no glob parsing.
        cmd = ["rclone", "copy", src_dir, dest, "--files-from-raw", listfile,
               "--progress", *extra]
        print("$", " ".join(cmd))
        return subprocess.run(cmd).returncode
    finally:
        os.unlink(listfile)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--remote", default=os.environ.get("BFM_RCLONE_REMOTE", "gdrive"),
                    help="rclone remote name (default: gdrive or $BFM_RCLONE_REMOTE)")
    ap.add_argument("--dest", default=os.environ.get("BFM_RCLONE_DEST", "BFM"),
                    help="base folder on the remote (default: BFM or $BFM_RCLONE_DEST)")
    ap.add_argument("--ri-folder", default="real_imag")
    ap.add_argument("--mp-folder", default="mag_phase")
    ap.add_argument("--dry-run", action="store_true", help="pass --dry-run to rclone")
    ap.add_argument("--offline", action="store_true",
                    help="skip checking which files are already on Drive")
    args = ap.parse_args()

    if not shutil.which("rclone"):
        sys.exit("rclone not found. Install it: sudo -v; curl https://rclone.org/install.sh | sudo bash")

    root = os.path.dirname(os.path.abspath(__file__))
    ri_dest = f"{args.remote}:{args.dest.strip('/')}/{args.ri_folder}"
    mp_dest = f"{args.remote}:{args.dest.strip('/')}/{args.mp_folder}"

    sessions = scan_sessions(root)
    if not sessions:
        sys.exit(f"No timestamped CSVs in {RI_DIR}/ or {MP_DIR}/")
    if not args.offline:
        print(f"Checking {ri_dest} and {mp_dest} ...")
        mark_uploaded(sessions, ri_dest, mp_dest)

    header = f"BFM → {ri_dest}  +  {mp_dest}" + ("   [DRY RUN]" if args.dry_run else "")
    chosen = curses.wrapper(pick, sessions, header)
    if not chosen:
        print("Nothing selected.")
        return

    incomplete = [s for s in chosen if not s.complete]
    print(f"\n{len(chosen)} session(s) selected:")
    for s in chosen:
        flag = "" if s.complete else ("   <-- RI ONLY" if s.ri else "   <-- MP ONLY")
        print(f"  {s.when}  {s.label}{flag}")
    if incomplete:
        print(f"\nWARNING: {len(incomplete)} session(s) have only one version.")
    if input("\nUpload? [y/N] ").strip().lower() != "y":
        print("Cancelled.")
        return

    extra = ["--dry-run"] if args.dry_run else []
    rc = rclone_copy(os.path.join(root, RI_DIR), [s.ri for s in chosen if s.ri], ri_dest, extra)
    rc |= rclone_copy(os.path.join(root, MP_DIR), [s.mp for s in chosen if s.mp], mp_dest, extra)
    print("Done." if rc == 0 else f"rclone reported errors (exit {rc}).")
    sys.exit(rc)


if __name__ == "__main__":
    main()
