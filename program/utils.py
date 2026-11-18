import csv
import os
import re
import subprocess
import sys
import threading
import time
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
import requests
import urllib3

from config import IMAGE_EXTENSIONS, LINK_COLUMNS

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

URL_RE = re.compile(r"https?://[^\s;,|]+")
LINK_RE = re.compile(r"(?:https?://|ftp://|doi:)[^\s;,|]+")


class ItemSkipped(Exception):
    pass


class TaskControl:
    def __init__(self):
        self.cancel_event = threading.Event()
        self.skip_event = threading.Event()
        self.skip_site = False  # the pending skip covers the whole website, not just this item
        self.skipped_hosts = set()
        # For tasks that work on several websites at the same time (image downloads), see ItemControl:
        self.skip_generation = 0    # goes up with every Skip, so each item in progress can tell it was skipped
        self.skip_requests = set()  # websites to skip, asked for by name
        self.active = {}            # website -> the item being downloaded from it now, for the Skip menu

    def reset(self):
        self.cancel_event.clear()
        self.skip_event.clear()
        self.skip_site = False
        self.skipped_hosts = set()
        self.skip_requests = set()
        self.active = {}

    def new_item(self):
        self.skip_event.clear()
        self.skip_site = False

    def request_skip(self, site=False, host=None):
        """Skip what is in progress. With `host`, skip that one website and leave the others running."""
        if host:
            self.skip_requests.add(host)
            return
        self.skip_site = site
        self.skip_generation += 1
        self.skip_event.set()

    @property
    def cancelled(self):
        return self.cancel_event.is_set()

    def check(self):
        if self.skip_event.is_set() or self.cancel_event.is_set():
            raise ItemSkipped()

    def sleep(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.check()
            time.sleep(min(0.25, max(end - time.monotonic(), 0)))


class ItemControl:
    """The task's controls as one of several lanes working side by side sees them.

    TaskControl has one "skip" switch, which the next item clears; with several items in progress that would
    skip some and miss others. Each lane therefore watches the skip counter from the moment its own item began:
    Skip skips every item in progress once, Cancel stops all lanes, and a website named in skip_requests is
    skipped by the lane working on it alone.
    """

    def __init__(self, parent, host):
        self.parent, self.host = parent, host
        self.skip_site = False
        self.generation = parent.skip_generation

    def new_item(self):
        self.skip_site = False
        self.generation = self.parent.skip_generation

    @property
    def cancelled(self):
        return self.parent.cancelled

    @property
    def skipped_hosts(self):
        return self.parent.skipped_hosts

    def check(self):
        parent = self.parent
        if parent.cancel_event.is_set():
            raise ItemSkipped()
        if self.host in parent.skip_requests:
            self.skip_site = True
            raise ItemSkipped()
        if parent.skip_generation != self.generation:
            self.skip_site = parent.skip_site
            raise ItemSkipped()

    def sleep(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.check()
            time.sleep(min(0.25, max(end - time.monotonic(), 0)))


def check(control):
    if control:
        control.check()


def pause(seconds, control=None):
    if control:
        control.sleep(seconds)
    else:
        time.sleep(seconds)


def is_blank(value):
    if value is None:
        return True
    if not isinstance(value, str) and pd.isna(value):
        return True
    return str(value).strip() == ""


def parse_issue_list(raw):
    items = raw if isinstance(raw, (list, tuple, set)) else re.split(r"[,;\n]", str(raw or ""))
    cleaned = (str(item).strip().strip("\"'").strip() for item in items)
    return list(dict.fromkeys(item for item in cleaned if item))


def scan_issues(occurrence_file):
    try:
        df = pd.read_csv(occurrence_file, sep="\t", usecols=lambda c: c == "issue", quoting=csv.QUOTE_NONE, dtype=str)
    except (ValueError, pd.errors.EmptyDataError):
        return Counter()
    counts = Counter()
    for issues in issue_sets(df):
        counts.update(issues)
    return counts


def insert_after_gbif(df, name, values):
    df = df.drop(columns=name, errors="ignore").copy()
    position = df.columns.get_loc("gbifID") + 1 if "gbifID" in df.columns else 0
    df.insert(position, name, values)
    return df


def read_csv(path, log=print):
    try:
        return pd.read_csv(path, dtype=str, low_memory=False, encoding="utf-8-sig")
    except pd.errors.ParserError:
        log("Standard CSV parser failed; retrying with the Python engine.")
        return pd.read_csv(path, dtype=str, engine="python", on_bad_lines="warn", encoding="utf-8-sig")


def read_dwca(path):
    return pd.read_csv(path, sep="\t", low_memory=False, quoting=csv.QUOTE_NONE, dtype=str)


def save_csv(df, path):
    """Write a CSV in one step: to a temporary file beside it, which then replaces the old file. A crash, a full
    disk or a power cut part-way through therefore never leaves half a file where the data was (the measurements
    and the dataset are rewritten after every sheet and every 25 images of a long run)."""
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    try:
        df.to_csv(temp, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8-sig")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def first_url_series(df, columns):
    result = pd.Series(pd.NA, index=df.index, dtype="object")
    for column in columns:
        if column in df.columns:
            result = result.fillna(df[column].astype("object").str.extract(f"({URL_RE.pattern})", expand=False))
    return result.fillna("")


def find_media_url(row):
    for column in ("media", *LINK_COLUMNS):
        value = row.get(column)
        if isinstance(value, str):
            match = URL_RE.search(value)
            if match:
                return match.group()
    return None


def all_links(row):
    text = " ".join(str(v) for v in row.dropna())
    return ", ".join(dict.fromkeys(LINK_RE.findall(text)))


def issue_sets(df):
    if "issue" not in df.columns:
        return pd.Series([frozenset()] * len(df), index=df.index, dtype="object")
    return df["issue"].fillna("").map(lambda s: frozenset(filter(None, re.split(r"[;,\s]+", s))))


def request_with_backoff(url, max_retries=3, log=print, control=None, stream=False):
    from .image_download import fetch  # imported here: image_download imports this module

    delay = 5
    error = ""
    for attempt in range(1, max_retries + 1):
        check(control)
        try:
            resp = fetch(url, stream=stream, control=control, log=log)
        except requests.exceptions.RequestException as exc:
            error = str(exc)
        else:
            if resp.status_code != 429 and resp.status_code < 500:
                resp.raise_for_status()
                return resp
            error = f"HTTP {resp.status_code}"
            resp.close()
        if attempt < max_retries:
            log(f"    {error}; retrying in {delay} s")
            pause(delay, control)
            delay *= 2
    raise RuntimeError(f"Request failed after {max_retries} attempts: {error}")


def index_images(media_dir):
    media_dir = Path(media_dir)
    if not media_dir.is_dir():
        return {}
    images = {}
    for path in sorted(media_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.setdefault(path.stem.strip(), path)
    return images


def gbif_ids(df):
    if "gbifID" not in df.columns:
        raise ValueError("The CSV has no gbifID column")
    return df["gbifID"].fillna("").astype(str).str.strip()


def extract_archive(zip_path, extract_dir, log=print):
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    log(f"Extracted to {extract_dir}")
    return str(extract_dir)


def find_dwca_files(folder):
    folder = Path(folder)
    occurrence = list(folder.rglob("occurrence.txt"))
    multimedia = list(folder.rglob("multimedia.txt"))
    if not occurrence or not multimedia:
        raise FileNotFoundError(f"occurrence.txt or multimedia.txt not found in {folder}")
    for occ in occurrence:
        for mul in multimedia:
            if occ.parent == mul.parent:
                return str(occ), str(mul)
    return str(occurrence[0]), str(multimedia[0])


def open_path(path):
    path = Path(path)
    if not path.exists():
        if path.suffix:
            raise FileNotFoundError(f"{path} does not exist yet")
        path.mkdir(parents=True, exist_ok=True)
    if sys.platform.startswith("win"):
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])
