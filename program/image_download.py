"""Download the specimen images listed in a dataset.

Includes per-site handling: saved sign-ins, browser-like retries and site-specific downloaders.
"""
import hashlib
import json
import os
import re
import queue
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse

import pandas as pd
import requests

from config import BROWSER_USER_AGENT, HEADERS, IMAGE_EXTENSIONS, SITE_LOGINS_FILE
from .utils import (ItemControl, ItemSkipped, TaskControl, check, find_media_url, gbif_ids, index_images, is_blank,
                    pause, read_csv, request_with_backoff, save_csv)

DENIED_STATUSES = (401, 403)
MODE_REQUESTS, MODE_BROWSER = "requests", "browser"
STRATEGIES = {
    "Automatic: plain request, then browser-like headers, then the browser": "auto",
    "Plain requests only (never open a browser)": "requests",
    "Always use the browser for sites that refuse access": "browser",
    "Ask me: stop and open Site logins": "ask",
}
PREFS = {"strategy": "auto", "challenge_wait": 180, "browser_delay": 2.0, "request_delay": 0.0,
         "parallel_sites": 4}   # websites downloaded from at the same time (one image at a time from each)
PLAIN_LANE, BROWSER_LANE = "plain", "browser"
SAVE_EVERY = 25                 # images between saves of the dataset during a run
_local = threading.local()      # plain_only: this thread is a plain lane and must not open the browser
_logins_lock = threading.RLock()  # site_logins.json is read, changed and written back by more than one lane
_last_request = {}
_gave_up = set()  # sites the browser could not get past in this run
BLOCK_MARKERS = re.compile(r"request rejected|just a moment|cf-challenge|captcha|access denied|attention required|"
                           r"verify you are (?:a )?human|please (?:log|sign) ?in", re.IGNORECASE)


class AccessDenied(Exception):
    """A site refused the request in a way a sign-in or a real browser may fix."""

    def __init__(self, url, status=None, reason=""):
        self.url, self.status, self.host = url, status, host_of(url)
        why = f"HTTP {status}" if status else (reason or "blocked")
        super().__init__(f"Access denied by {self.host} ({why}); sign in to this site from Images, Site logins")


def host_of(url):
    return (urlparse(url).hostname or "").lower()


def looks_blocked(html):
    return bool(BLOCK_MARKERS.search(html or ""))


def _domain_matches(host, domain):
    domain = domain.lstrip(".").lower()
    return host == domain or host.endswith("." + domain)


# ---------------------------------------------------------------- saved sign-ins

def load_logins():
    path = Path(SITE_LOGINS_FILE)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError):
        # never overwrite a file we could not read: keep it aside so the sign-ins in it are not lost
        try:
            path.replace(path.with_name(path.name + ".unreadable"))
        except OSError:
            pass
        return {}


def _write_logins(data):
    """Write atomically and keep the previous version as site_logins.json.bak."""
    path = Path(SITE_LOGINS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        os.chmod(temp, 0o600)
    except OSError:
        pass
    if path.exists():
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    temp.replace(path)
    reload_session()


def saved_hosts():
    """{host: when the sign-in was saved}"""
    return dict(load_logins().get("saved", {}))


def known_sites():
    """{host: when it last refused access}. Kept for good, so the list survives restarts and updates."""
    return dict(load_logins().get("refused", {}))


def site_reasons():
    """{host: why it is on the list}, for sites skipped by the user or given up on after repeated failures."""
    return dict(load_logins().get("reasons", {}))


def note_refused(host, reason=None):
    with _logins_lock:
        data = load_logins()
        data.setdefault("refused", {})[host] = datetime.now().isoformat(timespec="minutes")
        if reason:
            data.setdefault("reasons", {})[host] = reason
        else:
            data.get("reasons", {}).pop(host, None)
        _write_logins(data)


def site_mode(host):
    """How downloads from `host` are made: plain requests, or through the signed-in browser."""
    modes = load_logins().get("modes", {})
    best = max((h for h in modes if _domain_matches(host, h)), key=len, default=None)
    return modes.get(best, MODE_REQUESTS)


def set_site_mode(host, mode):
    with _logins_lock:
        data = load_logins()
        data.setdefault("modes", {})[host] = mode
        _write_logins(data)


def configure(**prefs):
    PREFS.update({k: v for k, v in prefs.items() if k in PREFS and v is not None})


def new_run():
    _gave_up.clear()


def throttle(host, mode, control=None):
    """Wait so that requests to one site keep a polite distance."""
    interval = PREFS["browser_delay"] if mode == MODE_BROWSER else PREFS["request_delay"]
    wait = _last_request.get(host, 0) + interval - time.monotonic()
    while wait > 0:
        if control:
            control.check()
        time.sleep(min(wait, 0.25))
        wait -= 0.25
    _last_request[host] = time.monotonic()


def save_login(host, cookies, user_agent=""):
    with _logins_lock:
        data = load_logins()
        known = {(c.get("name"), c.get("domain"), c.get("path")): c for c in data.get("cookies", [])}
        for cookie in cookies:
            known[(cookie.get("name"), cookie.get("domain"), cookie.get("path"))] = cookie
        data["cookies"] = list(known.values())
        data.setdefault("saved", {})[host] = datetime.now().isoformat(timespec="minutes")
        if user_agent:
            data["user_agent"] = user_agent
        _write_logins(data)


def forget_login(host):
    with _logins_lock:
        data = load_logins()
        data["cookies"] = [c for c in data.get("cookies", []) if not _domain_matches(host, c.get("domain", ""))
                           and not _domain_matches(c.get("domain", "").lstrip("."), host)]
        data.get("saved", {}).pop(host, None)
        _write_logins(data)


# ---------------------------------------------------------------- requests

_session = None


def reload_session():
    global _session
    _session = None


def session():
    global _session
    if _session is None:
        _session = requests.Session()
        for cookie in load_logins().get("cookies", []):
            if cookie.get("name") is not None:
                _session.cookies.set(cookie["name"], cookie.get("value", ""), domain=cookie.get("domain", ""),
                                     path=cookie.get("path", "/"))
    return _session


def _has_login(host):
    return any(_domain_matches(host, h) or _domain_matches(h, host) for h in saved_hosts())


def _browser_fetch(url, control, log):
    from . import browser_sign_in  # imported here: browser_sign_in imports this module
    return browser_sign_in.browser_fetch(url, control, log)


def fetch(url, stream=False, timeout=30, control=None, log=print):
    """GET `url`, escalating as far as the download preference allows:
    1. a plain request with any saved sign-in, retried once with browser-like headers;
    2. the same download through a visible browser, where a person can pass a check;
    3. raise AccessDenied, which lists the site under Site logins for a sign-in or a manual download.
    Other statuses are returned for the caller to judge."""
    host = host_of(url)
    strategy = PREFS["strategy"]
    if getattr(_local, "plain_only", False):
        # One of several lanes downloading at once. Only the browser lane may use the browser, so a refusal is
        # raised from here and the lane hands the site over (download_site).
        throttle(host, MODE_REQUESTS, control)
        return _plain_fetch(url, host, stream, timeout)
    if strategy == "browser" or site_mode(host) == MODE_BROWSER:
        return _through_browser(url, host, control, log)
    throttle(host, MODE_REQUESTS, control)
    try:
        return _plain_fetch(url, host, stream, timeout)
    except AccessDenied:
        if strategy != "auto":
            raise
        resp = _through_browser(url, host, control, log, fallback=True)
        set_site_mode(host, MODE_BROWSER)
        log(f"    {host} will be downloaded through the browser from now on")
        return resp


def _through_browser(url, host, control, log, fallback=False):
    from . import browser_sign_in  # imported here: browser_sign_in imports this module
    if host in _gave_up:
        raise AccessDenied(url, reason="the browser could not get past this site earlier in the run")
    if not browser_sign_in.available():
        raise AccessDenied(url, reason="a browser is needed; " + browser_sign_in.INSTALL_HINT)
    if fallback:
        log(f"    {host} refuses plain requests; trying through the browser")
    throttle(host, MODE_BROWSER, control)
    try:
        return browser_sign_in.browser_fetch(url, control, log)
    except (AccessDenied, ItemSkipped) as exc:
        if isinstance(exc, AccessDenied):
            _gave_up.add(host)
        raise
    except Exception as exc:  # no browser installed, profile in use, browser closed by the user, ...
        _gave_up.add(host)
        raise AccessDenied(url, reason=f"the browser could not be used: {str(exc).splitlines()[0][:120]}") from exc


SHORT_USER_AGENT = "GBIF-HerbariaPipeline/1.0"
# Some servers refuse by User-Agent text alone: oxalis.br.fgov.be answers 403 to anything but a bare product token
# (not to "Python requests", not even to a Chrome string). Keyed by host suffix.
SITE_USER_AGENTS = {"oxalis.br.fgov.be": SHORT_USER_AGENT}
_working_agent = {}  # host -> the User-Agent that was accepted, tried first from then on
CONNECT_TIMEOUT = 10  # seconds to wait for a server to answer at all; the read timeout is per call
_https_only = set()   # hosts whose http:// port did not answer while https:// did


def _secure(url):
    return "https://" + url[len("http://"):]


def _get(url, host, **kwargs):
    """One GET. Many links in GBIF are old http:// addresses of servers that now only answer on https:// (the
    http port times out), so a failed connection to an http:// address is tried again as https://, and the
    host is remembered so the rest of the run does not wait for the timeout again."""
    if url.startswith("http://") and host in _https_only:
        url = _secure(url)
    try:
        return session().get(url, **kwargs)
    except requests.ConnectionError:                      # refused, unreachable or timed out while connecting
        if not url.startswith("http://"):
            raise
        resp = session().get(_secure(url), **kwargs)
        _https_only.add(host)
        return resp


def _plain_fetch(url, host, stream, timeout):
    """Try the usual headers, then a bare User-Agent, then browser-like headers with a Referer."""
    saved = load_logins().get("user_agent") if _has_login(host) else None
    configured = next((ua for suffix, ua in SITE_USER_AGENTS.items() if _domain_matches(host, suffix)), None)
    referer = {"Referer": f"{urlparse(url).scheme}://{host}/", "Accept": "image/avif,image/webp,image/*,*/*;q=0.8"}
    attempts = [dict(HEADERS, **{"User-Agent": _working_agent.get(host) or configured or saved or HEADERS["User-Agent"]}),
                {"User-Agent": SHORT_USER_AGENT},
                {"User-Agent": saved or BROWSER_USER_AGENT, **referer}]
    seen, status = [], None
    for headers in attempts:
        if headers in seen:
            continue
        seen.append(headers)
        resp = _get(url, host, timeout=(CONNECT_TIMEOUT, timeout), verify=False, headers=headers, stream=stream)
        if resp.status_code not in DENIED_STATUSES:
            _working_agent[host] = headers["User-Agent"]
            return resp
        status = resp.status_code
        resp.close()
    raise AccessDenied(url, status)


# ---------------------------------------------------------------- site-specific downloaders

def download_special(url, stem, log=print, control=None, gbif_id=None):
    """Download `url` in a way that is specific to its site. Returns the saved file's Path, or None to let the
    general downloader handle the URL. `gbif_id` is the record the URL came from, for sites whose full-size image
    can only be found from the record's other links. Add a branch for each site that needs its own method.

    Most branches exist because the address GBIF holds is a reduced copy (640 to 2000 px) while the publisher
    serves the original somewhere else; each falls back to the general downloader when that does not work."""
    host = host_of(url)
    if host.endswith("oxalis.br.fgov.be"):
        return _download_meise(url, stem, log, control)
    if host == "images.mobot.org" and "/detailimages/" in url:
        return _download_tropicos(url, stem, log, control, gbif_id)
    if host == "object.jacq.org":
        return _download_jacq(url, stem, log, control)
    if host.startswith("jbrj-public") and host.endswith("amazonaws.com") and "source=DZI/" in url:
        return _download_jbrj(url, stem, log, control)
    if urlparse(url).path.rstrip("/").endswith("/manifest") or urlparse(url).path.endswith("manifest.json"):
        return _download_iiif(url, stem, log, control)
    if host == "drive.google.com":
        return _download_google_drive(url, stem, log, control)
    if host.endswith("nmnh.si.edu"):
        return _download_smithsonian(url, stem, log, control)
    if host.endswith("cria.org.br"):
        return _download_cria(url, stem, log, control)
    return None


IIIF_SIZES = ("max", "full", "!4000,4000", "!3000,3000", "!2000,2000", "!1000,1000")
IIIF_IMAGE_URL = re.compile(r"^(?P<base>.+?)/(?:full|square|\d+,\d+,\d+,\d+|pct:[^/]+)/[^/]+/[^/]+/[^/]+$")


def _try_get(url, control=None):
    """One plain request, no escalation. The response, or None for any refusal or error."""
    host = host_of(url)
    throttle(host, MODE_REQUESTS, control)
    try:
        resp = _plain_fetch(url, host, True, 30)
    except (AccessDenied, requests.RequestException):
        return None
    if resp.status_code >= 400:
        resp.close()
        return None
    return resp


def _iiif_image_base(url, control, log):
    """The IIIF image base URL (the part before /full/max/0/default.jpg) for a manifest or an image URL."""
    match = IIIF_IMAGE_URL.match(url)
    if match:
        return match.group("base")
    resp = _try_get(url, control)
    if resp is None:
        return None
    image = _extract_image_from_json(resp.text, log)
    match = image and IIIF_IMAGE_URL.match(image)
    if match:
        return match.group("base")
    try:  # some manifests name the image service rather than an image of a given size
        return _image_service_id(resp.json())
    except ValueError:
        return None


def _image_service_id(node):
    """The address of the first IIIF image service named in a manifest, or None."""
    if isinstance(node, list):
        return next(filter(None, map(_image_service_id, node)), None)
    if not isinstance(node, dict):
        return None
    services = node.get("service")
    for service in services if isinstance(services, list) else [services]:
        if isinstance(service, dict):
            kind = json.dumps([service.get(k) for k in ("@context", "profile", "type", "@type")])
            address = service.get("@id") or service.get("id")
            if address and ("iiif.io/api/image" in kind or "ImageService" in kind):
                return address.rstrip("/")
    return next(filter(None, map(_image_service_id, node.values())), None)


def _iiif_limits(info):
    """maxArea, maxWidth and maxHeight from an info.json: top-level in IIIF 3, inside the profile list in IIIF 2."""
    profile = info.get("profile")
    sources = [p for p in (profile if isinstance(profile, list) else []) if isinstance(p, dict)] + [info]
    return {k: int(source[k]) for source in sources for k in ("maxArea", "maxWidth", "maxHeight") if source.get(k)}


def _download_iiif(url, stem, log, control):
    """Download the IIIF image behind a manifest or an image URL at the largest size the server allows.
    Returns the saved Path or None."""
    base = _iiif_image_base(url, control, log)
    if not base:
        return None
    return _download_iiif_service(base, stem, log, control)


def _iiif_info(base, control):
    response = _try_get(base + "/info.json", control)
    if response is None:
        return None
    try:
        info = response.json()
    except ValueError:
        return None
    return info if isinstance(info, dict) and info.get("width") else None


def _download_iiif_service(base, stem, log, control, info=None):
    """Download the image of the IIIF image service at `base` at the largest size the server allows. A server limits
    what one request may return (info.json maxArea, maxWidth, maxHeight) and answers 403 to anything bigger, or
    scales the image down, so an image over the limit is fetched as full-resolution regions and joined.
    Returns the saved Path or None."""
    width, height, limit = 0, 0, {}
    info = info or _iiif_info(base, control) or {}
    try:
        width, height = int(info.get("width", 0)), int(info.get("height", 0))
        limit = _iiif_limits(info)
    except (ValueError, TypeError):
        info = {}

    if width and height and limit:
        max_w = limit.get("maxWidth", width)
        max_h = limit.get("maxHeight", limit.get("maxWidth", height)) if "maxHeight" in limit or "maxWidth" in limit \
            else height
        area = limit.get("maxArea", max_w * max_h)
        if width * height > area or width > max_w or height > max_h:
            edge = int(min(max_w, max_h, area ** 0.5, 4096))
            pieces = [(x, y, f"{base}/{x},{y},{min(edge, width - x)},{min(edge, height - y)}/"
                             f"{min(edge, width - x)},/0/default.jpg")
                      for y in range(0, height, edge) for x in range(0, width, edge)]
            try:
                path = _stitch(width, height, pieces, stem, log, control)
                log(f"    IIIF image joined from regions ({width} x {height})")
                return path
            except ItemSkipped:
                raise
            except Exception as exc:
                log(f"    Region download failed ({exc}); trying smaller sizes")

    sizes = list(IIIF_SIZES)
    if width and height and limit.get("maxArea"):
        scale = (limit["maxArea"] / (width * height)) ** 0.5
        sizes.insert(0, f"{int(width * scale)},")
    elif width:
        sizes.insert(2, f"{width},")
    # The simplest servers ("level 0") only serve the whole image at the sizes they list, and tiles.
    listed = sorted(((int(s["width"]), int(s["height"])) for s in info.get("sizes") or []
                     if isinstance(s, dict) and s.get("width") and s.get("height")), reverse=True)
    tiles = next((t for t in info.get("tiles") or [] if isinstance(t, dict) and t.get("width")
                  and 1 in (t.get("scaleFactors") or [])), None)
    if listed and "level0" in json.dumps(info.get("profile")):
        sizes = [f"{w},{h}" for w, h in listed[:2]]
        if tiles and width and height and min(listed[0]) < MIN_SHORT_SIDE:
            edge = int(tiles["width"])
            pieces = [(x, y, f"{base}/{x},{y},{min(edge, width - x)},{min(edge, height - y)}/"
                             f"{min(edge, width - x)},{min(edge, height - y)}/0/default.jpg")
                      for y in range(0, height, edge) for x in range(0, width, edge)]
            try:
                path = _stitch(width, height, pieces, stem, log, control)
                log(f"    IIIF image joined from tiles ({width} x {height})")
                return path
            except ItemSkipped:
                raise
            except Exception as exc:
                log(f"    Tile download failed ({exc}); trying the listed sizes")
    for size in sizes:
        resp = _try_get(f"{base}/full/{size}/0/default.jpg", control)
        if resp is None:
            log(f"    {host_of(base)} did not accept size {size}")
            continue
        path = _save_image(resp, stem, control)
        if path:
            log(f"    IIIF image at size {size}")
            return path
    return None  # the general downloader takes over, including its browser fallback


def _download_meise(url, stem, log, control):
    # Meise Botanic Garden (BR): the manifest loads but the image server can refuse the "max" size the manifest
    # points to, so ask the IIIF service for other sizes, largest first.
    return _download_iiif(url, stem, log, control)


TROPICOS_IIIF = "https://mbgimages.mobot.org/IIPServer/iipsrv.fcgi?IIIF="
# Where the originals of each volume in images.mobot.org detail-image addresses are kept on the Tropicos image server,
# as its viewer (tropicos.org/image/<image id>/viewer) opens them. The volume names do not always match.
TROPICOS_FOLDERS = {"TropicosImagesVol6": r"\\cbiimages03\TropicosImagesVol6",
                    "RISScanning01": r"\\cbiimages04\RISScanning01",
                    "TropicosImagesVol5": r"\\cbiimages02\TropicosImagesVol5",
                    "TropicosImageVol3": r"\\cbiimages02\TropicosImagesVol3",
                    "TropicosImages2": r"\\cbiimages02\TropicosImagesVol2"}
TROPICOS_SERVERS = [rf"\\cbiimages{n:02}" for n in range(1, 7)]  # tried for a volume missing from the table
_tropicos_found = {}  # volume missing from the table -> the folder that had its images, or None


def _tropicos_image_id(url, gbif_id, control):
    """The Tropicos image id of `url`, from the "references" link GBIF keeps with the record's images
    (tropicos.org/ImageFullView.aspx?imageid=...)."""
    resp = _try_get(f"https://api.gbif.org/v1/occurrence/{gbif_id}", control)
    try:
        media = resp.json().get("media", []) if resp is not None else []
    except (ValueError, AttributeError):
        return None
    path = urlparse(url).path
    for item in media:
        if isinstance(item, dict) and urlparse(str(item.get("identifier", ""))).path == path:
            found = re.search(r"imageid=(\d+)", str(item.get("references", "")), re.IGNORECASE)
            return found and found.group(1)
    return None


def _download_tropicos(url, stem, log, control, gbif_id):
    # Missouri Botanical Garden (Tropicos): images.mobot.org/.../detailimages/<volume>/<folder>/<GUID>.jpg is a
    # 640 px copy. The original scan is a JPEG 2000 named after the Tropicos image id on Tropicos's IIP image server,
    # which also speaks IIIF (at most 5000 px per request, so large scans are joined from regions).
    match = re.search(r"/detailimages/([^/]+)/([^/]+)/[^/]+$", urlparse(url).path)
    image_id = match and gbif_id and _tropicos_image_id(url, gbif_id, control)
    if not image_id:
        log("    Tropicos image id not found; downloading the 640 px copy")
        return None
    volume, folder = match.groups()
    if volume in TROPICOS_FOLDERS:
        places = [TROPICOS_FOLDERS[volume]]
    elif volume in _tropicos_found:
        places = [_tropicos_found[volume]] if _tropicos_found[volume] else []
    else:
        names = dict.fromkeys([volume, re.sub(r"^TropicosImages?(?:Vol)?(\d+)$", r"TropicosImagesVol\1", volume)])
        places = [f"{server}\\{name}" for name in names for server in TROPICOS_SERVERS]
    for place in places:
        base = TROPICOS_IIIF + quote(f"{place}\\{folder}\\{image_id}.jp2", safe="")
        info = _iiif_info(base, control)
        if info:
            if volume not in TROPICOS_FOLDERS:
                _tropicos_found[volume] = place
            return _download_iiif_service(base, stem, log, control, info)
    if volume not in TROPICOS_FOLDERS:
        _tropicos_found[volume] = None
    log(f"    Tropicos image server has no original for image {image_id}; downloading the 640 px copy")
    return None


def _download_jacq(url, stem, log, control):
    # JACQ: object.jacq.org/europeana/<herbarium>/<id>.jpg is a copy made for Europeana (about 1500 px). JACQ's
    # IIIF manifest for the same id points to the herbarium's own image server, which has the original.
    match = re.search(r"/(\d+)\.\w+$", urlparse(url).path)
    if not match:
        return None
    path = _download_iiif(f"https://api.jacq.org/v1/iiif/manifest/{match.group(1)}", stem, log, control)
    if not path:
        log("    JACQ has no IIIF image for this record; downloading the Europeana copy")
    return path


JBRJ_JPEGS = "https://jbrj-public-img.s3-sa-east-1.amazonaws.com/JPG/"


def _download_jbrj(url, stem, log, control):
    # Rio de Janeiro Botanical Garden (Reflora, JABOT): GBIF holds .../fsi/server?type=image&source=DZI/<key>.dzi,
    # the address of an image server that is gone (the bucket answers 403). The full-size JPEG of the same sheet is
    # in the garden's image bucket under the same key; the Deep Zoom tiles named by the link are the fallback.
    source = (parse_qs(urlparse(url).query).get("source") or [""])[0]
    match = re.fullmatch(r"DZI/(.+)\.dzi", source)
    if not match:
        return None
    resp = _try_get(JBRJ_JPEGS + quote(match.group(1)) + ".jpg", control)
    path = resp and _save_image(resp, stem, control)
    if path:
        return path
    try:
        return _download_deepzoom(f"https://{host_of(url)}/{quote(source)}", stem, log, control)
    except ItemSkipped:
        raise
    except Exception as exc:
        log(f"    Deep Zoom download failed ({exc})")
        return None


def _download_deepzoom(dzi_url, stem, log, control):
    """Join the largest level of a Deep Zoom image (<name>.dzi beside <name>_files/<level>/<column>_<row>.jpg)."""
    resp = _try_get(dzi_url, control)
    if resp is None:
        return None
    text = resp.text
    found = {k: re.search(rf'{k}="([^"]+)"', text) for k in ("TileSize", "Overlap", "Format", "Width", "Height")}
    if not all(found.values()):
        return None
    tile, overlap, width, height = (int(found[k].group(1)) for k in ("TileSize", "Overlap", "Width", "Height"))
    level = (max(width, height) - 1).bit_length()          # the last level is the full-size image
    files = re.sub(r"\.dzi$", "", dzi_url) + f"_files/{level}/"
    pieces = [(x * tile - (overlap if x else 0), y * tile - (overlap if y else 0),
               f"{files}{x}_{y}.{found['Format'].group(1)}")
              for y in range(-(-height // tile)) for x in range(-(-width // tile))]
    path = _stitch(width, height, pieces, stem, log, control)
    log(f"    Deep Zoom image joined from tiles ({width} x {height})")
    return path


def _save_image(resp, stem, control=None):
    """Write an image response to <stem>.<extension>. Returns the Path, or None when it is not an image."""
    kind = resp.headers.get("Content-Type", "").lower()
    if not kind.startswith("image/"):
        resp.close()
        return None
    path = Path(f"{stem}.{_detect_extension(kind, resp.url)}")
    partial = path.with_name(path.name + ".part")
    try:
        with resp, open(partial, "wb") as out:
            for chunk in resp.iter_content(1 << 16):
                if control:
                    control.check()
                out.write(chunk)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def _download_google_drive(url, stem, log, control):
    # Share links open a viewer page; the download address of the same file returns the image itself (the file as
    # uploaded: the herbaria sharing this way uploaded ~1000 px images, so no larger version exists).
    match = re.search(r"/file/d/([\w-]+)|[?&]id=([\w-]+)", url)
    if not match:
        return None
    file_id = match.group(1) or match.group(2)
    for address in (f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
                    f"https://drive.google.com/uc?export=download&id={file_id}"):
        resp = _try_get(address, control)
        path = resp and _save_image(resp, stem, control)
        if path:
            log("    Google Drive file downloaded")
            return path
    log("    Google Drive did not give the file (it may not be shared with everyone)")
    return None


SMITHSONIAN_IIIF = "https://ids.si.edu/ids/iiif/"


def _download_smithsonian(url, stem, log, control):
    # collections.nmnh.si.edu/media/?i=... redirects to the image on ids.si.edu, but its firewall rejects requests
    # that do not look like a browser's. The redirect scales the image to the height in h= (GBIF's links ask for
    # h=2000), but it names the image's ark id, and ids.si.edu serves that id over IIIF at full size.
    host = host_of(url)
    headers = {"User-Agent": BROWSER_USER_AGENT, "Accept": "image/avif,image/webp,image/*,*/*;q=0.8",
               "Accept-Language": "en-US,en;q=0.9", "Referer": f"https://{host}/"}
    throttle(host, MODE_REQUESTS, control)
    try:
        resp = session().get(url, headers=headers, timeout=30, verify=False, allow_redirects=False, stream=True)
        location = resp.headers.get("Location") if resp.is_redirect else None
        if location is None:
            path = _save_image(resp, stem, control)
            if not path:
                log("    The Smithsonian site rejected the request")
            return path
        resp.close()
        ark = re.search(r"ark:/\d+/\w+", unquote(location))
        path = ark and _download_iiif_service(SMITHSONIAN_IIIF + quote(ark.group(), safe=""), stem, log, control)
        if path:
            return path
        log("    The Smithsonian IIIF service did not give the full-size image; downloading the size in the link")
        resp = session().get(urljoin(url, location), headers=headers, timeout=60, verify=False, stream=True)
    except requests.RequestException as exc:
        log(f"    Smithsonian download failed: {exc}")
        return None
    return _save_image(resp, stem, control)


STITCH_WORKERS = 6
PIECE_RETRY_WAITS = (1, 4, 10)     # seconds before asking again for a piece of an image that did not arrive


def _zoomify_levels(width, height, tile):
    """(columns, rows, first tile number) for each level of a Zoomify pyramid, smallest first."""
    sizes = [(width, height)]
    while sizes[0][0] > tile or sizes[0][1] > tile:
        sizes.insert(0, ((sizes[0][0] + 1) // 2, (sizes[0][1] + 1) // 2))
    levels, first = [], 0
    for w, h in sizes:
        columns, rows = -(-w // tile), -(-h // tile)
        levels.append((columns, rows, first))
        first += columns * rows
    return levels, sizes


def _stitch(width, height, pieces, stem, log, control):
    """Fetch every piece, paste it into one image and save it as <stem>.jpg. `pieces` are (x, y, url)."""
    from concurrent.futures import ThreadPoolExecutor
    from io import BytesIO
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    def get(piece):
        # One piece that fails once must not cost the whole full-size image (the fallback is a reduced copy), and
        # image servers do refuse the odd request when several arrive together: try each piece a few times.
        x, y, address = piece
        for wait in (*PIECE_RETRY_WAITS, None):
            resp = _try_get(address, control)
            if resp is not None:
                with resp:
                    return x, y, resp.content
            if wait is None:
                raise ValueError(f"the piece at {x},{y} could not be fetched")
            pause(wait, control)

    image = Image.new("RGB", (width, height))
    log(f"    Fetching {len(pieces)} pieces")
    with ThreadPoolExecutor(STITCH_WORKERS) as pool:
        futures = [pool.submit(get, piece) for piece in pieces]
        try:
            for future in futures:
                if control:
                    control.check()
                x, y, data = future.result()
                with Image.open(BytesIO(data)) as part:
                    image.paste(part.convert("RGB"), (x, y))
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    path = Path(f"{stem}.jpg")
    partial = path.with_name(path.name + ".part")
    try:
        image.save(partial, format="JPEG", quality=95)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def _download_zoomify(base, width, height, tile, stem, log, control):
    """Fetch every tile of the largest level of a Zoomify image and join them into one JPEG."""
    levels, _ = _zoomify_levels(width, height, tile)
    top = len(levels) - 1
    columns, rows, first = levels[top]
    base = base if base.endswith("/") else base + "/"
    pieces = [(x * tile, y * tile, f"{base}TileGroup{(first + y * columns + x) // 256}/{top}-{x}-{y}.jpg")
              for y in range(rows) for x in range(columns)]
    return _stitch(width, height, pieces, stem, log, control)


def _download_cria(url, stem, log, control):
    # CRIA / speciesLink viewer pages only wrap an OpenSeadragon viewer. The image is a Zoomify tile pyramid whose
    # address, width and height are written into the page.
    page = _try_get(url, control)
    if page is None:
        return None
    html = page.text
    tiles = re.search(r"tilesUrl:\s*[\"']([^\"']+)", html)
    size = re.search(r"width:\s*(\d+),\s*height:\s*(\d+)", html)
    if not (tiles and size):
        log("    CRIA page has no tile information")
        return None
    width, height = int(size.group(1)), int(size.group(2))
    properties = _try_get(tiles.group(1).rstrip("/") + "/ImageProperties.xml", control)
    tile = 256
    if properties is not None:
        found = re.search(r'TILESIZE="(\d+)"', properties.text)
        tile = int(found.group(1)) if found else tile
    try:
        path = _download_zoomify(tiles.group(1), width, height, tile, stem, log, control)
        log(f"    CRIA image assembled from tiles ({width} x {height})")
        return path
    except ItemSkipped:
        raise
    except Exception as exc:
        log(f"    Tile download failed ({exc}); trying the speciesLink download")
    code = re.search(r"/viewer/([^/?#]+)", url)
    folder = re.search(r"https?://[^/]+/(?:[^/]+/)*?(herbaria/[^\"']+?)/?$", tiles.group(1))
    if code and folder:
        address = ("https://specieslink.net/search/util/osd-dezoomify?imagecode=" + code.group(1)
                   + f"&path={folder.group(1)}&width={width}&height={height}")
        resp = _try_get(address, control)
        return resp and _save_image(resp, stem, control)
    return None


# --------------------------------------------------------------------------- downloading the images in a dataset

BLOCKED_AFTER = 3  # skip a site for the rest of the run after this many refusals in a row
FAILING_AFTER = 5  # ...or after this many other failures in a row
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def _has_image_suffix(url):
    return urlparse(url).path.lower().endswith(IMAGE_SUFFIXES)


def _extract_image_from_json(json_text, log=print):
    try:
        data = json.loads(json_text)
    except (json.JSONDecodeError, TypeError) as exc:
        log(f"    IIIF JSON parse failed: {exc}")
        return None
    if not isinstance(data, dict):
        return None

    candidates = []
    for canvas in data.get("items", []):
        for page in canvas.get("items", []):
            for annotation in page.get("items", []):
                body = annotation.get("body", {})
                body = body[0] if isinstance(body, list) and body else body
                if isinstance(body, dict):
                    candidates.append(body.get("id") or body.get("@id"))
    for sequence in data.get("sequences", []):
        for canvas in sequence.get("canvases", []):
            for image in canvas.get("images", []):
                resource = image.get("resource", {})
                candidates.append(resource.get("@id") or resource.get("id"))
    candidates += [data.get(key) for key in ("imageUri", "imageUrl", "identifier")]
    return next((c for c in candidates if isinstance(c, str) and c.startswith("http")), None)


def _resolve_image_from_html(resp, page_url, log=print):
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ValueError("HTML page returned; install beautifulsoup4 to find images in web pages")
    soup = BeautifulSoup(resp.text, "html.parser")

    meta = (soup.find("meta", property="og:image")
            or soup.find("meta", attrs={"name": "twitter:image"})
            or soup.find("link", rel="image_src"))
    href = meta and (meta.get("content") or meta.get("href"))
    if href and href.strip():
        return urljoin(page_url, href.strip())

    manifest = soup.find("link", rel="alternate", type="application/ld+json")
    if manifest and manifest.get("href"):
        try:
            manifest_resp = request_with_backoff(urljoin(page_url, manifest["href"].strip()), log=log)
            image = _extract_image_from_json(manifest_resp.text, log)
            if image:
                return image
        except Exception as exc:
            log(f"    IIIF manifest failed: {exc}")

    for tag in soup.find_all("img", src=True):
        src = urljoin(page_url, tag["src"].strip())
        if not _has_image_suffix(src):
            continue
        try:
            if int(tag.get("width", 999)) < 100 or int(tag.get("height", 999)) < 100:
                continue
        except (TypeError, ValueError):
            pass
        return src

    for tag in soup.find_all("a", href=True):
        link = urljoin(page_url, tag["href"].strip())
        if _has_image_suffix(link):
            return link
    return None


def _detect_extension(content_type, url=""):
    content_type = content_type.lower()
    for key, ext in (("jpeg", "jpg"), ("jpg", "jpg"), ("png", "png"), ("tif", "tif"), ("gif", "gif"), ("webp", "webp")):
        if key in content_type:
            return ext
    path = urlparse(url).path.lower()
    for ext in ("jpg", "jpeg", "png", "tif", "tiff", "gif", "webp"):
        if path.endswith(f".{ext}"):
            return {"jpeg": "jpg", "tiff": "tif"}.get(ext, ext)
    return "jpg"


MIN_SHORT_SIDE = 2000  # px; a smaller sheet has too few pixels per mm to measure spikelets
ARCHIVE_NODE = re.compile(r"^https?://ia\d+\.us\.archive\.org/\d+/items/(.+)$")
GBIF_IMAGE_CACHE = "https://api.gbif.org/v1/image/cache/occurrence/{gbif_id}/media/{key}"


def canonical_url(url):
    """The lasting address of a link that names one server of a pool. The Internet Archive's storage nodes
    (ia802608.us.archive.org/13/items/...) are moved and retired; archive.org/download/... finds the file."""
    match = ARCHIVE_NODE.match(url)
    return f"https://archive.org/download/{match.group(1)}" if match else url


def image_size(path):
    """(width, height) of a saved image, or None when it cannot be read."""
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def _download_gbif_copy(url, gbif_id, stem, log, control):
    """Last resort for a link that is dead or unreachable: the copy GBIF made when it indexed the record. It is
    at most 1200 px tall, enough to read a label and measure a panicle but not a spikelet, so it is reported as
    a small image. Returns the saved Path or None."""
    if not gbif_id:
        return None
    key = hashlib.md5(url.encode("utf-8")).hexdigest()
    resp = _try_get(GBIF_IMAGE_CACHE.format(gbif_id=gbif_id, key=key), control)
    return (resp and _save_image(resp, stem, control)) or None


def _fetch_image(url, stem, log=print, control=None, gbif_id=None):
    url = canonical_url(url)
    special = download_special(url, stem, log, control, gbif_id)
    if special:
        return special
    resp = request_with_backoff(url, log=log, control=control, stream=True)
    content_type = resp.headers.get("Content-Type", "").lower()
    if "json" in content_type:
        path = _download_iiif(url, stem, log, control)      # a IIIF manifest: the largest size its server gives
        if path:
            resp.close()
            return path
        image_url = _extract_image_from_json(resp.text, log)
        if not image_url:
            raise ValueError("JSON response: no image URL found")
        log(f"    IIIF image: {image_url[:80]}")
        resp = request_with_backoff(image_url, log=log, control=control, stream=True)
    elif "text/html" in content_type:
        image_url = _resolve_image_from_html(resp, url, log)
        if not image_url:
            if looks_blocked(resp.text):
                raise AccessDenied(url, reason="the site showed a sign-in or verification page")
            raise ValueError("HTML page has no recognizable image (it may need JavaScript)")
        log(f"    Image from page: {image_url[:80]}")
        resp = request_with_backoff(image_url, log=log, control=control, stream=True)

    content_type = resp.headers.get("Content-Type", "").lower()
    if "text/html" in content_type or "json" in content_type:
        resp.close()
        raise ValueError(f"Final response is not an image (Content-Type: {content_type})")
    path = Path(f"{stem}.{_detect_extension(content_type, resp.url)}")
    partial = path.with_name(path.name + ".part")
    try:
        with resp, open(partial, "wb") as out:
            for chunk in resp.iter_content(1 << 16):
                check(control)
                out.write(chunk)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def _replace_old_versions(path):
    for old in path.parent.glob(f"{path.stem}.*"):
        if old != path and old.suffix.lower() in IMAGE_EXTENSIONS:
            old.unlink(missing_ok=True)


def media_status(csv_path, media_dir, log=print):
    df = read_csv(csv_path, log)
    ids = gbif_ids(df)
    images = index_images(media_dir)
    errors = df["media_error"] if "media_error" in df.columns else pd.Series("", index=df.index)
    items = {}
    for idx, gbif_id in ids.items():
        if not gbif_id or gbif_id in items:
            continue
        url = find_media_url(df.loc[idx]) or ""
        if gbif_id in images:
            status, detail = "Downloaded", images[gbif_id].name
        elif not url:
            status, detail = "No link", ""
        elif not is_blank(errors[idx]):
            status, detail = "Failed", str(errors[idx])
        else:
            status, detail = "Not downloaded", url
        items[gbif_id] = {"id": gbif_id, "status": status, "detail": detail, "image": images.get(gbif_id)}
    return list(items.values())


def download_media(csv_path, media_dir, log=print, control=None, progress=None, only_ids=None, force=False):
    from . import browser_sign_in  # imported here: browser_sign_in imports this module

    new_run()
    try:
        return _download_media(csv_path, media_dir, log, control, progress, only_ids, force)
    finally:
        browser_sign_in.close_fetcher()


def _download_media(csv_path, media_dir, log, control, progress, only_ids, force):
    from . import browser_sign_in  # imported here: browser_sign_in imports this module

    csv_path = Path(csv_path)
    media_dir = Path(media_dir)
    media_dir.mkdir(parents=True, exist_ok=True)
    failed_csv = csv_path.with_name(csv_path.stem + "_failed_media.csv")

    df = read_csv(csv_path, log).reset_index(drop=True)
    df.columns = df.columns.str.strip()
    for column in ("media_path", "media_error", "media_size"):
        df[column] = df[column].astype("object").fillna("") if column in df.columns else ""
        df[column] = df[column].astype("object")
    ids = gbif_ids(df)
    repeat = ids.groupby(ids).cumcount()
    rows = [idx for idx in df.index if ids[idx] and (only_ids is None or ids[idx] in only_ids)]
    images = index_images(media_dir)

    summary = {"total": len(rows), "downloaded": 0, "skipped": 0, "skipped_by_user": 0, "failed": 0,
               "cancelled": False, "blocked": {}, "skipped_sites": {}, "failing": {}, "small": [],
               "gbif_copies": []}

    state = threading.Lock()                # the table, the summary and the progress count are shared by the lanes
    control = control or TaskControl()
    finished = [0]

    def finished_one():
        with state:
            finished[0] += 1
            count = finished[0]
            if count % SAVE_EVERY == 0:
                save_csv(df, csv_path)
        if progress:
            progress(count, len(rows))

    def saved(idx, path, say, copy=False):
        """Record a saved image; one under MIN_SHORT_SIDE is kept but noted, so it is not mistaken for a good one."""
        _replace_old_versions(path)
        size = image_size(path)
        small = bool(size and min(size) < MIN_SHORT_SIDE)
        if small:
            say(f"Warning: only {size[0]} x {size[1]} px, too small to measure reliably "
                f"(under {MIN_SHORT_SIDE} px on the short side)")
        with state:
            df.at[idx, "media_path"] = str(path)
            df.at[idx, "media_error"] = ""
            df.at[idx, "media_size"] = f"{size[0]}x{size[1]}" if size else ""
            summary["downloaded"] += 1
            if small:
                summary["small"].append(ids[idx])
            if copy:
                summary["gbif_copies"].append(ids[idx])

    def failed(idx, message, group=None, host=None):
        with state:
            df.at[idx, "media_error"] = message
            summary["failed"] += 1
            if group:
                summary[group].setdefault(host, []).append(ids[idx])

    def gbif_copy(idx, label, url, say, item_control):
        """Save GBIF's reduced copy of an image whose own link does not work. True when there was one."""
        try:
            copy = _download_gbif_copy(url, ids[idx], media_dir / label, say, item_control)
        except ItemSkipped:
            copy = None
        if copy:
            say("Saved GBIF's reduced copy of this image instead")
            saved(idx, copy, say, copy=True)
        return bool(copy)

    scope = f"{len(rows)} selected records" if only_ids is not None else f"{len(rows)} records"
    log(f"Downloading images for {scope} to {media_dir}")

    # What is already there, or has no link, is settled first. The rest is sorted by website.
    sites = {}                              # website -> [(row, label, url)] in the dataset's order
    for idx in rows:
        label = ids[idx] if repeat[idx] == 0 else f"{ids[idx]}_{repeat[idx]}"
        existing = images.get(label)
        if existing and not force:
            df.at[idx, "media_path"] = str(existing)
            df.at[idx, "media_error"] = ""
            summary["skipped"] += 1
            continue
        url = find_media_url(df.loc[idx])
        if not url:
            df.at[idx, "media_error"] = "No media link found in row"
            summary["failed"] += 1
            continue
        sites.setdefault(host_of(url), []).append((idx, label, url))
    finished[0] = summary["skipped"] + summary["failed"]
    if progress:
        progress(finished[0], len(rows))

    hand_over = PREFS["strategy"] == "auto" and browser_sign_in.available()

    def download_site(host, items, lane):
        """Download one website's images, one after another. Returns the images left when the site refuses plain
        requests and the browser lane should take it over, otherwise []."""
        _local.plain_only = lane == PLAIN_LANE
        item_control = ItemControl(control, host)
        denied = failures = 0
        for position, (idx, label, url) in enumerate(items):
            if control.cancelled:
                return []
            item_control.new_item()

            def say(message, label=label):
                log(f"  [{label}] {str(message).strip()}")

            if denied >= BLOCKED_AFTER:
                failed(idx, f"Access denied by {host}; skipped after repeated refusals", "blocked", host)
            elif host in control.skipped_hosts:
                failed(idx, f"Site skipped by you: {host}", "skipped_sites", host)
            elif failures >= FAILING_AFTER:
                with state:
                    summary["failing"].setdefault(host, []).append(ids[idx])
                # the site is down for this run, but GBIF's reduced copy does not come from it
                if not gbif_copy(idx, label, url, say, item_control):
                    failed(idx, f"Site {host} skipped after {FAILING_AFTER} failures in a row")
            else:
                log(f"{label}: {url[:80]}")
                control.active[host] = label
                try:
                    path = _fetch_image(url, media_dir / label, say, item_control, ids[idx])
                    saved(idx, path, say)
                    denied = failures = 0
                except ItemSkipped:
                    if control.cancelled:
                        return []
                    if item_control.skip_site:
                        control.skipped_hosts.add(host)
                        control.skip_requests.discard(host)
                        say(f"Skipped, with the rest of {host}")
                        failed(idx, f"Site skipped by you: {host}", "skipped_sites", host)
                    else:
                        say("Skipped")
                        with state:
                            summary["skipped_by_user"] += 1
                except AccessDenied as exc:
                    if lane == PLAIN_LANE and hand_over:
                        say(f"{host} refuses plain requests; its images will go through the browser")
                        return items[position:]
                    say(f"Failed: {exc}")
                    if not denied:
                        note_refused(host)
                    denied += 1
                    failed(idx, str(exc), "blocked", host)
                except Exception as exc:
                    say(f"Failed: {exc}")
                    failures += 1
                    # a dead link or an unreachable server: GBIF's own reduced copy is better than nothing
                    if not gbif_copy(idx, label, url, say, item_control):
                        failed(idx, str(exc))
                    if failures == FAILING_AFTER:
                        log(f"{host} failed {FAILING_AFTER} times in a row; skipping it for the rest of this run")
                        with state:
                            summary["failing"].setdefault(host, []).append(ids[idx])
                finally:
                    control.active.pop(host, None)
            finished_one()
        return []

    # Lanes. Several websites are downloaded at the same time, each by one lane, so no site ever gets more than one
    # request at a time from this loop and a slow or stalled site holds up nobody else. The browser is a single
    # window that must stay on the thread that opened it, so every site that needs it (set to browser mode, or
    # found to refuse plain requests on the way) queues for one browser lane, which is this thread.
    browser_sites = queue.Queue()
    plain_sites = queue.Queue()
    for host in sorted(sites, key=lambda h: -len(sites[h])):           # the largest first: they take longest
        if PREFS["strategy"] == "browser" or site_mode(host) == MODE_BROWSER:
            browser_sites.put((host, sites[host]))
        else:
            plain_sites.put(host)

    def lane():
        while not control.cancelled:
            try:
                host = plain_sites.get_nowait()
            except queue.Empty:
                return
            try:
                left = download_site(host, sites[host], PLAIN_LANE)
            except Exception as exc:        # never let one site's surprise end the others' downloads unnoticed
                log(f"Error: downloads from {host} stopped: {exc}")
                left = []
            if left:
                browser_sites.put((host, left))

    count = max(1, min(int(PREFS["parallel_sites"]), plain_sites.qsize()))
    if plain_sites.qsize() > 1 and count > 1:
        log(f"{len(sites)} websites; downloading from up to {count} at the same time")
    lanes = [threading.Thread(target=lane, daemon=True) for _ in range(count)]
    for thread in lanes:
        thread.start()
    while True:
        try:
            host, items = browser_sites.get(timeout=0.2)
        except queue.Empty:
            if not any(thread.is_alive() for thread in lanes) and browser_sites.empty():
                break
            continue
        download_site(host, items, BROWSER_LANE)
    _local.plain_only = False
    summary["cancelled"] = control.cancelled

    save_csv(df, csv_path)
    failed = df[df["media_error"].astype(str).str.strip() != ""]
    if len(failed):
        save_csv(failed, failed_csv)
        log(f"{len(failed)} records without an image are listed in {failed_csv}")
    elif failed_csv.exists():
        failed_csv.unlink()

    if summary["cancelled"]:
        log("Image download cancelled")
    elif progress:
        progress(len(rows), len(rows))
    control.active.clear()
    for host, skipped_ids in summary["skipped_sites"].items():
        note_refused(host, "skipped by you")
        log(f"{host} was skipped by you for {len(skipped_ids)} images. It is listed under Images, Site logins.")
    for host, failing_ids in summary["failing"].items():
        note_refused(host, f"failed {FAILING_AFTER} times in a row")
        log(f"{host} kept failing and was skipped ({len(failing_ids)} images). It is listed under Images, Site logins.")
    for host, blocked_ids in summary["blocked"].items():
        log(f"{host} refused access for {len(blocked_ids)} images. Use Images, Site logins to sign in, then retry.")
    if summary["gbif_copies"]:
        log(f"{len(summary['gbif_copies'])} images came from GBIF's reduced copies because their own links did not "
            f"work (at most 1200 px tall).")
    if summary["small"]:
        log(f"{len(summary['small'])} downloaded images are under {MIN_SHORT_SIDE} px on the short side, too small "
            f"to measure seeds on (their size is in the media_size column): {', '.join(summary['small'][:20])}"
            f"{' ...' if len(summary['small']) > 20 else ''}")
    log(f"Image download complete: {summary['downloaded']} downloaded, {summary['skipped']} already present, "
        f"{summary['skipped_by_user']} skipped by you, {summary['failed']} failed")
    return summary
