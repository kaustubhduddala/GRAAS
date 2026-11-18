"""Tests for unattended runs: full-size image sources, dead links, short scale bars, and long measuring runs
that must survive a crash or a model server that goes away.

Run with:  python -m unittest discover tests
"""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pandas as pd
import requests
from PIL import Image

from program import image_download as dl
from program import utils
from program.measure import calibration, pipeline
from program.measure.providers import ProviderError, ProviderUnavailable


class FakeResponse:
    def __init__(self, body=b"", kind="image/jpeg", status=200, url="https://example.org/x.jpg"):
        self.content, self.status_code, self.url = body, status, url
        self.headers = {"Content-Type": kind}
        self.text = body.decode("utf-8", "replace") if isinstance(body, bytes) else body

    def iter_content(self, size):
        yield self.content

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def jpeg_bytes(size=(60, 80)):
    path = Path(tempfile.mkdtemp()) / "x.jpg"
    Image.new("RGB", size, (200, 190, 170)).save(path)
    return path.read_bytes()


class FullSizeSources(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        dl._https_only.clear()

    def test_archive_org_storage_nodes_become_the_lasting_address(self):
        self.assertEqual(dl.canonical_url("https://ia802608.us.archive.org/13/items/JBGP-HERBARIO/JBGP-1.jpg"),
                         "https://archive.org/download/JBGP-HERBARIO/JBGP-1.jpg")
        self.assertEqual(dl.canonical_url("https://example.org/a.jpg"), "https://example.org/a.jpg")

    def test_jbrj_dead_image_server_link_goes_to_the_jpeg_of_the_same_key(self):
        asked = []

        def get(address, control=None):
            asked.append(address)
            return FakeResponse(jpeg_bytes())

        url = "http://jbrj-public.s3-sa-east-1.amazonaws.com/fsi/server?type=image&source=DZI/rspf/rspf/0/0/34/28/rspf00003428.dzi"
        with mock.patch.object(dl, "_try_get", side_effect=get):
            path = dl.download_special(url, self.folder / "1", lambda _: None)
        self.assertEqual(asked, ["https://jbrj-public-img.s3-sa-east-1.amazonaws.com/JPG/rspf/rspf/0/0/34/28/rspf00003428.jpg"])
        self.assertTrue(path.is_file())

    def test_deep_zoom_pieces_are_the_last_level_and_allow_for_the_overlap(self):
        xml = '<Image TileSize="128" Overlap="1" Format="jpg"><Size Width="300" Height="200"/></Image>'
        seen = {}

        def stitch(width, height, pieces, stem, log, control):
            seen.update(width=width, height=height, pieces=pieces)
            return Path(f"{stem}.jpg")

        with mock.patch.object(dl, "_try_get", return_value=FakeResponse(xml.encode(), "application/xml")), \
                mock.patch.object(dl, "_stitch", side_effect=stitch):
            dl._download_deepzoom("https://host/DZI/a/b.dzi", self.folder / "2", lambda _: None, None)
        self.assertEqual((seen["width"], seen["height"]), (300, 200))
        self.assertEqual(len(seen["pieces"]), 3 * 2)
        self.assertEqual(seen["pieces"][0], (0, 0, "https://host/DZI/a/b_files/9/0_0.jpg"))    # 2**9 covers 300 px
        self.assertEqual(seen["pieces"][1][:2], (127, 0))                 # the second tile starts one px early
        self.assertEqual(seen["pieces"][3][:2], (0, 127))

    def test_a_piece_of_a_joined_image_that_fails_once_is_asked_for_again(self):
        asked = []

        def get(address, control=None):
            asked.append(address)
            return None if asked.count(address) == 1 and address.endswith("b") else FakeResponse(jpeg_bytes((10, 10)))

        with mock.patch.object(dl, "_try_get", side_effect=get), mock.patch.object(dl, "pause") as pause:
            path = dl._stitch(20, 10, [(0, 0, "https://x/a"), (10, 0, "https://x/b")], self.folder / "4",
                              lambda _: None, None)
        self.assertEqual(dl.image_size(path), (20, 10))
        self.assertEqual(asked.count("https://x/b"), 2)
        pause.assert_called_once()

    def test_an_http_address_whose_port_is_closed_is_fetched_over_https_and_remembered(self):
        calls = []

        def get(url, **kwargs):
            calls.append(url)
            if url.startswith("http://"):
                raise requests.ConnectTimeout("no answer")
            return FakeResponse(b"ok")

        with mock.patch.object(dl, "session", return_value=mock.Mock(get=get)):
            dl._get("http://old.example.org/a.jpg", "old.example.org")
            dl._get("http://old.example.org/b.jpg", "old.example.org")
        self.assertEqual(calls, ["http://old.example.org/a.jpg", "https://old.example.org/a.jpg",
                                 "https://old.example.org/b.jpg"])

    def test_a_manifest_that_names_only_an_image_service_is_understood(self):
        manifest = {"items": [{"items": [{"items": [{"body": {
            "id": "https://iiif.example.org/E1/info.json",
            "service": [{"id": "https://iiif.example.org/E1", "type": "ImageService3"}]}}]}]}]}
        self.assertEqual(dl._image_service_id(manifest), "https://iiif.example.org/E1")
        self.assertEqual(dl._iiif_limits({"profile": ["level2", {"maxWidth": 5000}], "maxArea": 9}),
                         {"maxWidth": 5000, "maxArea": 9})

    def test_a_level_0_server_is_asked_for_the_sizes_it_lists(self):
        info = {"width": 6965, "height": 10035, "profile": "level0",
                "sizes": [{"width": 108, "height": 156}, {"width": 2843, "height": 4096}],
                "tiles": [{"width": 256, "scaleFactors": [1, 2, 4]}]}
        asked = []

        def get(address, control=None):
            asked.append(address)
            return FakeResponse(jpeg_bytes())

        with mock.patch.object(dl, "_try_get", side_effect=get):
            dl._download_iiif_service("https://iiif.example.org/E1", self.folder / "3", lambda _: None, None, info)
        self.assertEqual(asked, ["https://iiif.example.org/E1/full/2843,4096/0/default.jpg"])


class DeadLinks(unittest.TestCase):
    def test_a_dead_link_falls_back_to_gbifs_copy_and_the_small_image_is_reported(self):
        folder = Path(tempfile.mkdtemp())
        csv = folder / "master.csv"
        utils.save_csv(pd.DataFrame({"gbifID": ["11", "22"],
                                     "identifier": ["http://gone.example.org/a.jpg", "http://ok.example.org/b.jpg"]}), csv)

        def fetch(url, stem, log, control, gbif_id=None):
            if "gone" in url:
                raise RuntimeError("Request failed after 3 attempts: HTTP 404")
            path = Path(f"{stem}.jpg")
            Image.new("RGB", (2400, 3200), "white").save(path)
            return path

        def gbif_copy(url, gbif_id, stem, log, control):
            path = Path(f"{stem}.jpg")
            Image.new("RGB", (800, 1200), "white").save(path)
            return path

        log = []
        with mock.patch.object(dl, "_fetch_image", side_effect=fetch), \
                mock.patch.object(dl, "_download_gbif_copy", side_effect=gbif_copy), \
                mock.patch.object(dl, "note_refused"):
            summary = dl._download_media(csv, folder / "media", log.append, None, None, None, False)
        self.assertEqual((summary["downloaded"], summary["failed"]), (2, 0))
        self.assertEqual((summary["gbif_copies"], summary["small"]), (["11"], ["11"]))
        saved = utils.read_csv(csv).set_index("gbifID")
        self.assertEqual((saved.at["11", "media_size"], saved.at["22", "media_size"]), ("800x1200", "2400x3200"))
        self.assertTrue(any("too small to measure" in line for line in log))

    def test_the_gbif_copy_address_is_keyed_by_the_original_link(self):
        asked = []
        with mock.patch.object(dl, "_try_get", side_effect=lambda address, control=None: asked.append(address)):
            self.assertIsNone(dl._download_gbif_copy("http://a.example.org/x.jpg", "42", "stem", print, None))
        self.assertEqual(len(asked), 1)
        self.assertRegex(asked[0], r"^https://api\.gbif\.org/v1/image/cache/occurrence/42/media/[0-9a-f]{32}$")


class SeveralSitesAtOnce(unittest.TestCase):
    """Images are downloaded from several websites at the same time, one image at a time from each."""

    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.csv = self.folder / "master.csv"
        hosts = [f"{letter}.example.org" for letter in "abcd"]
        rows = [(str(1000 + 10 * h + n), f"https://{host}/{n}.jpg") for h, host in enumerate(hosts) for n in range(3)]
        utils.save_csv(pd.DataFrame(rows, columns=["gbifID", "identifier"]), self.csv)
        self.control = utils.TaskControl()
        self.lock = threading.Lock()
        self.busy, self.most, self.threads, self.clashes = {}, [0], {}, []

    def fetch(self, seconds=0.15, refuse=None, wait_for_skip=None):
        """A stand-in for _fetch_image that takes `seconds`, and records who was downloading what, when."""
        def fake(url, stem, log, control, gbif_id=None):
            host = dl.host_of(url)
            if host == refuse and getattr(dl._local, "plain_only", False):
                raise dl.AccessDenied(url, 403)
            with self.lock:
                if self.busy.get(host):
                    self.clashes.append(host)
                self.busy[host] = True
                self.most[0] = max(self.most[0], sum(self.busy.values()))
                self.threads.setdefault(host, set()).add(threading.current_thread())
            try:
                end = time.monotonic() + (5.0 if host == wait_for_skip else seconds)
                while time.monotonic() < end:
                    control.check()
                    time.sleep(0.01)
            finally:
                with self.lock:
                    self.busy[host] = False
            path = Path(f"{stem}.jpg")
            Image.new("RGB", (2400, 3200), "white").save(path)
            return path
        return fake

    def run_download(self, fake, sites=4):
        log = []
        with mock.patch.object(dl, "_fetch_image", side_effect=fake), mock.patch.object(dl, "note_refused"), \
                mock.patch.object(dl, "set_site_mode"), mock.patch.dict(dl.PREFS, {"parallel_sites": sites}):
            started = time.monotonic()
            summary = dl._download_media(self.csv, self.folder / "media", log.append, self.control, None, None, False)
        return summary, time.monotonic() - started, log

    def test_four_sites_take_about_as_long_as_one(self):
        summary, seconds, _ = self.run_download(self.fetch(0.15))
        self.assertEqual((summary["downloaded"], summary["failed"]), (12, 0))
        self.assertEqual(self.clashes, [])                   # never two downloads from one website at once
        self.assertEqual(self.most[0], 4)
        self.assertLess(seconds, 12 * 0.15 * 0.6)            # one after another would take 1.8 s
        saved = utils.read_csv(self.csv)
        self.assertTrue((saved["media_size"] == "2400x3200").all())

    def test_one_at_a_time_is_still_possible(self):
        summary, _, _ = self.run_download(self.fetch(0.02), sites=1)
        self.assertEqual((summary["downloaded"], self.most[0]), (12, 1))

    def test_a_site_that_refuses_plain_requests_is_finished_by_the_one_browser_lane(self):
        summary, _, log = self.run_download(self.fetch(0.02, refuse="b.example.org"))
        self.assertEqual((summary["downloaded"], summary["failed"]), (12, 0))
        self.assertEqual(self.threads["b.example.org"], {threading.current_thread()})   # where the browser lives
        self.assertNotIn(threading.current_thread(), self.threads["a.example.org"])
        self.assertTrue(any("will go through the browser" in line for line in log))

    def test_without_a_browser_a_refusing_site_is_reported_and_the_rest_carry_on(self):
        with mock.patch.dict(dl.PREFS, {"strategy": "requests"}):
            summary, _, _ = self.run_download(self.fetch(0.02, refuse="b.example.org"))
        self.assertEqual((summary["downloaded"], summary["failed"]), (9, 3))
        self.assertEqual(len(summary["blocked"]["b.example.org"]), 3)

    def test_one_website_can_be_skipped_by_name_while_the_others_finish(self):
        threading.Timer(0.3, lambda: self.control.request_skip(True, "c.example.org")).start()
        summary, seconds, _ = self.run_download(self.fetch(0.02, wait_for_skip="c.example.org"))
        self.assertEqual((summary["downloaded"], summary["failed"]), (9, 3))
        self.assertEqual(len(summary["skipped_sites"]["c.example.org"]), 3)
        self.assertLess(seconds, 3.0)
        errors = utils.read_csv(self.csv).set_index("gbifID")["media_error"].fillna("")
        self.assertEqual(sorted(i for i, e in errors.items() if "skipped by you" in e), ["1020", "1021", "1022"])

    def test_skip_skips_every_image_in_progress_once_and_cancel_stops_all_lanes(self):
        threading.Timer(0.2, self.control.request_skip).start()
        summary, _, _ = self.run_download(self.fetch(0.4))
        self.assertEqual((summary["skipped_by_user"], summary["downloaded"]), (4, 8))
        self.control.reset()
        self.setUp()
        threading.Timer(0.2, self.control.cancel_event.set).start()
        summary, seconds, _ = self.run_download(self.fetch(0.4))
        self.assertTrue(summary["cancelled"])
        self.assertEqual(summary["downloaded"], 0)
        self.assertLess(seconds, 1.5)


def sheet_with_scale_bar(ppc=100.0, sheet_width_cm=29.6, ticks=True, chart=True):
    """A sheet photographed on black with a colour chart and a 5 cm scale bar above it, like the TEX-LL images:
    the bar marks whole centimetres with thin white lines, alternate blocks filled, millimetres in the first."""
    margin, top = int(1.0 * ppc), int(2.2 * ppc)
    w, h = int(sheet_width_cm * ppc) + 2 * margin, int(42 * ppc) + top + margin
    image = np.full((h, w, 3), 12, np.uint8)
    cv2.rectangle(image, (margin, top), (w - margin, h - margin), (236, 226, 200), -1)
    if chart:                                              # patches 1.26 cm wide: regular, but coloured
        colours = [(20, 20, 20), (240, 240, 240), (200, 30, 120), (220, 40, 40), (240, 220, 30), (30, 160, 70),
                   (40, 150, 220), (60, 40, 160)]
        for i, colour in enumerate(colours * 2):
            x = margin + int(i * 1.26 * ppc)
            cv2.rectangle(image, (x, int(0.3 * ppc)), (x + int(1.26 * ppc), int(1.6 * ppc)), colour, -1)
    x0, y0, y1 = w - margin - int(6.0 * ppc), int(0.5 * ppc), int(1.5 * ppc)
    middle = (y0 + y1) // 2
    for k in range(5):
        a, b = x0 + int(round(k * ppc)), x0 + int(round((k + 1) * ppc))
        if k % 2:
            cv2.rectangle(image, (a, middle), (b, y1), (245, 245, 245), -1)
    for k in range(6):
        x = x0 + int(round(k * ppc))
        cv2.line(image, (x, y0), (x, y1), (245, 245, 245), 3)
    cv2.line(image, (x0, y0), (x0 + int(5 * ppc), y0), (245, 245, 245), 3)
    cv2.line(image, (x0, middle), (x0 + int(5 * ppc), middle), (245, 245, 245), 3)
    if ticks:
        for m in range(1, 10):
            x = x0 + int(round(m * ppc / 10))
            cv2.line(image, (x, y0), (x, middle), (245, 245, 245), 2)
    return image


class ShortScaleBar(unittest.TestCase):
    def test_a_five_centimetre_bar_sets_the_scale_instead_of_the_sheet_width(self):
        """The paper here is 29.6 cm wide, so the sheet width alone would be 2% out; the bar is exact."""
        cal = calibration.calibrate(sheet_with_scale_bar(ppc=100.0))
        self.assertEqual(cal.source, "ruler")
        self.assertAlmostEqual(cal.px_per_cm, 100.0, delta=0.5)
        self.assertAlmostEqual(cal.sheet_px_per_cm, 102.1, delta=1.0)
        self.assertEqual(cal.uncertain, [])

    def test_a_colour_chart_alone_is_not_a_scale_bar(self):
        rgb = sheet_with_scale_bar(ppc=100.0)
        rgb[:210, -760:] = 12                                # paint the bar out; the chart stays
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        self.assertIsNone(calibration.find_scale_bar(rgb, gray, 126.0))    # looked for at the chart's own pitch
        self.assertEqual(calibration.calibrate(rgb).source, "sheet")

    def test_the_bar_is_located_to_a_fraction_of_a_percent_at_other_resolutions(self):
        for ppc in (70.0, 178.0):
            rgb = sheet_with_scale_bar(ppc=ppc, ticks=False)
            bar = calibration.find_scale_bar(rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), ppc * 1.03)
            self.assertIsNotNone(bar, ppc)
            self.assertAlmostEqual(bar["px_per_cm"] / ppc, 1.0, delta=0.005)


class LongRuns(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.media = self.folder / "media"
        self.media.mkdir()
        for n in range(12):
            Image.new("RGB", (60, 80), (236, 226, 200)).save(self.media / f"{100 + n}.jpg")
        self.csv = self.folder / "measurements.csv"
        self.detector = mock.Mock()
        self.detector.describe.return_value = "test"

    def measure(self, side_effect):
        with mock.patch.object(pipeline, "measure_sheet", side_effect=side_effect), \
                mock.patch.object(pipeline, "pause") as pause, \
                mock.patch.object(pipeline, "draw_overlay"):
            log = []
            try:
                summary = pipeline.measure_images(self.media, self.csv, log=log.append, detector=self.detector)
            except ProviderError as exc:
                summary = exc
            return summary, log, pause

    def test_a_server_that_goes_away_for_a_moment_does_not_end_the_run(self):
        calls = []

        def measure_sheet(image, detector, profile, log=print, control=None, status=None):
            calls.append(Path(image).name)
            if len(calls) in (2, 3):                       # the second sheet fails twice, then works
                raise ProviderUnavailable("Could not reach LM Studio")
            return pipeline.blank_result(image), {}

        summary, log, pause = self.measure(measure_sheet)
        self.assertEqual((summary["measured"], summary["failed"]), (12, 0))
        self.assertEqual([call.args[0] for call in pause.call_args_list], list(pipeline.RETRY_WAITS_S[:2]))
        self.assertEqual(calls[1:4], ["101.jpg"] * 3)

    def test_a_server_that_stays_away_stops_the_run_with_everything_saved(self):
        def measure_sheet(image, detector, profile, log=print, control=None, status=None):
            if Path(image).name != "100.jpg":
                raise ProviderUnavailable("Could not reach LM Studio")
            return pipeline.blank_result(image), {}

        outcome, log, pause = self.measure(measure_sheet)
        self.assertIsInstance(outcome, ProviderUnavailable)
        self.assertEqual(pause.call_count, len(pipeline.RETRY_WAITS_S))
        self.assertEqual(list(utils.read_csv(self.csv)["gbifID"]), ["100"])
        self.assertTrue(any("run Measure again" in line for line in log))

    def test_the_same_failure_on_sheet_after_sheet_stops_the_run(self):
        summary, log, _ = self.measure(ValueError("the model answered with no JSON"))
        self.assertEqual(summary["failed"], pipeline.STOP_AFTER_FAILURES)
        self.assertIn("in a row failed", summary["stopped"])
        self.assertEqual(len(utils.read_csv(self.csv)), pipeline.STOP_AFTER_FAILURES)

    def test_a_csv_is_replaced_whole_or_not_at_all(self):
        path = self.folder / "data.csv"
        utils.save_csv(pd.DataFrame({"a": ["1"]}), path)
        with mock.patch.object(pd.DataFrame, "to_csv", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                utils.save_csv(pd.DataFrame({"a": ["2"]}), path)
        self.assertEqual(list(utils.read_csv(path)["a"]), ["1"])
        self.assertEqual([p.name for p in self.folder.glob("data.csv*")], ["data.csv"])


if __name__ == "__main__":
    unittest.main()
