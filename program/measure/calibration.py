"""Pixels per centimetre for a herbarium sheet image.

Follows the lab protocol: use a printed ruler when there is one, otherwise assume the standard
29 cm sheet width, and give no scale when neither can be trusted. Ruler tick spacing is measured
with a Fourier transform of the intensity along the ruler, which is precise to well under 1%; the
hard part is finding the ruler, so a reading is only accepted when its millimetre and centimetre
spacings agree with each other and with the sheet, or when an AI hint says where the ruler is.
A short scale bar that marks whole centimetres (find_scale_bar) is read from the positions of its
lines instead, and is accepted when it agrees with the sheet or shows millimetre ticks of its own.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import paper_reference

SHEET_WIDTH_CM = 29.0
SHEET_HEIGHT_CM = 42.0
# Without an AI hint a CV-found ruler only refines the sheet estimate (within 5%), unless it is a
# long ruler with very strong millimetre ticks. Colour charts and inch scales (1/8 in ticks) can
# otherwise pass for a metric ruler at a scale ~10% off.
RULER_AGREEMENT = 0.05
LONG_RULER_CM = 15.0
LONG_RULER_TICKS = 1000.0   # millimetre-tick prominence of a clear, long metric ruler
LONG_RULER_AGREEMENT = 0.25
# Tick spacings (in cm) a ruler can show, for reading a ruler an AI has located.
TICK_UNITS_CM = {"1 mm": 0.1, "5 mm": 0.5, "1 cm": 1.0, "1/16 in": 0.15875, "1/8 in": 0.3175, "1/4 in": 0.635,
                 "1 in": 2.54}
HINT_TOLERANCE = 0.20       # an AI ruler hint's rough scale may be off by this much
# An AI-located ruler is often a colour chart or a scale on another label; when the sheet edges are
# clear it must agree with the sheet width to within this share, or the sheet is used instead.
HINT_SHEET_AGREEMENT = 0.08
HINT_CUT_SHEET_AGREEMENT = 0.15   # the same when the sheet runs off the image (its width is then a lower bound)
# Below these resolutions a trait cannot be measured to a useful precision (1 px is >3% of a
# typical leaf width or >4% of a typical spikelet).
MIN_PX_PER_CM = {"panicle_length": 10.0, "leaf_width": 30.0, "seed_length": 60.0}


@dataclass
class Calibration:
    px_per_cm: float | None
    source: str                              # "ruler", "sheet", or "none"
    sheet_px_per_cm: float | None = None
    ruler_px_per_cm: float | None = None
    sheet_box: list | None = None            # [x0, y0, x1, y1] of the paper, full-resolution pixels
    ruler_line: list | None = None           # [[x, y], [x, y]] along the measured ruler span
    ruler_span_cm: float | None = None
    flags: list = field(default_factory=list)
    uncertain: list = field(default_factory=list)   # reasons the scale should be checked by a person

    def describe(self):
        if not self.px_per_cm:
            return "No scale: " + ("; ".join(self.flags) or "no ruler and the sheet edges are not visible")
        what = {"ruler": f"printed ruler ({self.ruler_span_cm:.0f} cm span)" if self.ruler_span_cm else "printed ruler",
                "manual": f"a {self.ruler_span_cm:g} cm length marked by a reviewer" if self.ruler_span_cm
                else "a reviewer",
                "sheet": f"sheet width = {SHEET_WIDTH_CM:g} cm"}.get(self.source, self.source)
        text = f"{self.px_per_cm:.1f} px/cm from {what}"
        if self.source == "ruler" and self.sheet_px_per_cm:
            text += f" (sheet width gives {self.sheet_px_per_cm:.1f})"
        return text + ("; " + "; ".join(self.flags) if self.flags else "")

    def allows(self, trait, minimum=None):
        return bool(self.px_per_cm) and self.px_per_cm >= (minimum or MIN_PX_PER_CM)[trait]

    @classmethod
    def from_dict(cls, data):
        return cls(**{k: v for k, v in (data or {}).items() if k in cls.__dataclass_fields__})

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# --------------------------------------------------------------------------- spectral helpers

def _spectrum(profile, oversample=16):
    x = np.asarray(profile, dtype=np.float64)
    x = (x - x.mean()) * np.hanning(len(x))
    nfft = 1 << int(np.ceil(np.log2(max(len(x), 8) * oversample)))
    power = np.abs(np.fft.rfft(x, nfft)) ** 2
    return np.fft.rfftfreq(nfft), power, nfft


def _peak(freqs, power, nfft, pmin, pmax, context=2.0):
    """(period, prominence) of the strongest peak with a period in [pmin, pmax], sub-bin accurate.

    Prominence is the peak power over the median power of periods within `context` times the
    search band, so a narrow search band does not hide how much the peak stands out.
    """
    band = np.where((freqs > 1.0 / pmax) & (freqs < 1.0 / pmin))[0]
    ref = np.where((freqs > 1.0 / (pmax * context)) & (freqs < context / pmin))[0]
    if len(band) < 3:
        return None, 0.0
    k = band[power[band].argmax()]
    d = 0.0
    if 0 < k < len(power) - 1:
        y0, y1, y2 = np.log(power[k - 1:k + 2] + 1e-12)
        denom = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / denom if denom else 0.0
    period = nfft / (k + d)
    prominence = float(power[k] / (np.median(power[ref if len(ref) > len(band) else band]) + 1e-12))
    return float(period), prominence


def _band_profile(gray, edge, line, half, start, end):
    """Mean intensity profile along a ruler running parallel to `edge` at `line`."""
    img = gray if edge in ("top", "bottom") else gray.T
    rows = img[max(0, line - half):line + half + 1, start:end].astype(np.float32)
    return rows.mean(axis=0)


# --------------------------------------------------------------------------- sheet

def sheet_estimate(rgb):
    """Cream paper extent from colour: returns dict(ok, box, width_px, height_px, touches_border)."""
    import cv2

    h0, w0 = rgb.shape[:2]
    scale = 1000.0 / max(h0, w0)
    small = cv2.resize(rgb, (max(1, int(w0 * scale)), max(1, int(h0 * scale))), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.int16)
    light, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    ref_l, ref_a, ref_b = paper_reference(light, a, b)       # cream or neutral white: found from the pixels
    paper = ((light > 120) & (np.abs(light - ref_l) < 45) & (b >= ref_b - 12) & (b < ref_b + 28)
             & (np.abs(a - ref_a) < 15))
    h, w = paper.shape

    def longest_run(flags):
        best, start = (0, 0, 0), None
        for i, flag in enumerate(np.append(flags, False)):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                if i - start > best[0]:
                    best = (i - start, start, i)
                start = None
        return best

    # A column belongs to the sheet when some of it is paper; plants can cover most of a column.
    run_x = longest_run(paper[int(0.2 * h):int(0.8 * h)].mean(axis=0) > 0.12)
    run_y = longest_run(paper[:, int(0.2 * w):int(0.8 * w)].mean(axis=1) > 0.12)
    if run_x[0] < 0.3 * w or run_y[0] < 0.3 * h:
        return {"ok": False}
    x0, x1, y0, y1 = run_x[1], run_x[2], run_y[1], run_y[2]
    touches = x0 <= 1 or x1 >= w - 1
    return {"ok": True, "box": [x0 / scale, y0 / scale, x1 / scale, y1 / scale],
            "width_px": (x1 - x0) / scale, "height_px": (y1 - y0) / scale,
            "touches_border": bool(touches), "height_touches": bool(y0 <= 1 or y1 >= h - 1)}


# --------------------------------------------------------------------------- ruler search (no AI)

def _ruler_candidates(gray, prior):
    """Scan bands along the four edges for long, thick, high-contrast periodic structure."""
    g = gray.astype(np.float32)
    height, width = g.shape
    win = int(min(max(8 * prior, 256), 0.9 * min(height, width)))
    classes = {"mm": (max(2.5, 0.065 * prior), 0.15 * prior, True), "cm": (0.6 * prior, 2.6 * prior, False)}
    found = []
    for edge in ("top", "bottom", "left", "right"):
        img = g if edge in ("top", "bottom") else g.T
        n_lines, n_along = img.shape
        band = int(0.16 * n_lines)
        step = max(1, int(prior / 40))
        lines = list(range(0, band) if edge in ("top", "left") else range(n_lines - band, n_lines))[::step]
        k = max(1, int(round(0.15 * prior / step)))
        for s0 in range(0, max(1, n_along - win + 1), max(1, win // 4)):
            strips = img[lines, s0:s0 + win]
            if strips.shape[1] < win:
                continue
            amp = strips.std(axis=1)
            for name, (pmin, pmax, derivative) in classes.items():
                if pmax <= pmin * 1.2:
                    continue
                x = np.diff(strips, axis=1) if derivative else strips
                x = (x - x.mean(axis=1, keepdims=True)) * np.hanning(x.shape[1])[None, :]
                nfft = 1 << int(np.ceil(np.log2(x.shape[1] * 4)))
                power = np.abs(np.fft.rfft(x, nfft, axis=1)) ** 2
                freqs = np.fft.rfftfreq(nfft)
                band_idx = (freqs > 1.0 / pmax) & (freqs < 1.0 / pmin)
                pb = power[:, band_idx]
                idx = pb.argmax(axis=1)
                lo, hi = np.clip(idx - 3, 0, pb.shape[1]), np.clip(idx + 4, 0, pb.shape[1])
                score = np.array([pb[i, lo[i]:hi[i]].sum() for i in range(len(idx))]) / (pb.sum(axis=1) + 1e-9)
                period = 1.0 / freqs[band_idx][idx]
                quality = score * np.clip(amp / 50.0, 0, 1)
                if len(quality) > k:
                    q = np.array([quality[i:i + k].min() for i in range(len(quality) - k + 1)])
                    p = np.array([np.median(period[i:i + k]) for i in range(len(quality) - k + 1)])
                    spread = np.array([period[i:i + k].std() / period[i:i + k].mean()
                                       for i in range(len(quality) - k + 1)])
                    q = q * (spread < 0.03)
                else:
                    q, p = quality, period
                i = int(q.argmax())
                found.append({"edge": edge, "cls": name, "line": int(lines[min(len(lines) - 1, i + k // 2)]),
                              "start": int(s0), "end": int(s0 + win), "q": float(q[i]), "period": float(p[i])})
    found.sort(key=lambda c: -c["q"])
    return found


def _extend_and_refine(gray, cand, prior):
    """Extend a candidate along its band while it stays periodic, then refine the period."""
    img = (gray if cand["edge"] in ("top", "bottom") else gray.T).astype(np.float32)
    n_along = img.shape[1]
    period0 = cand["period"]
    half = int(max(2, 0.04 * prior))
    profile = img[max(0, cand["line"] - half):cand["line"] + half + 1].mean(axis=0)
    w = int(max(4 * period0, 64))
    ok = np.zeros(n_along, bool)
    for s in range(0, n_along - w, max(1, w // 4)):
        seg = profile[s:s + w]
        if seg.std() < 15:
            continue
        freqs, power, nfft = _spectrum(seg, oversample=4)
        period, prom = _peak(freqs, power, nfft, max(3.0, period0 * 0.8), period0 * 1.25)
        if period and prom > 4 and abs(period - period0) / period0 < 0.08:
            ok[s:s + w] = True
    centre = (cand["start"] + cand["end"]) // 2
    a, b = cand["start"], cand["end"]
    if ok[centre]:
        a = b = centre
        while a > 0 and ok[a - 1]:
            a -= 1
        while b < n_along - 1 and ok[b + 1]:
            b += 1
    freqs, power, nfft = _spectrum(profile[a:b])
    period, prom = _peak(freqs, power, nfft, period0 * 0.87, period0 * 1.15)
    return {**cand, "period": period or period0, "start": int(a), "end": int(b), "prominence": prom}


def _tick_spacing(gray, cand, px_per_cm):
    """Look for 1 mm ticks next to the band; (px/cm from ticks, prominence) or (None, 0)."""
    if px_per_cm / 10 < 3:
        return None, 0.0
    img = (gray if cand["edge"] in ("top", "bottom") else gray.T).astype(np.float32)
    n_lines = img.shape[0]
    a, b = cand["start"], cand["end"]
    r0, r1 = max(0, int(cand["line"] - 0.8 * px_per_cm)), min(n_lines, int(cand["line"] + 0.8 * px_per_cm))
    best = (None, 0.0)
    for line in range(r0, r1, max(1, int(px_per_cm / 50))):
        prof = np.diff(img[line:line + max(2, int(px_per_cm / 40)), a:b].mean(axis=0))
        freqs, power, nfft = _spectrum(prof)
        period, prom = _peak(freqs, power, nfft, px_per_cm / 10 * 0.96, px_per_cm / 10 * 1.04, context=2.5)
        if period and prom > best[1]:
            best = (10 * period, prom)
    return best


def _ruler_line(cand, px_per_cm):
    along = [cand["start"], cand["end"]]
    if cand["edge"] in ("top", "bottom"):
        return [[along[0], cand["line"]], [along[1], cand["line"]]]
    return [[cand["line"], along[0]], [cand["line"], along[1]]]


def find_ruler(gray, prior):
    """First confirmed CV ruler reading as dict(px_per_cm, line, span_cm, confirmed) or None.

    `confirmed` means the millimetre ticks and the centimetre period agree within 3%. Colour
    charts can pass that test over short spans, so short rulers still need the sheet to agree.
    """
    for cand in _ruler_candidates(gray, prior)[:6]:
        cand = _extend_and_refine(gray, cand, prior)
        period = cand["period"]
        readings = {"1mm": period * 10, "1cm": period, "2cm": period / 2, "5mm": period * 2}
        if not period or period <= 0:
            continue
        name, px_per_cm = min(readings.items(), key=lambda kv: abs(np.log(kv[1] / prior)))
        ticks, prominence = _tick_spacing(gray, cand, px_per_cm)
        confirmed = bool(ticks and prominence > 8 and abs(ticks / px_per_cm - 1) < 0.03)
        if confirmed:
            px_per_cm = ticks
        span_cm = (cand["end"] - cand["start"]) / px_per_cm
        if span_cm < 3 or not confirmed:
            continue
        return {"px_per_cm": float(px_per_cm), "line": _ruler_line(cand, px_per_cm), "span_cm": float(span_cm),
                "confirmed": confirmed, "reading": name, "tick_prominence": float(prominence)}
    return None


# --------------------------------------------------------------------------- short scale bars (no AI)

BAR_SEARCH = 0.12           # a scale bar's centimetre pitch is looked for within this share of the sheet's scale
BAR_MIN_LINES = 5           # centimetre lines in a row that must fit (a 4 cm span at least)
BAR_MIN_ROWS = 3            # neighbouring image rows that must show the same lattice
BAR_EDGE = 35.0             # grey-level step that counts as a line or a block edge of the bar
BAR_MAX_COLOUR = 40.0       # mean saturation (0-255) above which the strip is a colour chart, not a scale bar


def _lattice_fit(positions, strengths, pitch):
    """Fit x = start + k * pitch to edge positions: dict(pitch, start, first, last, rms) or None.

    The phase is where the edges agree best modulo the pitch. Edges away from the lattice are left out, more
    strictly on each pass, and only the longest unbroken run of lattice lines is kept: a stray edge of a logo or
    a label a few centimetres away can sit on the lattice by chance and would tilt the fit. The pitch is the
    least-squares slope over that run.
    """
    if len(positions) < BAR_MIN_LINES:
        return None
    angle = 2 * np.pi * positions / pitch
    start = (np.angle(np.sum(strengths * np.exp(1j * angle))) % (2 * np.pi)) * pitch / (2 * np.pi)
    for tolerance in (0.08, 0.05, 0.03):
        k = np.round((positions - start) / pitch)
        keep = np.abs(positions - (start + k * pitch)) < max(tolerance * pitch, 2.5)
        if len(np.unique(k[keep])) < BAR_MIN_LINES:
            return None
        first, last = _longest_run(k[keep])
        keep &= (k >= first) & (k <= last)
        if last - first + 1 < BAR_MIN_LINES:
            return None
        pitch, start = np.polyfit(k[keep], positions[keep], 1)
    residual = positions[keep] - (start + k[keep] * pitch)
    return {"pitch": float(pitch), "start": float(start), "first": int(first), "last": int(last),
            "rms": float(np.sqrt(np.mean(residual ** 2)))}


def _longest_run(ks):
    """(first, last) of the longest run of consecutive integers in ks."""
    ks = sorted(set(int(k) for k in ks))
    best = run = (ks[0], ks[0])
    for k in ks[1:]:
        run = (run[0], k) if k == run[1] + 1 else (k, k)
        if run[1] - run[0] > best[1] - best[0]:
            best = run
    return best


def _row_lattice(profile, prior):
    """The centimetre lattice of strong edges along one image row, or None: dict(pitch, x0, x1, lines).

    A scale bar is drawn in lines and blocks a whole number of centimetres apart, so the positions of its strong
    edges lie on a lattice whose pitch is the scale. Edges are located to a fraction of a pixel, which makes the
    pitch far more exact than the spectrum of so short an object could be.
    """
    smooth = np.convolve(profile, np.ones(3) / 3, mode="same")
    step = np.abs(smooth[2:] - smooth[:-2])                           # centred difference, at index i + 1
    peaks = np.where((step[1:-1] >= step[:-2]) & (step[1:-1] > step[2:]) & (step[1:-1] > BAR_EDGE))[0] + 1
    if len(peaks) < BAR_MIN_LINES:
        return None
    y0, y1, y2 = step[peaks - 1], step[peaks], step[peaks + 1]
    denom = y0 - 2 * y1 + y2
    positions = peaks + 1 + np.where(denom != 0, 0.5 * (y0 - y2) / np.where(denom != 0, denom, 1), 0.0)
    strengths = step[peaks]
    # A printed line a few pixels wide gives an edge on each side: join the two into the line's centre. (Where a
    # white block starts at the line only one edge shows, half a line width off; the fit's tolerance covers that.)
    joined, weights = [positions[0] * strengths[0]], [strengths[0]]
    last = positions[0]
    for position, strength in zip(positions[1:], strengths[1:]):
        if position - last < max(0.05 * prior, 5.5):          # a line's two edges; never less than their blur
            joined[-1] += position * strength
            weights[-1] += strength
        else:
            joined.append(position * strength)
            weights.append(strength)
        last = position
    strengths = np.array(weights)
    positions = np.array(joined) / strengths
    if len(positions) < BAR_MIN_LINES:
        return None
    best = None
    # Candidate pitches come from the gaps between edges themselves (a gap is 1, 2, 3 ... pitches long).
    gaps = (positions[None, :] - positions[:, None]).ravel()
    lo, hi = prior * (1 - BAR_SEARCH), prior * (1 + BAR_SEARCH)
    candidates = np.concatenate([gaps[(gaps / n > lo) & (gaps / n < hi)] / n for n in (1, 2, 3, 4, 5)])
    if not len(candidates):
        return None
    counts, edges = np.histogram(candidates, bins=max(8, int((hi - lo) / (0.01 * prior))), range=(lo, hi))
    counts = counts + np.append(counts[1:], 0) + np.append(0, counts[:-1])      # a pitch on a bin border counts once
    for index in np.argsort(counts)[::-1][:4]:
        if counts[index] < BAR_MIN_LINES - 1:
            break
        fit = _lattice_fit(positions, strengths, (edges[index] + edges[index + 1]) / 2)
        if fit is None or not lo < fit["pitch"] < hi or fit["rms"] > 0.02 * fit["pitch"]:
            continue
        lines = fit["last"] - fit["first"] + 1
        if best is None or lines > best["lines"]:
            best = {"pitch": fit["pitch"], "x0": fit["start"] + fit["first"] * fit["pitch"],
                    "x1": fit["start"] + fit["last"] * fit["pitch"], "lines": lines}
    return best


def _bar_ticks(profile, pitch):
    """True when some centimetre of the profile shows ten evenly spaced millimetre ticks (most bars have one)."""
    if pitch / 10 < 3:
        return False
    cell = int(pitch)
    for s in range(0, max(1, len(profile) - cell), max(1, cell // 4)):
        freqs, power, nfft = _spectrum(np.diff(profile[s:s + cell]))
        period, prominence = _peak(freqs, power, nfft, pitch / 10 * 0.94, pitch / 10 * 1.06, context=2.5)
        if period and prominence > 12:
            return True
    return False


def find_scale_bar(rgb, gray, prior):
    """A short printed scale bar (a few centimetres of lines or chequered blocks) near an edge of the image.

    find_ruler needs millimetre ticks along the whole ruler, which the 5 or 10 cm bars many herbaria photograph
    beside the sheet do not have: they mark whole centimetres, with millimetres in one of them at most. Such a bar
    is found here as a lattice of strong edges one centimetre apart that (1) repeats over several neighbouring
    rows, (2) is black and white, unlike a colour chart, and (3) has a pitch near the sheet's own scale.
    Returns dict(px_per_cm, line, span_cm, confirmed, reading, ticks) or None.
    """
    import cv2

    saturation = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1]
    found = []
    half = max(1, int(prior / 60))
    for edge in ("top", "bottom", "left", "right"):
        img = (gray if edge in ("top", "bottom") else gray.T).astype(np.float32)
        sat = saturation if edge in ("top", "bottom") else saturation.T
        n_lines = img.shape[0]
        band = int(0.16 * n_lines)
        step = max(2, int(prior / 25))
        lines = list(range(half, band, step) if edge in ("top", "left") else range(n_lines - band, n_lines - half, step))
        rows = []
        for line in lines:
            profile = img[line - half:line + half + 1].mean(axis=0)
            if np.percentile(profile, 98) - np.percentile(profile, 2) < 2 * BAR_EDGE:
                continue                                              # a flat row: background or plain paper
            fit = _row_lattice(profile, prior)
            if fit is None:
                continue
            a, b = int(max(0, fit["x0"])), int(min(len(profile), fit["x1"] + 1))
            if sat[line - half:line + half + 1, a:b].mean() > BAR_MAX_COLOUR:
                continue
            rows.append({**fit, "line": line, "profile": profile[a:b]})
        # The same bar seen in neighbouring rows: group rows whose spans overlap and whose pitches agree.
        rows.sort(key=lambda r: r["line"])
        groups = []
        for row in rows:
            for group in groups:
                last = group[-1]
                if (row["line"] - last["line"] <= 3 * step and abs(row["pitch"] / last["pitch"] - 1) < 0.02
                        and min(row["x1"], last["x1"]) - max(row["x0"], last["x0"]) > 2 * row["pitch"]):
                    group.append(row)
                    break
            else:
                groups.append([row])
        for group in groups:
            if len(group) < BAR_MIN_ROWS:
                continue
            pitch = float(np.median([r["pitch"] for r in group]))
            longest = max(group, key=lambda r: r["lines"])
            middle = group[len(group) // 2]
            ticks = any(_bar_ticks(r["profile"], pitch) for r in group)
            cand = {"edge": edge, "line": middle["line"], "start": longest["x0"], "end": longest["x1"]}
            found.append({"px_per_cm": pitch, "line": _ruler_line(cand, pitch), "span_cm": float(longest["lines"] - 1),
                          "confirmed": True, "reading": "scale bar", "ticks": ticks, "rows": len(group),
                          "spread": float(np.std([r["pitch"] for r in group]) / pitch)})
    if not found:
        return None
    # Millimetre ticks settle it; otherwise the bar seen in the most rows wins.
    return max(found, key=lambda f: (f["ticks"], f["rows"]))


# --------------------------------------------------------------------------- ruler from an AI hint

def ruler_from_hint(gray, box, rough_px_per_cm):
    """Measure the ruler inside an AI-provided box [x0, y0, x1, y1] near a rough scale.

    Tries both orientations and every thin band across the box, keeping the band whose
    millimetre (or centimetre) period is most prominent within HINT_TOLERANCE of the rough scale.
    """
    h, w = gray.shape
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, x1 = max(0, min(x0, x1)), min(w, max(x0, x1))
    y0, y1 = max(0, min(y0, y1)), min(h, max(y0, y1))
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    lo, hi = rough_px_per_cm * (1 - HINT_TOLERANCE), rough_px_per_cm * (1 + HINT_TOLERANCE)
    best = None
    for edge, (n_lines, start, end, first) in (("top", (y1 - y0, x0, x1, y0)), ("left", (x1 - x0, y0, y1, x0))):
        if end - start < 3 * lo:
            continue
        half = max(1, int(rough_px_per_cm / 40))
        for line in range(first, first + n_lines, max(1, int(rough_px_per_cm / 30))):
            prof = _band_profile(gray, edge, line, half, start, end)
            diff = np.diff(prof)
            for unit_cm in TICK_UNITS_CM.values():
                if lo * unit_cm < 3:
                    continue
                x = diff if unit_cm < 0.4 else prof
                freqs, power, nfft = _spectrum(x)
                period, prom = _peak(freqs, power, nfft, lo * unit_cm, hi * unit_cm)
                if period and (best is None or prom > best["prominence"]):
                    best = {"px_per_cm": period / unit_cm, "prominence": prom, "edge": edge, "line": line,
                            "start": start, "end": end}
    if best is None or best["prominence"] < 6:
        return None
    return {"px_per_cm": float(best["px_per_cm"]), "line": _ruler_line(best, best["px_per_cm"]),
            "span_cm": float((best["end"] - best["start"]) / best["px_per_cm"]), "confirmed": True,
            "reading": "hint"}


# --------------------------------------------------------------------------- decision

def calibrate(rgb, ruler_hint=None, sheet_hint=None):
    """Calibration for an RGB numpy image.

    `ruler_hint` = {"box": [...], "px_per_cm": rough or None} and `sheet_hint` = [x0, y0, x1, y1] come
    from the layout pass; the sheet hint is only used when the paper colour cannot be found.
    """
    import cv2

    flags = []
    sheet = sheet_estimate(rgb)
    if not sheet.get("ok") and sheet_hint:
        x0, y0, x1, y1 = sheet_hint
        w, h = rgb.shape[1], rgb.shape[0]
        sheet = {"ok": True, "box": list(sheet_hint), "width_px": x1 - x0, "height_px": y1 - y0,
                 "touches_border": x0 <= 2 or x1 >= w - 2, "height_touches": y0 <= 2 or y1 >= h - 2}
        flags.append("sheet edges taken from the AI layout (paper colour not recognised)")
    sheet_px = sheet["width_px"] / SHEET_WIDTH_CM if sheet.get("ok") else None
    sheet_reliable = bool(sheet_px) and not sheet["touches_border"]
    if sheet.get("ok") and sheet["touches_border"]:
        flags.append("sheet edge touches the image border")
    if sheet_reliable and not sheet["height_touches"]:
        height_px = sheet["height_px"] / SHEET_HEIGHT_CM
        if abs(height_px / sheet_px - 1) > 0.08:
            flags.append(f"sheet is not standard 29 x 42 cm (height suggests {height_px:.1f} px/cm)")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    prior = sheet_px or max(rgb.shape[:2]) / SHEET_HEIGHT_CM

    hinted = None
    if ruler_hint and ruler_hint.get("box"):
        # A trained model gives only the ruler's box; the sheet width then says roughly what to expect.
        hinted = ruler_from_hint(gray, ruler_hint["box"], float(ruler_hint.get("px_per_cm") or prior))
        if hinted is None:
            flags.append("AI-marked ruler could not be measured")
        elif sheet_px:
            off = hinted["px_per_cm"] / sheet_px - 1
            if abs(off) > (HINT_SHEET_AGREEMENT if sheet_reliable else HINT_CUT_SHEET_AGREEMENT):
                flags.append(f"AI-marked ruler ignored ({hinted['px_per_cm']:.1f} px/cm is {100 * off:+.0f}% "
                             f"from the sheet width)")
                hinted = None
    found = find_ruler(gray, prior)
    clear = False
    if found:
        ratio = found["px_per_cm"] / sheet_px if sheet_px else 1.0
        agrees = bool(sheet_px) and abs(ratio - 1) <= RULER_AGREEMENT
        clear = (found["span_cm"] >= LONG_RULER_CM and found["tick_prominence"] >= LONG_RULER_TICKS
                 and abs(ratio - 1) <= LONG_RULER_AGREEMENT)
        if not (agrees or clear):
            if not hinted:
                flags.append(f"ruler-like pattern ignored ({found['px_per_cm']:.1f} px/cm, "
                             f"{found['span_cm']:.0f} cm long, does not match the sheet)")
            found = None
    bar = None
    if not hinted and not (found and clear):
        # No long, clear ruler: look for a short scale bar. The sheet's own scale says where to look; the image
        # height is a second guess, for sheets whose paper could not be outlined (a plant covering most of it).
        guess = max(rgb.shape[:2]) / SHEET_HEIGHT_CM
        bars = [find_scale_bar(rgb, gray, near)
                for near in [prior] + ([guess] if abs(guess / prior - 1) > BAR_SEARCH / 2 else [])]
        bar = max((b for b in bars if b), key=lambda b: (b["ticks"], b["rows"]), default=None)
        if bar and sheet_px and abs(bar["px_per_cm"] / sheet_px - 1) > BAR_SEARCH and not bar["ticks"]:
            flags.append(f"scale-bar-like pattern ignored ({bar['px_per_cm']:.1f} px/cm, does not match the sheet)")
            bar = None
        if bar and found and abs(bar["px_per_cm"] / found["px_per_cm"] - 1) > 0.02:
            # Two readings. A bar with its own millimetre ticks beats a ruler-like pattern that was only believed
            # because it matched the sheet width (printed label text can pass for one); otherwise the ruler stays.
            if bar["ticks"]:
                flags.append(f"ruler-like pattern ignored ({found['px_per_cm']:.1f} px/cm; the scale bar reads "
                             f"{bar['px_per_cm']:.1f})")
                found = None
            else:
                bar = None
        if bar:
            flags[:] = [f for f in flags if not f.endswith("does not match the sheet)")]
    ruler = hinted or found or bar
    if hinted and found and abs(hinted["px_per_cm"] / found["px_per_cm"] - 1) > RULER_AGREEMENT and sheet_px:
        # The AI's tick numbers are often misread, which locks the hinted reading onto a wrong period;
        # keep whichever reading is closer to the sheet width.
        ruler = min((hinted, found), key=lambda r: abs(math.log(r["px_per_cm"] / sheet_px)))
        other = found if ruler is hinted else hinted
        flags.append(f"two ruler readings disagree ({other['px_per_cm']:.1f} px/cm not used)")

    uncertain = []
    result = Calibration(px_per_cm=None, source="none", sheet_px_per_cm=sheet_px,
                         sheet_box=sheet.get("box") if sheet.get("ok") else None, flags=flags, uncertain=uncertain)
    if ruler:
        result.px_per_cm, result.source = ruler["px_per_cm"], "ruler"
        result.ruler_px_per_cm, result.ruler_line, result.ruler_span_cm = (ruler["px_per_cm"], ruler["line"],
                                                                           ruler["span_cm"])
        if sheet_px and abs(ruler["px_per_cm"] / sheet_px - 1) > RULER_AGREEMENT:
            flags.append(f"sheet width disagrees with the ruler by {100 * (sheet_px / ruler['px_per_cm'] - 1):+.0f}%")
            if not sheet["touches_border"]:
                uncertain.append("the ruler and the sheet width disagree")
    elif sheet_px:
        # The lab protocol also uses the sheet when it touches the image border; it is flagged above.
        result.px_per_cm, result.source = sheet_px, "sheet"
        uncertain += [f for f in flags if f.startswith(("sheet edge touches", "sheet is not standard",
                                                       "sheet edges taken from", "AI-marked ruler",
                                                       "ruler-like pattern"))]
    for trait, minimum in MIN_PX_PER_CM.items():
        if result.px_per_cm and result.px_per_cm < minimum:
            flags.append(f"resolution too low for {trait.replace('_', ' ')} ({result.px_per_cm:.0f} < {minimum:g} px/cm)")
    return result


def manual_calibration(previous, a, b, length_cm):
    """A scale set by a person: two points a known distance apart (like Set Scale in ImageJ)."""
    span_px = math.hypot(b[0] - a[0], b[1] - a[1])
    if length_cm <= 0 or span_px < 5:
        raise ValueError("Mark two points a known distance apart")
    data = dict(previous or {})
    data.update(px_per_cm=span_px / length_cm, source="manual", ruler_px_per_cm=span_px / length_cm,
                ruler_line=[list(a), list(b)], ruler_span_cm=float(length_cm), uncertain=[],
                flags=[f for f in data.get("flags", []) if not f.startswith("resolution too low")])
    return Calibration.from_dict(data)


def load_rgb(path):
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))
