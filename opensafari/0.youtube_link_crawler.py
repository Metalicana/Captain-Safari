#!/usr/bin/env python3
"""Collect YouTube candidate links for the OpenSafari data pipeline.

The downloader in ``1.download_preprocess.py`` expects direct MP4 links, while
YouTube discovery naturally starts from watch URLs. This script gathers public
watch URLs plus metadata into a CSV so candidates can be reviewed, converted, or
handled by a later yt-dlp-backed download stage.
"""

import argparse
import csv
import html
import json
import logging
import os
import re
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in minimal envs.
    yaml = None


YOUTUBE_WATCH = "https://www.youtube.com/watch?v={video_id}"
YOUTUBE_API_SEARCH = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_API_VIDEOS = "https://www.googleapis.com/youtube/v3/videos"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)


@dataclass
class Candidate:
    video_id: str
    url: str
    title: str = ""
    channel: str = ""
    duration_seconds: Optional[int] = None
    view_count: Optional[int] = None
    published: str = ""
    query: str = ""
    source: str = ""
    thumbnail: str = ""
    description: str = ""

    def text_for_filtering(self) -> str:
        return " ".join([self.title, self.channel, self.description]).lower()


def load_config(path: Path) -> Dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required for --config. Install pyyaml or pass CLI args directly.")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def normalize_video_id(raw: str) -> Optional[str]:
    match = re.search(r"^[a-zA-Z0-9_-]{11}$", raw)
    return raw if match else None


def parse_duration_to_seconds(value) -> Optional[int]:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    parts = text.split(":")
    if all(part.isdigit() for part in parts):
        seconds = 0
        for part in parts:
            seconds = seconds * 60 + int(part)
        return seconds
    return None


def parse_iso8601_duration(value: str) -> Optional[int]:
    if not value:
        return None
    match = re.fullmatch(
        r"P(?:(?P<days>\d+)D)?"
        r"(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?",
        value,
    )
    if not match:
        return None
    days = int(match.group("days") or 0)
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def clean_text(value: str) -> str:
    return html.unescape(value or "").replace("\n", " ").strip()


def first_text(runs_container) -> str:
    if not runs_container:
        return ""
    if isinstance(runs_container, dict):
        if "simpleText" in runs_container:
            return str(runs_container.get("simpleText") or "")
        runs = runs_container.get("runs") or []
        return "".join(str(run.get("text", "")) for run in runs)
    return ""


def first_thumbnail(renderer: Dict) -> str:
    thumbs = (((renderer.get("thumbnail") or {}).get("thumbnails")) or [])
    if not thumbs:
        return ""
    return thumbs[-1].get("url", "")


def parse_int_from_text(text: str) -> Optional[int]:
    if not text:
        return None
    lowered = text.lower().replace(",", "")
    match = re.search(r"([\d.]+)\s*([kmb])?\s+views?", lowered)
    if not match:
        return None
    value = float(match.group(1))
    suffix = match.group(2)
    multiplier = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}.get(suffix, 1)
    return int(value * multiplier)


def find_balanced_json(text: str, marker: str) -> Optional[str]:
    idx = text.find(marker)
    if idx < 0:
        return None
    start = text.find("{", idx)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for pos in range(start, len(text)):
        char = text[pos]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : pos + 1]
    return None


def walk_json(obj) -> Iterable[Dict]:
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from walk_json(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk_json(value)


def html_search(query: str, cfg: Dict) -> List[Candidate]:
    search_cfg = cfg.get("search", {})
    url = (
        "https://www.youtube.com/results"
        f"?search_query={quote_plus(query)}"
        f"&hl={quote_plus(search_cfg.get('language', 'en-US'))}"
        f"&gl={quote_plus(search_cfg.get('region', 'US'))}"
    )
    request = Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    timeout = int(search_cfg.get("timeout_seconds", 20))
    with urlopen(request, timeout=timeout) as response:
        html = response.read().decode("utf-8", errors="replace")

    initial_data = find_balanced_json(html, "ytInitialData")
    if not initial_data:
        raise RuntimeError("Could not locate ytInitialData in YouTube search response.")
    data = json.loads(initial_data)

    candidates: List[Candidate] = []
    for node in walk_json(data):
        renderer = node.get("videoRenderer")
        if not renderer:
            continue
        video_id = normalize_video_id(str(renderer.get("videoId", "")))
        if not video_id:
            continue
        title = first_text(renderer.get("title"))
        channel = first_text(renderer.get("ownerText"))
        duration = parse_duration_to_seconds(first_text(renderer.get("lengthText")))
        views = parse_int_from_text(first_text(renderer.get("viewCountText")))
        published = first_text(renderer.get("publishedTimeText"))
        candidates.append(
            Candidate(
                video_id=video_id,
                url=YOUTUBE_WATCH.format(video_id=video_id),
                title=title,
                channel=channel,
                duration_seconds=duration,
                view_count=views,
                published=published,
                query=query,
                source="youtube_html_search",
                thumbnail=first_thumbnail(renderer),
                description=first_text(renderer.get("detailedMetadataSnippets", {})),
            )
        )
    return candidates


def ytdlp_search(query: str, cfg: Dict) -> List[Candidate]:
    search_cfg = cfg.get("search", {})
    max_results = int(search_cfg.get("max_results_per_query", 40))
    command = [
        "yt-dlp",
        "--flat-playlist",
        "--dump-json",
        f"ytsearch{max_results}:{query}",
    ]
    timeout = max(30, int(search_cfg.get("timeout_seconds", 20)) * max(1, max_results // 10))
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=timeout)
    candidates = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        video_id = normalize_video_id(str(item.get("id", "")))
        if not video_id:
            continue
        url = item.get("webpage_url") or YOUTUBE_WATCH.format(video_id=video_id)
        candidates.append(
            Candidate(
                video_id=video_id,
                url=url,
                title=item.get("title") or "",
                channel=item.get("channel") or item.get("uploader") or "",
                duration_seconds=parse_duration_to_seconds(item.get("duration")),
                view_count=item.get("view_count"),
                published=str(item.get("upload_date") or item.get("release_date") or ""),
                query=query,
                source="yt-dlp_search",
                thumbnail=item.get("thumbnail") or "",
                description=item.get("description") or "",
            )
        )
    return candidates


def youtube_api_search(query: str, cfg: Dict) -> List[Candidate]:
    search_cfg = cfg.get("search", {})
    api_key_env = search_cfg.get("api_key_env", "YOUTUBE_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Set {api_key_env} to use backend=api.")

    target_results = int(search_cfg.get("max_results_per_query", 40))
    per_page = min(50, target_results)
    base_params = {
        "part": "snippet",
        "type": "video",
        "q": query,
        "maxResults": per_page,
        "key": api_key,
        "safeSearch": search_cfg.get("safe_search", "moderate"),
        "relevanceLanguage": search_cfg.get("language", "en-US").split("-")[0],
        "regionCode": search_cfg.get("region", "US"),
        "videoEmbeddable": search_cfg.get("video_embeddable", "true"),
        "order": search_cfg.get("order", "relevance"),
    }
    if search_cfg.get("published_after"):
        base_params["publishedAfter"] = search_cfg["published_after"]
    if search_cfg.get("published_before"):
        base_params["publishedBefore"] = search_cfg["published_before"]

    timeout = int(search_cfg.get("timeout_seconds", 20))

    candidates = []
    next_page_token = None
    while len(candidates) < target_results:
        params = dict(base_params)
        if next_page_token:
            params["pageToken"] = next_page_token
        request = Request(
            f"{YOUTUBE_API_SEARCH}?{urlencode(params)}",
            headers={"User-Agent": DEFAULT_USER_AGENT},
        )
        with urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))

        for item in data.get("items", []):
            video_id = normalize_video_id(str((item.get("id") or {}).get("videoId", "")))
            if not video_id:
                continue
            snippet = item.get("snippet") or {}
            thumbs = snippet.get("thumbnails") or {}
            thumbnail = ""
            for key in ("maxres", "standard", "high", "medium", "default"):
                if key in thumbs:
                    thumbnail = thumbs[key].get("url", "")
                    break
            candidates.append(
                Candidate(
                    video_id=video_id,
                    url=YOUTUBE_WATCH.format(video_id=video_id),
                    title=clean_text(snippet.get("title") or ""),
                    channel=clean_text(snippet.get("channelTitle") or ""),
                    published=snippet.get("publishedAt") or "",
                    query=query,
                    source="youtube_data_api_search",
                    thumbnail=thumbnail,
                    description=clean_text(snippet.get("description") or ""),
                )
            )
            if len(candidates) >= target_results:
                break
        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

    if search_cfg.get("enrich_videos", True):
        enrich_youtube_api_videos(candidates, cfg, api_key)
    return candidates


def enrich_youtube_api_videos(candidates: List[Candidate], cfg: Dict, api_key: str) -> None:
    if not candidates:
        return
    search_cfg = cfg.get("search", {})
    timeout = int(search_cfg.get("timeout_seconds", 20))
    by_id = {candidate.video_id: candidate for candidate in candidates}
    ids = list(by_id)
    for start in range(0, len(ids), 50):
        batch_ids = ids[start : start + 50]
        params = {
            "part": "contentDetails,statistics,snippet",
            "id": ",".join(batch_ids),
            "key": api_key,
        }
        request = Request(
            f"{YOUTUBE_API_VIDEOS}?{urlencode(params)}",
            headers={"User-Agent": DEFAULT_USER_AGENT},
        )
        with urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))

        for item in data.get("items", []):
            candidate = by_id.get(item.get("id"))
            if not candidate:
                continue
            snippet = item.get("snippet") or {}
            statistics = item.get("statistics") or {}
            content_details = item.get("contentDetails") or {}
            candidate.duration_seconds = parse_iso8601_duration(content_details.get("duration", ""))
            if statistics.get("viewCount"):
                candidate.view_count = int(statistics["viewCount"])
            candidate.title = clean_text(snippet.get("title") or candidate.title)
            candidate.channel = clean_text(snippet.get("channelTitle") or candidate.channel)
            candidate.description = clean_text(snippet.get("description") or candidate.description)


def collect_query(query: str, cfg: Dict) -> List[Candidate]:
    backend = (cfg.get("search", {}).get("backend") or "auto").lower()
    if backend == "api":
        return youtube_api_search(query, cfg)
    if backend in ("auto", "yt-dlp") and shutil.which("yt-dlp"):
        try:
            return ytdlp_search(query, cfg)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            if backend == "yt-dlp":
                raise RuntimeError(f"yt-dlp search failed for query '{query}': {exc}") from exc
            logging.warning("yt-dlp failed for %r; falling back to HTML search: %s", query, exc)
    if backend == "yt-dlp":
        raise RuntimeError("backend=yt-dlp requested, but yt-dlp is not on PATH.")
    return html_search(query, cfg)


def rejection_reason(candidate: Candidate, cfg: Dict) -> str:
    filters = cfg.get("filters", {})
    duration = candidate.duration_seconds
    min_duration = filters.get("min_duration_seconds")
    max_duration = filters.get("max_duration_seconds")
    if duration is not None and min_duration is not None and duration < int(min_duration):
        return f"duration_lt_{min_duration}"
    if duration is not None and max_duration is not None and duration > int(max_duration):
        return f"duration_gt_{max_duration}"

    text = candidate.text_for_filtering()
    required = [kw.lower() for kw in filters.get("require_any_keywords", [])]
    rejected = [kw.lower() for kw in filters.get("reject_any_keywords", [])]
    if required and not any(keyword in text for keyword in required):
        return "missing_required_keyword"
    for keyword in rejected:
        if keyword in text:
            return f"rejected_keyword:{keyword}"
    return ""


def matched_keywords(candidate: Candidate, cfg: Dict) -> str:
    text = candidate.text_for_filtering()
    keywords = list(cfg.get("filters", {}).get("require_any_keywords", []))
    return "|".join(keyword for keyword in keywords if keyword.lower() in text)


def read_existing_ids(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return {row.get("video_id", "") for row in csv.DictReader(handle) if row.get("video_id")}


def read_seen_ids(paths: Iterable[Path]) -> set:
    seen = set()
    for path in paths:
        seen.update(read_existing_ids(path))
    return seen


def write_candidates(candidates: List[Candidate], output_csv: Path, cfg: Dict, append: bool) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_id",
        "url",
        "mp4",
        "title",
        "channel",
        "duration_seconds",
        "view_count",
        "published",
        "query",
        "source",
        "thumbnail",
        "matched_keywords",
        "rejected_reason",
        "collected_at",
    ]
    mode = "a" if append and output_csv.exists() else "w"
    with output_csv.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        collected_at = datetime.now(timezone.utc).isoformat()
        for item in candidates:
            writer.writerow(
                {
                    "video_id": item.video_id,
                    "url": item.url,
                    "mp4": item.url,
                    "title": item.title,
                    "channel": item.channel,
                    "duration_seconds": item.duration_seconds or "",
                    "view_count": item.view_count or "",
                    "published": item.published,
                    "query": item.query,
                    "source": item.source,
                    "thumbnail": item.thumbnail,
                    "matched_keywords": matched_keywords(item, cfg),
                    "rejected_reason": rejection_reason(item, cfg),
                    "collected_at": collected_at,
                }
            )


def summarize_candidates(candidates: List[Candidate], cfg: Dict) -> Counter:
    return Counter(rejection_reason(item, cfg) or "accepted" for item in candidates)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect YouTube candidate links for OpenSafari.")
    parser.add_argument("--config", type=str, default="youtube_crawl_config.yaml")
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--query", action="append", default=None, help="Search query. May be repeated.")
    parser.add_argument("--max-results-per-query", type=int, default=None)
    parser.add_argument("--backend", choices=["auto", "yt-dlp", "html", "api"], default=None)
    parser.add_argument("--append", action="store_true", help="Append to an existing CSV.")
    parser.add_argument("--seen-csv", action="append", default=[], help="CSV whose video_id values should be skipped. May be repeated.")
    parser.add_argument("--accepted-only", action="store_true", help="Write only rows that pass filters.")
    parser.add_argument("--include-rejected", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", help="Print summary without writing CSV.")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_arg_parser().parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists() and not cfg_path.is_absolute():
        script_relative = Path(__file__).resolve().parent / cfg_path
        if script_relative.exists():
            cfg_path = script_relative
    cfg = load_config(cfg_path) if cfg_path.exists() else {}
    cfg.setdefault("search", {})
    cfg.setdefault("filters", {})
    cfg.setdefault("resume", {})

    if args.query:
        cfg["search"]["queries"] = args.query
    if args.max_results_per_query is not None:
        cfg["search"]["max_results_per_query"] = args.max_results_per_query
    if args.backend is not None:
        cfg["search"]["backend"] = args.backend

    queries = cfg.get("search", {}).get("queries") or []
    if not queries:
        raise SystemExit("No queries supplied. Add search.queries to config or pass --query.")

    output_csv = Path(args.output_csv or cfg.get("output_csv", "youtube_candidates.csv"))
    seen_paths = [Path(path) for path in cfg.get("resume", {}).get("seen_csvs", [])]
    seen_paths.extend(Path(path) for path in args.seen_csv)
    if args.append and cfg.get("resume", {}).get("skip_existing_video_ids", True):
        seen_paths.append(output_csv)
    seen = read_seen_ids(seen_paths)
    if seen:
        logging.info("Loaded %d seen video IDs from %d CSV file(s).", len(seen), len(seen_paths))
    collected: List[Candidate] = []
    all_candidates: List[Candidate] = []
    skipped_seen = 0

    for query in queries:
        logging.info("Collecting YouTube candidates for query: %s", query)
        try:
            query_candidates = collect_query(query, cfg)
        except (HTTPError, URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            logging.error("Failed query %r: %s", query, exc)
            continue

        max_results = int(cfg.get("search", {}).get("max_results_per_query", 40))
        for item in query_candidates[:max_results]:
            if item.video_id in seen:
                skipped_seen += 1
                continue
            seen.add(item.video_id)
            all_candidates.append(item)
            if not args.accepted_only or not rejection_reason(item, cfg):
                collected.append(item)

        sleep_seconds = float(cfg.get("search", {}).get("sleep_seconds", 1.5))
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    summary = summarize_candidates(all_candidates, cfg)
    logging.info("Collected %d unique candidates.", len(all_candidates))
    if skipped_seen:
        logging.info("Skipped %d already-seen candidates.", skipped_seen)
    for reason, count in summary.most_common():
        logging.info("  %s: %d", reason, count)
    if args.accepted_only:
        logging.info("Writing %d accepted rows.", len(collected))
    else:
        logging.info("Writing %d rows including rejected candidates.", len(collected))

    if args.dry_run:
        for item in collected[:20]:
            reason = rejection_reason(item, cfg) or "accepted"
            print(f"{reason}\t{item.duration_seconds or ''}\t{item.video_id}\t{item.title}\t{item.url}")
        return 0

    write_candidates(collected, output_csv, cfg, append=args.append)
    logging.info("Wrote %s", output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
