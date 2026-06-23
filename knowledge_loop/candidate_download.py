from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import config
from .models import (
    CandidateDownloadManifest,
    CandidateDownloadRecord,
    CandidateDownloadWarning,
    YouTubeCandidate,
    YouTubeSearchRunSnapshot,
)


DEFAULT_DOWNLOAD_TOP_N = 3
DEFAULT_YTDLP_FORMAT = "bv*+ba/b"
DEFAULT_USE_BROWSER_COOKIES = True
DEFAULT_BROWSER_FOR_COOKIES = "chrome"
DEFAULT_COOKIES_PATH = str(config.PROJECT_ROOT / "chatbot_system" / "cookies.txt")


@dataclass(frozen=True)
class DownloadOutcome:
    local_video_path: Path
    yt_dlp_info_path: Path | None = None
    cookie_source: str | None = None


Downloader = Callable[[YouTubeCandidate, Path], DownloadOutcome]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, payload: dict[str, Any], pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2 if pretty else None, ensure_ascii=False, default=str)


def load_youtube_search_run(path: str | Path) -> YouTubeSearchRunSnapshot:
    return YouTubeSearchRunSnapshot.model_validate(_read_json(Path(path)))


def normalize_filename(value: str, fallback: str = "untitled_video") -> str:
    normalized = re.sub(r'[<>:"/\\|?*\x00-\x1f! ]+', "_", str(value or ""))
    normalized = re.sub(r"_{2,}", "_", normalized).strip("._ ")
    return normalized or fallback


def select_download_candidates(
    snapshot: YouTubeSearchRunSnapshot,
    *,
    top_n: int = DEFAULT_DOWNLOAD_TOP_N,
) -> tuple[list[YouTubeCandidate], list[CandidateDownloadWarning]]:
    selected: list[YouTubeCandidate] = []
    warnings: list[CandidateDownloadWarning] = []
    for candidate in snapshot.ranked_candidates:
        if len(selected) >= max(1, top_n):
            break
        if not candidate.source_url:
            warnings.append(
                CandidateDownloadWarning(
                    code="candidate_missing_source_url",
                    message="Candidate has no source URL and cannot be downloaded",
                    candidate_id=candidate.candidate_id,
                )
            )
            continue
        selected.append(candidate)

    if len(selected) < max(1, top_n):
        warnings.append(
            CandidateDownloadWarning(
                code="insufficient_candidates_for_download",
                message=f"Selected {len(selected)} candidates for download; requested {max(1, top_n)}",
            )
        )
    return selected, warnings


def _candidate_record_base(candidate: YouTubeCandidate, *, rank: int, download_dir: Path) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "rank": rank,
        "source_url": candidate.source_url,
        "youtube_video_id": candidate.youtube_video_id,
        "title": candidate.title,
        "channel_title": candidate.channel_title,
        "publish_date": candidate.publish_date,
        "duration_seconds": candidate.duration_seconds,
        "final_score": candidate.final_score,
        "llm_score": candidate.llm_score,
        "metadata_score": candidate.metadata_score,
        "download_dir": str(download_dir),
    }


def _write_candidate_record(download_dir: Path, record: CandidateDownloadRecord, pretty: bool = True) -> Path:
    path = download_dir / "download_record.json"
    _write_json(path, record.model_dump(mode="json"), pretty=pretty)
    return path


def _find_downloaded_video(download_dir: Path) -> Path | None:
    ignored_suffixes = {".json", ".part", ".ytdl", ".temp", ".tmp"}
    candidates = [
        path
        for path in download_dir.iterdir()
        if path.is_file() and path.suffix.lower() not in ignored_suffixes and not path.name.endswith(".part")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_size)


def _cookie_source(
    *,
    use_browser_cookies: bool,
    browser_for_cookies: str,
    cookiefile: str | None,
) -> str:
    if use_browser_cookies:
        return f"browser:{browser_for_cookies}"
    if cookiefile:
        return f"cookiefile:{cookiefile}"
    return "none"


def _build_base_ytdlp_opts(
    *,
    ytdlp_format: str,
    use_browser_cookies: bool,
    browser_for_cookies: str,
    cookiefile: str | None,
    quiet: bool,
) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "format": ytdlp_format,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": quiet,
        "no_warnings": False,
    }
    if use_browser_cookies:
        opts["cookies_from_browser"] = (browser_for_cookies,)
    elif cookiefile:
        opts["cookiefile"] = cookiefile
    return opts


def download_with_ytdlp(
    candidate: YouTubeCandidate,
    download_dir: Path,
    *,
    ytdlp_format: str = DEFAULT_YTDLP_FORMAT,
    use_browser_cookies: bool = DEFAULT_USE_BROWSER_COOKIES,
    browser_for_cookies: str = DEFAULT_BROWSER_FOR_COOKIES,
    cookiefile: str | None = DEFAULT_COOKIES_PATH,
    quiet: bool = False,
) -> DownloadOutcome:
    try:
        import yt_dlp  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - exercised only in real runtime.
        raise RuntimeError("yt-dlp is not installed in the active environment") from exc

    download_dir.mkdir(parents=True, exist_ok=True)
    info_path = download_dir / "yt_dlp_info.json"
    cookie_source = _cookie_source(
        use_browser_cookies=use_browser_cookies,
        browser_for_cookies=browser_for_cookies,
        cookiefile=cookiefile,
    )
    base_opts = _build_base_ytdlp_opts(
        ytdlp_format=ytdlp_format,
        use_browser_cookies=use_browser_cookies,
        browser_for_cookies=browser_for_cookies,
        cookiefile=cookiefile,
        quiet=quiet,
    )

    with yt_dlp.YoutubeDL(base_opts) as ydl:
        info = ydl.extract_info(candidate.source_url, download=False)

    title = candidate.title or candidate.candidate_id
    ext = "mp4"
    if isinstance(info, dict):
        title = str(info.get("title") or title)
        ext = str(info.get("ext") or ext)
    normalized_title = normalize_filename(title, fallback=candidate.candidate_id)
    download_opts = {
        **base_opts,
        "outtmpl": str(download_dir / f"{normalized_title}.%(ext)s"),
    }

    with yt_dlp.YoutubeDL(download_opts) as ydl:
        ydl.download([candidate.source_url])

    _write_json(info_path, info if isinstance(info, dict) else {"info": str(info)}, pretty=True)
    expected_mp4 = download_dir / f"{normalized_title}.mp4"
    fallback_ext = download_dir / f"{normalized_title}.{ext}"
    if expected_mp4.exists():
        downloaded = expected_mp4
    elif fallback_ext.exists():
        downloaded = fallback_ext
    else:
        downloaded = _find_downloaded_video(download_dir)
    if downloaded is None:
        raise RuntimeError(f"yt-dlp completed but no video file was found in {download_dir}")
    return DownloadOutcome(
        local_video_path=downloaded.resolve(),
        yt_dlp_info_path=info_path.resolve(),
        cookie_source=cookie_source,
    )


def build_ytdlp_downloader(
    *,
    ytdlp_format: str = DEFAULT_YTDLP_FORMAT,
    use_browser_cookies: bool = DEFAULT_USE_BROWSER_COOKIES,
    browser_for_cookies: str = DEFAULT_BROWSER_FOR_COOKIES,
    cookiefile: str | None = DEFAULT_COOKIES_PATH,
    quiet: bool = False,
) -> Downloader:
    def _download(candidate: YouTubeCandidate, download_dir: Path) -> DownloadOutcome:
        return download_with_ytdlp(
            candidate,
            download_dir,
            ytdlp_format=ytdlp_format,
            use_browser_cookies=use_browser_cookies,
            browser_for_cookies=browser_for_cookies,
            cookiefile=cookiefile,
            quiet=quiet,
        )

    return _download


def run_candidate_downloads(
    snapshot: YouTubeSearchRunSnapshot,
    *,
    search_run_path: str | None = None,
    state_root: str | Path | None = None,
    top_n: int = DEFAULT_DOWNLOAD_TOP_N,
    downloader: Downloader | None = None,
    dry_run: bool = False,
) -> CandidateDownloadManifest:
    generated_at = datetime.now(timezone.utc)
    generated = generated_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    resolved_state_root = Path(state_root or config.DEFAULT_STATE_ROOT).resolve()
    download_root = resolved_state_root / config.CANDIDATE_DOWNLOADS_DIR_NAME / f"download_run_{generated}"
    selected, warnings = select_download_candidates(snapshot, top_n=top_n)
    records: list[CandidateDownloadRecord] = []
    active_downloader = downloader or build_ytdlp_downloader()

    for rank, candidate in enumerate(selected, start=1):
        safe_title = normalize_filename(candidate.title, fallback=candidate.candidate_id)[:90]
        candidate_dir = download_root / f"{rank:02d}_{candidate.candidate_id}_{safe_title}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        base = _candidate_record_base(candidate, rank=rank, download_dir=candidate_dir)

        if dry_run:
            record = CandidateDownloadRecord(
                **base,
                status="skipped",
                failure_reason="dry_run",
            )
            records.append(record)
            _write_candidate_record(candidate_dir, record)
            continue

        try:
            outcome = active_downloader(candidate, candidate_dir)
            record = CandidateDownloadRecord(
                **base,
                status="downloaded",
                local_video_path=str(outcome.local_video_path),
                yt_dlp_info_path=str(outcome.yt_dlp_info_path) if outcome.yt_dlp_info_path else None,
                cookie_source=outcome.cookie_source,
                downloaded_at=datetime.now(timezone.utc),
            )
        except Exception as exc:  # noqa: BLE001
            record = CandidateDownloadRecord(
                **base,
                status="failed",
                failure_reason=str(exc),
            )
            warnings.append(
                CandidateDownloadWarning(
                    code="candidate_download_failed",
                    message=str(exc),
                    candidate_id=candidate.candidate_id,
                )
            )
        records.append(record)
        _write_candidate_record(candidate_dir, record)

    return CandidateDownloadManifest(
        generated_at=generated_at,
        loop_profile_id=snapshot.loop_profile_id,
        search_run_path=search_run_path,
        search_run_generated_at=snapshot.generated_at,
        target_gap=snapshot.target_gap,
        requested_top_n=max(1, top_n),
        selected_candidate_count=len(selected),
        successful_download_count=sum(1 for record in records if record.status == "downloaded"),
        failed_download_count=sum(1 for record in records if record.status == "failed"),
        skipped_download_count=sum(1 for record in records if record.status == "skipped"),
        download_root=str(download_root),
        records=records,
        warnings=warnings,
    )


def persist_download_manifest(
    manifest: CandidateDownloadManifest,
    state_root: str | Path | None = None,
    output_path: str | Path | None = None,
    pretty: bool = True,
) -> list[Path]:
    resolved_state_root = Path(state_root or config.DEFAULT_STATE_ROOT).resolve()
    download_dir = resolved_state_root / config.CANDIDATE_DOWNLOADS_DIR_NAME
    generated = manifest.generated_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    paths = [
        download_dir / f"download_manifest_{generated}.json",
        download_dir / "latest_download_manifest.json",
    ]
    if output_path:
        paths.append(Path(output_path).resolve())
    payload = manifest.model_dump(mode="json")
    for path in paths:
        _write_json(path, payload, pretty=pretty)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download top ranked loop candidates with yt-dlp.")
    parser.add_argument(
        "--search-run",
        default=str(config.DEFAULT_STATE_ROOT / config.SEARCH_RUNS_DIR_NAME / "latest_youtube_search.json"),
        help="YouTube search run JSON path",
    )
    parser.add_argument("--top-n", type=int, default=DEFAULT_DOWNLOAD_TOP_N)
    parser.add_argument("--state-root", default=str(config.DEFAULT_STATE_ROOT))
    parser.add_argument("--output", default=None, help="Optional extra output JSON path")
    parser.add_argument("--format", default=DEFAULT_YTDLP_FORMAT, help="yt-dlp format selector")
    parser.add_argument("--cookiefile", default=DEFAULT_COOKIES_PATH)
    parser.add_argument(
        "--browser-for-cookies",
        "--cookies-from-browser",
        dest="browser_for_cookies",
        default=DEFAULT_BROWSER_FOR_COOKIES,
        help="Browser name for yt-dlp cookies_from_browser",
    )
    parser.add_argument("--no-browser-cookies", action="store_true", help="Use --cookiefile instead of browser cookies")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write a skipped manifest without downloading")
    parser.add_argument("--compact", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    search_run_path = Path(args.search_run).resolve()
    snapshot = load_youtube_search_run(search_run_path)
    downloader = build_ytdlp_downloader(
        ytdlp_format=args.format,
        use_browser_cookies=not args.no_browser_cookies,
        browser_for_cookies=args.browser_for_cookies,
        cookiefile=args.cookiefile,
        quiet=args.quiet,
    )
    manifest = run_candidate_downloads(
        snapshot,
        search_run_path=str(search_run_path),
        state_root=args.state_root,
        top_n=max(1, args.top_n),
        downloader=downloader,
        dry_run=args.dry_run,
    )
    paths = persist_download_manifest(
        manifest,
        state_root=args.state_root,
        output_path=args.output,
        pretty=not args.compact,
    )

    print(f"Download profile: {manifest.loop_profile_id}")
    print(f"Target gap: {manifest.target_gap.gap_id} {manifest.target_gap.topic_key} {manifest.target_gap.facet}")
    print(f"Selected candidates: {manifest.selected_candidate_count}")
    print(f"Downloaded: {manifest.successful_download_count}")
    print(f"Failed: {manifest.failed_download_count}")
    print(f"Skipped: {manifest.skipped_download_count}")
    print(f"Warnings: {len(manifest.warnings)}")
    for record in manifest.records:
        print(f"{record.rank}. {record.status.upper()} {record.candidate_id} {record.title}")
        if record.local_video_path:
            print(f"   video: {record.local_video_path}")
        if record.cookie_source:
            print(f"   cookies: {record.cookie_source}")
        if record.failure_reason:
            print(f"   reason: {record.failure_reason}")
    for warning in manifest.warnings:
        print(f"WARNING {warning.code}: {warning.message}")
    for path in paths:
        print(f"Wrote: {path}")
    return 1 if manifest.failed_download_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
