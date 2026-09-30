"""Real-platform calibration profiles for Society-Core evidence packs."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

from .calibration import CalibrationTarget
from .hashing import stable_hash


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    weight = index - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    denominator = sqrt(x_var * y_var)
    if denominator == 0:
        return 0.0
    return numerator / denominator


@dataclass(frozen=True)
class PlatformSource:
    source_id: str
    platform: str
    query: str
    source_url: str
    collected_at_utc: str
    command: str
    caveats: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlatformForumRecord:
    source_id: str
    platform: str
    record_id: str
    title: str
    author: str
    url: str
    score: float = 0.0
    comment_count: int = 0
    labels: tuple[str, ...] = ()
    raw_ref: str | None = None


@dataclass(frozen=True)
class PlatformFeedbackRecord:
    source_id: str
    platform: str
    record_id: str
    title: str
    author: str
    url: str
    state: str = "unknown"
    comment_count: int = 0
    reaction_count: int = 0
    labels: tuple[str, ...] = ()
    raw_ref: str | None = None


@dataclass(frozen=True)
class PlatformMetricProfile:
    source_id: str
    platform: str
    record_count: int
    active_author_count: int
    total_comment_count: int
    mean_comments: float
    p90_comments: float
    zero_comment_share: float
    mean_score: float
    score_comment_correlation: float
    issue_closed_share: float
    label_diversity: int
    caveats: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlatformCalibrationReport:
    report_id: str
    sources: tuple[PlatformSource, ...]
    forum_records: tuple[PlatformForumRecord, ...]
    feedback_records: tuple[PlatformFeedbackRecord, ...]
    profiles: dict[str, PlatformMetricProfile]
    calibration_targets: tuple[CalibrationTarget, ...]
    caveats: tuple[str, ...] = ()

    def hash(self) -> str:
        return stable_hash(self)


def normalize_hn_records(source: PlatformSource, rows: list[dict[str, Any]]) -> tuple[PlatformForumRecord, ...]:
    records: list[PlatformForumRecord] = []
    for row in rows:
        record_id = _str(row.get("id") or row.get("objectID") or row.get("item_id"))
        if not record_id:
            continue
        records.append(
            PlatformForumRecord(
                source_id=source.source_id,
                platform=source.platform,
                record_id=record_id,
                title=_str(row.get("title")),
                author=_str(row.get("author") or row.get("by")),
                url=_str(row.get("url") or f"https://news.ycombinator.com/item?id={record_id}"),
                score=_float(row.get("score")),
                comment_count=_int(row.get("comments") or row.get("descendants")),
                labels=("hacker_news",),
                raw_ref=record_id,
            )
        )
    return tuple(records)


def normalize_reddit_records(source: PlatformSource, rows: list[dict[str, Any]]) -> tuple[PlatformForumRecord, ...]:
    records: list[PlatformForumRecord] = []
    for row in rows:
        record_id = _str(row.get("postId") or row.get("id") or row.get("name"))
        if not record_id:
            continue
        subreddit = _str(row.get("subreddit")).replace("r/", "")
        labels = tuple(label for label in ("reddit", subreddit, _str(row.get("post_hint"))) if label)
        records.append(
            PlatformForumRecord(
                source_id=source.source_id,
                platform=source.platform,
                record_id=record_id,
                title=_str(row.get("title")),
                author=_str(row.get("author")),
                url=_str(row.get("url") or f"https://www.reddit.com/comments/{record_id}/"),
                score=_float(row.get("score") or row.get("upvotes")),
                comment_count=_int(row.get("comments") or row.get("num_comments")),
                labels=labels,
                raw_ref=record_id,
            )
        )
    return tuple(records)


def _label_name(label: Any) -> str:
    if isinstance(label, dict):
        return _str(label.get("name"))
    return _str(label)


def normalize_github_issue_records(
    source: PlatformSource,
    rows: list[dict[str, Any]],
) -> tuple[PlatformFeedbackRecord, ...]:
    records: list[PlatformFeedbackRecord] = []
    for row in rows:
        record_id = _str(row.get("number") or row.get("id") or row.get("url"))
        if not record_id:
            continue
        labels = tuple(sorted(label for label in (_label_name(item) for item in row.get("labels", [])) if label))
        reactions = row.get("reactions") or {}
        if isinstance(reactions, dict):
            reaction_count = sum(_int(value) for value in reactions.values())
        else:
            reaction_count = 0
        author = row.get("author")
        if isinstance(author, dict):
            author_name = _str(author.get("login") or author.get("name"))
        else:
            author_name = _str(author)
        records.append(
            PlatformFeedbackRecord(
                source_id=source.source_id,
                platform=source.platform,
                record_id=record_id,
                title=_str(row.get("title")),
                author=author_name,
                url=_str(row.get("url")),
                state=_str(row.get("state"), "unknown").lower(),
                comment_count=_int(row.get("comments")),
                reaction_count=reaction_count,
                labels=labels,
                raw_ref=record_id,
            )
        )
    return tuple(records)


def build_platform_metric_profile(
    *,
    source: PlatformSource,
    forum_records: tuple[PlatformForumRecord, ...] = (),
    feedback_records: tuple[PlatformFeedbackRecord, ...] = (),
) -> PlatformMetricProfile:
    comments = [float(record.comment_count) for record in forum_records]
    comments.extend(float(record.comment_count) for record in feedback_records)
    scores = [float(record.score) for record in forum_records]
    score_comments = [float(record.comment_count) for record in forum_records]
    authors = {record.author for record in forum_records if record.author}
    authors.update(record.author for record in feedback_records if record.author)
    labels = {
        label
        for record in (*forum_records, *feedback_records)
        for label in record.labels
        if label
    }
    closed_count = sum(1 for record in feedback_records if record.state in {"closed", "merged"})
    record_count = len(forum_records) + len(feedback_records)
    caveats = list(source.caveats)
    if record_count < 10:
        caveats.append("small_platform_sample")
    if not forum_records and not feedback_records:
        caveats.append("empty_platform_sample")
    return PlatformMetricProfile(
        source_id=source.source_id,
        platform=source.platform,
        record_count=record_count,
        active_author_count=len(authors),
        total_comment_count=int(sum(comments)),
        mean_comments=_mean(comments),
        p90_comments=_quantile(comments, 0.9),
        zero_comment_share=sum(1 for value in comments if value <= 0) / max(1, len(comments)),
        mean_score=_mean(scores),
        score_comment_correlation=_correlation(scores, score_comments) if scores else 0.0,
        issue_closed_share=closed_count / max(1, len(feedback_records)) if feedback_records else 0.0,
        label_diversity=len(labels),
        caveats=tuple(caveats),
    )


def _target_interval(center: float, half_width: float, *, lower_floor: float = 0.0, upper_cap: float = 1.0) -> tuple[float, float]:
    return (_clamp(center - half_width, lower_floor, upper_cap), _clamp(center + half_width, lower_floor, upper_cap))


def calibration_targets_from_profiles(
    profiles: tuple[PlatformMetricProfile, ...],
) -> tuple[CalibrationTarget, ...]:
    usable = [profile for profile in profiles if profile.record_count > 0]
    if not usable:
        return ()
    total_records = sum(profile.record_count for profile in usable)
    total_comments = sum(profile.total_comment_count for profile in usable)
    zero_share = _mean([profile.zero_comment_share for profile in usable])
    mean_comments_value = _mean([profile.mean_comments for profile in usable])
    p90_comments_value = _mean([profile.p90_comments for profile in usable])
    label_diversity = sum(profile.label_diversity for profile in usable)
    positive_comment_share = total_comments / max(1, total_comments + total_records)
    reply_lower, reply_upper = _target_interval(positive_comment_share, 0.2)
    # Feed samples observe active posters, not all lurkers. This target intentionally
    # stays broad and source-labeled as a participation proxy.
    silent_center = _clamp(zero_share + 0.35)
    silent_lower, silent_upper = _target_interval(silent_center, 0.25)
    depth_center = _clamp((mean_comments_value / max(1.0, p90_comments_value)) * 3.0, 0.05, 8.0)
    depth_lower = max(0.01, depth_center * 0.35)
    depth_upper = min(8.0, max(depth_lower + 0.05, depth_center * 1.8))
    cascade_center = _clamp(1.0 - zero_share)
    cascade_lower, cascade_upper = _target_interval(cascade_center, 0.25)
    controversy_center = _clamp(label_diversity / max(20.0, total_records * 2.0), 0.01, 0.6)
    controversy_lower, controversy_upper = _target_interval(controversy_center, 0.12)
    source_label = "+".join(sorted({profile.platform for profile in usable}))
    source_url = "mixed:platform_calibration_report"
    return (
        CalibrationTarget("reply_share", reply_lower, reply_upper, f"empirical_reply_proxy:{source_label}", source_url),
        CalibrationTarget("silent_user_fraction", silent_lower, silent_upper, f"empirical_lurker_proxy:{source_label}", source_url),
        CalibrationTarget("mean_thread_depth", depth_lower, depth_upper, f"empirical_thread_depth_proxy:{source_label}", source_url),
        CalibrationTarget("cascade_root_fraction", cascade_lower, cascade_upper, f"empirical_cascade_proxy:{source_label}", source_url),
        CalibrationTarget("controversy_share", controversy_lower, controversy_upper, f"empirical_controversy_proxy:{source_label}", source_url),
    )


def build_platform_calibration_report(
    *,
    sources: tuple[PlatformSource, ...],
    forum_records: tuple[PlatformForumRecord, ...],
    feedback_records: tuple[PlatformFeedbackRecord, ...],
    report_id: str = "society_core_platform_calibration_v15",
) -> PlatformCalibrationReport:
    profiles: dict[str, PlatformMetricProfile] = {}
    for source in sources:
        source_forum = tuple(record for record in forum_records if record.source_id == source.source_id)
        source_feedback = tuple(record for record in feedback_records if record.source_id == source.source_id)
        profiles[source.source_id] = build_platform_metric_profile(
            source=source,
            forum_records=source_forum,
            feedback_records=source_feedback,
        )
    caveats: list[str] = []
    if not sources:
        caveats.append("no_platform_sources")
    if not forum_records:
        caveats.append("no_forum_records")
    if not feedback_records:
        caveats.append("no_github_feedback_records")
    if not any(source.platform == "reddit" for source in sources):
        caveats.append("missing_reddit_source")
    if not any(source.platform == "hacker_news" for source in sources):
        caveats.append("missing_hacker_news_source")
    if not any(source.platform == "github" for source in sources):
        caveats.append("missing_github_source")
    targets = calibration_targets_from_profiles(tuple(profiles.values()))
    return PlatformCalibrationReport(
        report_id=report_id,
        sources=sources,
        forum_records=forum_records,
        feedback_records=feedback_records,
        profiles=profiles,
        calibration_targets=targets,
        caveats=tuple(caveats),
    )


def records_from_raw_payloads(
    payloads: dict[str, list[dict[str, Any]]],
    sources: tuple[PlatformSource, ...],
) -> tuple[tuple[PlatformForumRecord, ...], tuple[PlatformFeedbackRecord, ...]]:
    source_by_id = {source.source_id: source for source in sources}
    forum_records: list[PlatformForumRecord] = []
    feedback_records: list[PlatformFeedbackRecord] = []
    for source_id, rows in payloads.items():
        source = source_by_id[source_id]
        if source.platform == "hacker_news":
            forum_records.extend(normalize_hn_records(source, rows))
        elif source.platform == "reddit":
            forum_records.extend(normalize_reddit_records(source, rows))
        elif source.platform == "github":
            feedback_records.extend(normalize_github_issue_records(source, rows))
    return tuple(forum_records), tuple(feedback_records)
