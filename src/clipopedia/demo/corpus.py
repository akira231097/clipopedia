"""A small, entirely fictional corpus used by the offline demo.

Every show, host, and guest below is invented for illustration. The texts are
short transcript-style snippets, just rich enough that hybrid search, reranking,
and selection produce sensible, differentiated results.
"""

from __future__ import annotations

from datetime import timedelta

from ..dateutils import today_utc
from ..models import Clip, MediaItem, Mention
from ..retrieval.gazetteer import Gazetteer

# (show, host, guest, topics, text, days_ago, duration_seconds)
_RAW: list[tuple[str, str, str, list[str], str, int, int]] = [
    (
        "The Long Game", "Mara Quinn", "Dr. Lena Ortiz",
        ["ai agents", "autonomy", "tooling"],
        "The leap with AI agents isn't raw intelligence, it's giving them reliable tools and "
        "letting them retry. Once an agent can call an API, read the result, and correct itself, "
        "you get genuinely autonomous workflows.",
        12, 280,
    ),
    (
        "The Long Game", "Mara Quinn", "Sam Whitfield",
        ["fundraising", "startups", "dilution"],
        "Founders obsess over valuation, but the term sheet detail that actually decides your "
        "future is the liquidation preference. I've watched great companies get gutted by a 2x "
        "participating preference nobody read closely.",
        40, 320,
    ),
    (
        "Builders & Backers", "Devin Cole", "Priya Raman",
        ["burnout", "founder mental health", "pacing"],
        "Burnout didn't arrive as exhaustion for me. It showed up as cynicism. I stopped caring "
        "about the thing I'd bled for. The fix wasn't a vacation, it was rebuilding a team I "
        "could actually delegate to.",
        6, 240,
    ),
    (
        "Builders & Backers", "Devin Cole", "Hiro Tanaka",
        ["climate tech", "hardware", "manufacturing"],
        "Climate hardware is brutal because atoms don't follow Moore's law. You can't iterate a "
        "factory the way you iterate software. Every design change is a six-month tooling cycle "
        "and a new supply chain conversation.",
        75, 360,
    ),
    (
        "Signal / Noise", "Avery Sloan", "Dr. Lena Ortiz",
        ["ai agents", "evaluation", "reliability"],
        "Everyone demos an agent that works once. The hard part is evaluation: how do you measure "
        "whether it succeeds on the long tail of messy real inputs? Without an eval harness you "
        "are just shipping vibes.",
        3, 200,
    ),
    (
        "Signal / Noise", "Avery Sloan", "Marcus Bell",
        ["remote work", "culture", "async"],
        "Remote work didn't kill culture, bad defaults did. If your default is a meeting, remote "
        "feels lonely and slow. If your default is a written doc anyone can comment on, remote "
        "feels faster than an office ever did.",
        28, 300,
    ),
    (
        "The Deep End", "Noa Frank", "Priya Raman",
        ["leadership", "delegation", "scaling"],
        "The skill that gets you to ten people actively hurts you at fifty. Doing the work "
        "yourself was your superpower; now it's the bottleneck. Leadership at scale is mostly "
        "learning to be comfortable not touching the work.",
        18, 260,
    ),
    (
        "The Deep End", "Noa Frank", "Sam Whitfield",
        ["pricing", "go to market", "revenue"],
        "Most startups underprice out of fear. Your first ten customers should feel like the "
        "product is almost too cheap. If nobody is pushing back on price, you've left your whole "
        "runway on the table.",
        52, 220,
    ),
    (
        "Frontier Notes", "Ravi Menon", "Dr. Lena Ortiz",
        ["ai safety", "alignment", "agents"],
        "The risk with autonomous agents isn't a dramatic rogue AI. It's a boring one that "
        "confidently does the wrong thing at scale because nobody put a human in the loop on the "
        "irreversible actions.",
        2, 190,
    ),
    (
        "Frontier Notes", "Ravi Menon", "Elena Vossberg",
        ["longevity", "health", "research"],
        "The longevity field is finally separating from snake oil because we now have biomarkers "
        "we can actually move. Forget the supplements; sleep, zone-two cardio, and muscle mass do "
        "more than any pill on the market.",
        9, 350,
    ),
    (
        "Builders & Backers", "Devin Cole", "Sam Whitfield",
        ["fundraising", "pitching", "narrative"],
        "Investors don't fund features, they fund a story about an inevitable future. Your pitch "
        "should make them feel late, like this is already happening and they need to get on "
        "before the door closes.",
        21, 230,
    ),
    (
        "The Long Game", "Mara Quinn", "Marcus Bell",
        ["productivity", "focus", "deep work"],
        "Deep work isn't about willpower, it's about geometry. I rearranged my calendar so that "
        "mornings are a single uninterrupted block. Protecting that one block changed my output "
        "more than any app ever did.",
        4, 210,
    ),
    (
        "Signal / Noise", "Avery Sloan", "Hiro Tanaka",
        ["supply chain", "hardware", "resilience"],
        "After the shortages, smart hardware teams stopped optimizing purely for cost. They now "
        "design for substitutability: every critical component has a second source, even if it's "
        "ten percent more expensive.",
        33, 270,
    ),
    (
        "The Deep End", "Noa Frank", "Elena Vossberg",
        ["habits", "behavior change", "psychology"],
        "Lasting behavior change is never about motivation. Motivation is weather. You design the "
        "environment so the good choice is the lazy choice, and then you let your future self be "
        "wonderfully lazy.",
        14, 245,
    ),
]


def build_demo_corpus() -> list[Clip]:
    today = today_utc()
    clips: list[Clip] = []
    for i, (show, host, guest, topics, text, days_ago, dur_s) in enumerate(_RAW):
        episode_id = f"ep-{i // 2:02d}"
        clips.append(
            Clip(
                chunk_id=f"clip-{i:03d}",
                episode_id=episode_id,
                show_title=show,
                episode_title=f"{guest} on {topics[0]}",
                text=text,
                guests=[guest],
                hosts=[host],
                speakers=[host, guest],
                topics=topics,
                published_date=today - timedelta(days=days_ago),
                start_ms=0,
                end_ms=dur_s * 1000,
                duration_ms=dur_s * 1000,
                audio_url=f"https://clips.example.com/{episode_id}/clip-{i:03d}.mp3",
                video_url=f"https://clips.example.com/{episode_id}/clip-{i:03d}.mp4",
            )
        )
    return clips


def build_demo_gazetteer(clips: list[Clip] | None = None) -> Gazetteer:
    clips = clips or build_demo_corpus()
    guests = sorted({g for c in clips for g in c.guests})
    hosts = sorted({h for c in clips for h in c.hosts})
    shows = sorted({c.show_title for c in clips})
    return Gazetteer(guests=guests, hosts=hosts, shows=shows)


def demo_mentions() -> list[Mention]:
    return [
        Mention(id="1001", text="@clipopedia best clip on AI agents and reliability?",
                author_handle="curious_dev", author_name="Curious Dev"),
        Mention(id="1002", text="@clipopedia anything recent on founder burnout?",
                author_handle="tired_founder", author_name="Tired Founder"),
        Mention(id="1003", text="hey! gm", author_handle="friendly", author_name="Friendly"),
    ]


def demo_media_example() -> MediaItem:
    return MediaItem(url="https://img.example.com/whiteboard.png")
