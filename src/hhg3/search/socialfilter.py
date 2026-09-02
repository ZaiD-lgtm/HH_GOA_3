"""Map a URL to a social platform, so we can keep only real social-media posts."""

from __future__ import annotations

from urllib.parse import urlparse

PLATFORMS: dict[str, str] = {
    "instagram.com": "instagram",
    "x.com": "x",
    "twitter.com": "x",
    "facebook.com": "facebook",
    "fb.com": "facebook",
    "linkedin.com": "linkedin",
    "tiktok.com": "tiktok",
    "reddit.com": "reddit",
    "youtube.com": "youtube",
    "youtu.be": "youtube",
    "threads.net": "threads",
    "threads.com": "threads",
    "pinterest.com": "pinterest",
    "tumblr.com": "tumblr",
    "vk.com": "vk",
    "flickr.com": "flickr",
    "mastodon.social": "mastodon",
    "bsky.app": "bluesky",
    "weibo.com": "weibo",
}


def domain_of(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def platform_of(url: str) -> str | None:
    """Return the platform slug for `url`, or None when it is not social media."""
    host = domain_of(url)
    if not host:
        return None
    for suffix, platform in PLATFORMS.items():
        if host == suffix or host.endswith("." + suffix):
            return platform
    return None
