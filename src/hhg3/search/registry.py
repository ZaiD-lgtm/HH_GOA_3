"""Provider selection.

`auto` walks the preference order and takes the first provider whose credentials
are present. SerpAPI leads because Google Lens surfaces social pages that plain
web detection often misses; Cloud Vision follows because it is cheaper and needs
no public image host; Yandex is the keyless last resort.

Bing Visual Search used to sit in this list. Microsoft retired the whole Bing
Search API family on 2025-08-11 - no new signups, existing keys return 410 - so
the provider was removed rather than left as a trap.
"""

from __future__ import annotations

from hhg3.logging_utils import info, warn
from hhg3.search.base import ProviderUnavailable, SearchProvider
from hhg3.search.gcv_web import GoogleVisionWebProvider
from hhg3.search.mock import MockProvider
from hhg3.search.serpapi_lens import SerpApiLensProvider
from hhg3.search.yandex import YandexProvider

_BUILDERS = {
    "serpapi": SerpApiLensProvider,
    "gcv": GoogleVisionWebProvider,
    "yandex": YandexProvider,
    "mock": MockProvider,
}

_AUTO_ORDER = ["serpapi", "gcv", "yandex"]


def list_providers() -> dict[str, SearchProvider]:
    return {key: builder() for key, builder in _BUILDERS.items()}


def get_provider(name: str = "auto") -> SearchProvider:
    if name != "auto":
        if name not in _BUILDERS:
            raise ProviderUnavailable("unknown search provider: " + name)
        return _BUILDERS[name]()
    for key in _AUTO_ORDER:
        provider = _BUILDERS[key]()
        if provider.available():
            info("search provider: " + provider.name)
            return provider
        warn("search provider %r not configured" % key)
    raise ProviderUnavailable(
        "no genuine search provider configured - set SERPAPI_KEY or "
        "GOOGLE_VISION_API_KEY, or pass --provider yandex/mock explicitly"
    )
