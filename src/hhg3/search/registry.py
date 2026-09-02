"""Provider selection. `auto` prefers keyed APIs, then scraping, then mock."""

from __future__ import annotations

from hhg3.logging_utils import info, warn
from hhg3.search.base import ProviderUnavailable, SearchProvider
from hhg3.search.bing_visual import BingVisualSearchProvider
from hhg3.search.mock import MockProvider
from hhg3.search.serpapi_lens import SerpApiLensProvider
from hhg3.search.yandex import YandexProvider

_BUILDERS = {
    "serpapi": SerpApiLensProvider,
    "bing": BingVisualSearchProvider,
    "yandex": YandexProvider,
    "mock": MockProvider,
}

_AUTO_ORDER = ["serpapi", "bing", "yandex"]


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
        "BING_VISUAL_SEARCH_KEY, or pass --provider yandex/mock explicitly"
    )
