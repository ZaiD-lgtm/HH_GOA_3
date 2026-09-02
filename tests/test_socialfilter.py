import pytest

from hhg3.search.socialfilter import domain_of, platform_of


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.instagram.com/p/Cxyz/", "instagram"),
        ("https://x.com/user/status/1", "x"),
        ("https://twitter.com/user/status/1", "x"),
        ("https://in.linkedin.com/in/someone", "linkedin"),
        ("https://old.reddit.com/r/pics/comments/a", "reddit"),
        ("https://example.com/blog/post", None),
        ("https://notinstagram.com/p/1", None),
        ("not a url", None),
    ],
)
def test_platform_detection(url, expected):
    assert platform_of(url) == expected


def test_domain_strips_www():
    assert domain_of("https://www.instagram.com/p/1") == "instagram.com"
