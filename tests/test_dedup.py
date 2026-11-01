from app.ingest.dedup import near_duplicates
from app.ingest.sites import SITES

GUIDE = "Install the build tools, clone the repository, run make and make install, then start the server. " * 5


def test_near_duplicate_siblings_dropped_but_not_cousins():
    pages = {
        "https://x/build/rocky-8/": GUIDE + "Use dnf on Rocky 8.",
        "https://x/build/rocky-9/": GUIDE + "Use dnf on Rocky 9.",
        "https://x/other/rocky-9/": GUIDE + "Use dnf on Rocky 9.",   # same text, different parent: kept
        "https://x/build/macos/": "Use Homebrew: brew install redis. " * 5,
    }
    assert near_duplicates(pages) == {"https://x/build/rocky-9/": "https://x/build/rocky-8/"}


def test_redis_noise_paths_excluded():
    site = SITES["redis"]
    base = "https://redis.io/docs/latest/"
    for path in ["operate/oss_and_stack/stack-with-enterprise/release-notes/x/", "commands/redis-8-4-commands/",
                 "develop/use-cases/rate-limiter/nodejs/", "develop/clients/jedis/connect/",
                 "develop/tools/insight/release-notes/v.2.16.0/", "operate/oss_and_stack/install/archive/a/"]:
        assert not site.wants(base + path), path
    for path in ["commands/zadd/", "develop/use-cases/rate-limiter/redis-py/", "develop/clients/redis-py/connect/",
                 "develop/clients/pools-and-muxing/", "operate/oss_and_stack/management/persistence/"]:
        assert site.wants(base + path), path
