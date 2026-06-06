"""Small subprocess entrypoint for FastF1 live timing recording.

FastF1's command-line wrapper does not expose the `no_auth` mode that exists
on SignalRClient. The dashboard uses this runner so the free collector can
attempt a public/no-login subscription and write whatever the stream permits.
"""

from __future__ import annotations

import argparse
import logging
import sys

import fastf1.livetiming.client as fastf1_client
from fastf1.livetiming.client import SignalRClient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--auth", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO)
    logger = logging.getLogger("F1LiveCollector")
    logger.info("Starting FastF1 runner no_auth=%s", not args.auth)
    no_auth = not args.auth
    if no_auth:
        # FastF1 3.8 exposes `no_auth=True`, but the bundled signalrcore
        # version rejects `access_token_factory=None`. An empty callable keeps
        # the public/no-login behavior while satisfying signalrcore's type
        # check.
        fastf1_client.get_auth_token = lambda: ""

    client = SignalRClient(
        args.file,
        filemode="a" if args.append else "w",
        timeout=args.timeout,
        logger=logger,
        no_auth=False,
    )
    client.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
