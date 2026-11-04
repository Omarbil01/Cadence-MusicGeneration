"""Console logs for local development and Docker log collection."""

import logging


def configure_logging(level="INFO"):
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
