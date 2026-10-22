import os
import logging.config

def setup_logging():
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": (
                    "%(asctime)s %(levelname)-8s "
                    "[%(name)s] %(message)s"
                )
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            }
        },
        "root": {
            "level": os.getenv("LOG_LEVEL", "INFO").upper(),
            "handlers": ["console"],
        }
    })
