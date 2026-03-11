"""Entrypoint to start the Database Course Agent server."""

import logging
import os
import sys

import uvicorn
import yaml


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings_path = os.environ.get("MODULAR_RAG_SETTINGS_PATH", "config/settings.yaml")
    try:
        with open(settings_path, encoding="utf-8") as f:
            settings = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Settings file not found: {settings_path}")
        sys.exit(1)

    server_cfg = settings.get("server", {})
    host = server_cfg.get("host", "0.0.0.0")
    port = server_cfg.get("port", 8000)

    logging.getLogger().info("Starting Database Course Agent on %s:%d", host, port)
    uvicorn.run(
        "src.server.app:create_app",
        host=host,
        port=port,
        factory=True,
        reload=False,
    )


if __name__ == "__main__":
    main()
