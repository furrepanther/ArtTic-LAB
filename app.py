import argparse
import logging
import sys
import os
import signal
import warnings

# --- Warning Suppression ---
# General cleanup
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# Specific suppression for the CLIP/Transformer mismatch warning
warnings.filterwarnings("ignore", message=".*clip_text_model.*")

from helpers.cli_manager import (
    setup_logging,
    log_system_info,
    APP_LOGGER_NAME,
    setup_web_logging,
)

parser = argparse.ArgumentParser(description="ArtTic-LAB: Universal AI Art Station")
parser.add_argument(
    "--disable-filters", action="store_true", help="Disable custom log formatting"
)
parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
parser.add_argument("--port", type=int, default=7860, help="Port number")
parser.add_argument(
    "--share", action="store_true", help="Launch with public share link enabled"
)
args = parser.parse_args()

setup_logging(disable_filters=args.disable_filters)
logger = logging.getLogger(APP_LOGGER_NAME)


def signal_handler(sig, frame):
    print("\n")
    logger.info("Shutdown signal received. Exiting...")
    sys.exit(0)


def launch_web_ui():
    try:
        import uvicorn
        import transformers
        import diffusers

        # Set library verbosity to error to reduce log spam (like "Loaded model..."),
        # but this DOES NOT disable progress bars (tqdm).
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

        from web.server import app as fastapi_app
        from core.logic import toggle_share
    except ImportError as e:
        logger.error(f"Dependencies missing: {e}")
        logger.error("Please run ./install.sh")
        sys.exit(1)

    # Handle Ngrok Sharing
    if args.share:
        logger.info("Enabling public share link...")
        res = toggle_share()
        if res["status"] == "connected":
            logger.info(f"Public URL: {res['url']}")
        else:
            logger.error(f"Share failed: {res.get('message')}")

    # Display System Info
    if not args.disable_filters:
        setup_web_logging()
        if os.name == "nt":
            os.system("cls")
        else:
            os.system("clear")
        log_system_info()
    else:
        log_system_info()

    logger.info(f"Local URL: http://{args.host}:{args.port}")
    logger.info("Press Ctrl+C to stop.")

    # Configure Uvicorn
    config = uvicorn.Config(
        fastapi_app,
        host=args.host,
        port=args.port,
        log_level="critical",  # Suppress Uvicorn startup logs
        access_log=False,
    )
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    os.makedirs("./outputs", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./loras", exist_ok=True)

    launch_web_ui()
