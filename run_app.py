#!/usr/bin/env python3
"""
Main script to run the Healthcare NMT Application
"""

import argparse
import sys
import os

# Ensure UTF-8 encoding for Windows console
if sys.platform.startswith('win'):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

# Add the package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from healthcare_nmt import NMTApp, setup_logging


def main():
    """Main function to run the application"""
    parser = argparse.ArgumentParser(
        description="Healthcare Neural Machine Translation System"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the application on"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a shareable link"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging("INFO")

    try:
        # Create and launch the application
        app = NMTApp(args.config)

        # Override config with command line arguments
        if args.port != 7860:
            app.config['ui']['port'] = args.port
        if args.share:
            app.config['ui']['share'] = args.share
        if args.debug:
            app.config['ui']['debug'] = args.debug

        logger.info(f"Starting Healthcare NMT Application")
        logger.info(f"Config file: {args.config}")
        logger.info(f"Port: {app.config['ui']['port']}")
        logger.info(f"Share: {app.config['ui']['share']}")

        # Launch the application
        app.launch(
            server_port=app.config['ui']['port'],
            share=app.config['ui']['share'],
            debug=app.config['ui']['debug']
        )

    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()