"""
Command-line interface for the PyTorch Block Extractor.

This module provides the CLI interface and argument parsing for the block extraction tool.
"""

import argparse
import json
from pathlib import Path

from .extract_blocks import BlockExtractor


def main():
    """Main CLI function that uses the BlockExtractor API."""
    p = argparse.ArgumentParser(description="Extract reusable PyTorch blocks")
    p.add_argument("--block", type=str, help="Extract a specific block by name")
    p.add_argument("--blocks", nargs="+", help="Extract multiple blocks")
    p.add_argument("--retry-failed", action="store_true", help="Retry failed blocks from previous run")
    from .utils.path_resolver import get_config_file_path
    p.add_argument("--names-json", type=Path, default=get_config_file_path("nn_block_names.json"),
                   help="If --block/--blocks not provided, read names from this JSON (default: ab/rag/config/nn_block_names.json)")
    p.add_argument("--limit", type=int, default=None, help="Max number of names to process from the list")
    p.add_argument("--start-from", type=str, default=None,
                   help="Skip names until (and including) this one, then start")
    p.add_argument("--stop-on-fail", action="store_true", help="Stop batch as soon as one extraction fails")
    p.add_argument("--progress-every", type=int, default=10, help="Log progress every N blocks")
    p.add_argument("--index-mode", type=str, choices=("missing", "force", "skip"),
                   default="missing", help="Indexing policy: missing (default), force, or skip")
    p.add_argument("--no-validate", action="store_true", help="Disable automatic validation and movement of valid blocks to 'block' directory")
    p.add_argument("--cleanup-invalid", action="store_true", help="Remove invalid blocks after validation")
    p.add_argument("--project-dir", type=Path, default=None, help="Project directory where blocks will be created (default: current directory)")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    p.add_argument("--force-reclone", action="store_true", help="Force re-cloning of all repositories")
    args = p.parse_args()

    # Initialize extractor
    extractor = BlockExtractor(index_mode=args.index_mode, project_dir=args.project_dir, verbose=args.verbose)

    # Force re-clone if requested
    if args.force_reclone:
        print("🔄 Force re-cloning all repositories...")
        extractor.force_reclone_repositories()

    # Warm caches + build index ONCE (policy-controlled)
    ok = extractor.warm_index_once()
    if not ok:
        return

    # Single block extraction
    if args.block:
        res = extractor.extract_single_block(
            args.block, 
            validate=not args.no_validate, 
            cleanup_invalid=args.cleanup_invalid
        )
        print(json.dumps(res, indent=2))
        return

    # Multiple blocks extraction
    if args.blocks:
        results = extractor.extract_multiple_blocks(
            args.blocks,
            validate=not args.no_validate,
            cleanup_invalid=args.cleanup_invalid
        )
        print(json.dumps(results, indent=2))
        return

    # Retry failed blocks
    if args.retry_failed:
        retried = extractor.retry_failed_blocks(
            validate=not args.no_validate,
            cleanup_invalid=args.cleanup_invalid
        )
        print(json.dumps(retried, indent=2))
        return

    # Default mode: extract from JSON file
    result = extractor.extract_blocks_from_file(
        json_path=args.names_json,
        limit=args.limit,
        start_from=args.start_from
    )
    
    # Validation is now done incrementally during extraction
    
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
