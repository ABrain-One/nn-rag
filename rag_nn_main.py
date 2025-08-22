#!/usr/bin/env python3
"""
Smart RAG-NN Main Entry Point
Implements user's suggested strategy without hardcoding
"""

import asyncio
import argparse
import logging
import os
import sys
from dotenv import load_dotenv

from rag_nn.core.pipeline import run_smart_pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def _configure_console_encoding():
    """Ensure stdout/stderr can print unicode on Windows consoles."""
    try:
        # Prefer UTF-8; fallback to replacing unsupported characters
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        # Older Python or redirected streams may not support reconfigure
        pass

def setup_environment():
    """Setup environment and validate requirements"""
    # Load environment variables
    load_dotenv()
    
    # Get GitHub token
    github_token = os.getenv('GITHUB_TOKEN')
    if not github_token:
        print("❌ ERROR: GITHUB_TOKEN not found in environment variables")
        print("Please set your GitHub token in the .env file:")
        print("GITHUB_TOKEN=your_token_here")
        sys.exit(1)
    
    return github_token

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Smart RAG-NN Pipeline - Intelligent PyTorch Code Block Discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python smart_rag_nn_main.py --name RepConv DWConv Conv
  python smart_rag_nn_main.py --name Attention --output results --blocks final
  python smart_rag_nn_main.py --blocks-file my_blocks.txt

Strategy:
  1. Search GitHub API with type=code for direct code discovery
  2. Analyze top 3 candidates based on criteria (dependencies, lines, PyTorch)
  3. Cache successful repositories for future use
  4. Smart dependency resolution: same file → imports → direct fetch
  5. Health check with executability testing (no stubs)
        """
    )
    
    # Block selection (mutually exclusive group)
    block_group = parser.add_mutually_exclusive_group(required=False)
    block_group.add_argument(
        '--name', 
        nargs='+',
        help='One or more block names to process (e.g., RepConv DWConv Conv)'
    )
    block_group.add_argument(
        '--blocks-file',
        type=str,
        default='default_blocks.json',
        help='File containing block names (one per line, default: default_blocks.json)'
    )
    
    # Directory configuration
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Output directory for temporary files (default: output)'
    )
    parser.add_argument(
        '--blocks',
        type=str,
        default='blocks',
        help='Directory for healthy/final blocks (default: blocks)'
    )
    
    # Processing options
    parser.add_argument(
        '--no-health-check',
        action='store_true',
        help='Skip health checking and file organization'
    )
    parser.add_argument(
        '--max-candidates',
        type=int,
        default=3,
        help='Maximum candidates to analyze per block (default: 3)'
    )
    
    # Output options
    parser.add_argument(
        '--report-file',
        type=str,
        help='Save detailed report to file'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress non-error output'
    )
    
    return parser.parse_args()

def load_blocks_from_file(file_path: str) -> list:
    """Load block names from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            blocks = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        return blocks
    except FileNotFoundError:
        print(f"❌ ERROR: Blocks file not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: Failed to read blocks file: {e}")
        sys.exit(1)

async def main():
    """Main entry point"""
    try:
        _configure_console_encoding()
        # Parse arguments
        args = parse_arguments()
        
        # Setup environment
        github_token = setup_environment()
        
        # Determine block names
        if args.name:
            block_names = args.name
        elif args.blocks_file:
            block_names = load_blocks_from_file(args.blocks_file)
        else:
            # Use default blocks file
            block_names = load_blocks_from_file('default_blocks.json')
        
        if not block_names:
            print("❌ ERROR: No block names provided")
            sys.exit(1)
        
        # Configure logging level
        if args.quiet:
            logging.getLogger().setLevel(logging.WARNING)
        
        # Display startup information
        if not args.quiet:
            print("🚀 Smart RAG-NN Pipeline")
            print("=" * 50)
            print(f"📋 Processing {len(block_names)} blocks: {', '.join(block_names)}")
            print(f"🔑 Using GitHub token: {github_token[:10]}...")
            print(f"📁 Output directory: {args.output}")
            print(f"📁 Blocks directory: {args.blocks}")
            print(f"🧬 Strategy: Search API + Smart Dependencies + Health Check")
            print()
        
        # Run the smart pipeline
        results = await run_smart_pipeline(
            block_names=block_names,
            github_token=github_token,
            output_dir=args.output,
            blocks_dir=args.blocks
        )
        
        # Save detailed report if requested
        if args.report_file:
            from rag_nn.core.pipeline import SmartRAGPipeline
            pipeline = SmartRAGPipeline(github_token, args.output, args.blocks)
            report = pipeline.generate_report(results)
            
            with open(args.report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"📄 Detailed report saved to: {args.report_file}")
        
        # Return appropriate exit code
        stats = results["statistics"]
        if stats["healthy_blocks"] > 0:
            print(f"\n✅ SUCCESS: {stats['healthy_blocks']} healthy blocks ready in {args.blocks}/")
            return 0
        else:
            print(f"\n❌ WARNING: No healthy blocks generated. Check logs for issues.")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️ Pipeline interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {e}")
        logging.exception("Fatal error in main")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
