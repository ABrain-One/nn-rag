import sys
import argparse
from .retriever import Retriever


def main():
    parser = argparse.ArgumentParser(
        description="Harvest ≤800-line PyTorch blocks from GitHub"
    )
    parser.add_argument(
        "--dump", default="blocks", help="Directory to dump all blocks"
    )
    parser.add_argument(
        "--name", help="Fetch a single block by name and print to stdout"
    )
    args = parser.parse_args()

    retr = Retriever()
    if args.name:
        code = retr.get_block(args.name)
        if code:
            print(code)
        else:
            sys.exit(f"❌ failed to fetch block '{args.name}'")
    else:
        retr.dump_all_blocks(args.dump)


if __name__ == "__main__":
    main()