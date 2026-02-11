import argparse

from gsMap import __version__
from gsMap.config import cli_function_registry


def main():
    parser = create_parser()
    try:
        import argcomplete

        argcomplete.autocomplete(parser)
    except ImportError:
        pass
    args = parser.parse_args()
    if args.subcommand is None:
        parser.print_help()
        exit(1)
    args.func(args)


def create_parser():
    parser = argparse.ArgumentParser(
        description=" gsMap: genetically informed spatial mapping of cells for complex traits",
        formatter_class=argparse.RawTextHelpFormatter,
        prog="gsMap",
    )
    parser.add_argument(
        "--version", "-v", action="version", version=f"gsMap version {__version__}"
    )
    subparsers = parser.add_subparsers(
        dest="subcommand", help="Subcommands", title="Available subcommands"
    )
    for subcommand in cli_function_registry.values():
        subcommand_parser = subparsers.add_parser(
            subcommand.name,
            help=subcommand.description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        subcommand.add_args_function(subcommand_parser)
        subcommand_parser.set_defaults(func=subcommand.func)
    return parser


if __name__ == "__main__":
    main()
