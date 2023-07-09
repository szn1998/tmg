"""Console script for tmg."""

import fire


def help():
    print("tmg")
    print("=" * len("tmg"))
    print("Tibetan Music Generation Project")

def main():
    fire.Fire({
        "help": help
    })


if __name__ == "__main__":
    main() # pragma: no cover
