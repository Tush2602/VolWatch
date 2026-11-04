import sys

from pipeline.factory import build_pipeline


def main() -> None:
    result = build_pipeline().run(force_retrain="--force-retrain" in sys.argv)
    print(result)


if __name__ == "__main__":
    main()