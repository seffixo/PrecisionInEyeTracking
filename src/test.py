# LIBARIES:
import argparse


def main(parameter):
    print("Hallo ", parameter)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("--parameter", help="Description of the parameter")

    arg = parser.parse_args()
    main(arg.parameter)
