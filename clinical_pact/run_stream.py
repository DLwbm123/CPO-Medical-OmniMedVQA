"""Fail closed until the actual Qwen/TRL distributed trainer is implemented."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['train', 'smoke'])
    parser.parse_args()
    raise SystemExit('Unavailable: PACT Qwen/TRL integration and distributed equivalence validation are not implemented; no training was started.')


if __name__ == '__main__':
    main()
