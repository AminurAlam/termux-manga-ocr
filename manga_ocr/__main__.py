import fire

from manga_ocr.run import run


def main():
    try:
        fire.Fire(run)
    except KeyboardInterrupt:
        exit()


if __name__ == "__main__":
    main()
