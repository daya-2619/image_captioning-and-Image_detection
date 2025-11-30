import argparse
from vision_module import analyze_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="output.jpg")
    args = parser.parse_args()

    caption, det, annotated = analyze_image(args.image)

    print("\nCaption:", caption)
    print("\nDetections:")
    for d in det:
        print(f"- {d['label']} ({d['score']:.2f})")

    annotated.save(args.out)
    print("\nSaved annotated image to:", args.out)

if __name__ == "__main__":
    main()
