# Plan

## Use tesseract with Armenian

1. Try tesseract on an extract of a newspaper with optimal pre-processing
2. Find similar font to the one used in the newspaper
3. Create ground truth files using `text2image` and train with tesstrain : `make training MODEL_NAME=Mk-Parz-U START_MODEL=hye TESSDATA=../tesseract/tessdata MAX_ITERATIONS=100`

## Explore Armenian character recognition

- Review of character level datasets like MNIST, their size, error rates with modern ConvNets.
- Build a dataset of the same size as MNIST, with 6k images per class. Use augmentation to get to 7k (slight rotation, stretch, different fonts, noise)
- Run a LeNet with dropout on it, as well as other weaker nets. Plot results, see error improve
- Run the model on data outside of the dataset, like a different font or extract from the newspaper (line with manual segmentation)

## Character recognition datasets

| Dataset         | Number of Examples | Number of Classes | Number of Examples per Class | Image Size |
| --------------- | ------------------ | ----------------- | ---------------------------- | ---------- |
| MNIST           | 70,000             | 10                | 7000                         | 28x28      |
| Kuzushiji-MNIST | 70,000             | 10                | 7000                         | 28x28      |
| Kuzushiji-49    | 270,912            | 49                | ~5528                        | 28x28      |
| Kuzushiji-Kanji | 140,426            | 3832              | 1-1766                       | 64x64      |