"""Generate two small synthetic 'research papers' for demos.

Run:  python scripts/make_sample_pdfs.py
Writes into sample_pdfs/. The papers are synthetic and say so on page 1,
so a demo never depends on copyrighted material.
"""
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "sample_pdfs"

SAMPLE_PAPERS = [
    (
        "sample_paper_1_transformers.pdf",
        "Tiny Transformers for Text Classification: A Synthetic Sample Paper",
        [
            "Tiny Transformers for Text Classification: A Synthetic Sample Paper. "
            "Abstract: This synthetic paper was created to demonstrate ResearchPilot "
            "AI with realistic content. We study whether small transformer models "
            "can classify news topics accurately. Research question: How well can a "
            "compact transformer encoder classify news topics compared to simpler "
            "baseline methods?",
            "Methodology: We train a six layer transformer encoder with self "
            "attention on the AG News topic classification dataset, using 120000 "
            "training examples and 7600 test examples. We compare against a "
            "logistic regression baseline and a bag of words model. All experiments "
            "run on a single consumer GPU over three random seeds.",
            "Findings: The compact transformer reaches 87.4 percent accuracy on the "
            "AG News test set, compared to 82.1 percent for logistic regression and "
            "79.6 percent for the bag of words baseline. Training completes in under "
            "two hours. Limitations: The study only covers English news text, the "
            "model is not evaluated on longer documents, and the dataset is limited "
            "to four topic classes.",
        ],
    ),
    (
        "sample_paper_2_leaf_disease.pdf",
        "CNNs for Plant Leaf Disease Detection: A Synthetic Sample Paper",
        [
            "Convolutional Networks for Plant Leaf Disease Detection: A Synthetic "
            "Sample Paper. Abstract: This synthetic paper was created to demonstrate "
            "ResearchPilot AI with realistic content. We ask whether transfer "
            "learning with convolutional networks can detect plant diseases from "
            "leaf photographs collected by farmers.",
            "Methodology: We fine tune a ResNet-18 convolutional network pretrained "
            "on ImageNet using 5000 labelled leaf photographs from the PlantVillage "
            "dataset, covering ten crop species and twelve disease categories. "
            "Images are resized to 224 by 224 pixels and augmented with random "
            "horizontal flips and slight colour jitter during training.",
            "Findings: The fine tuned model achieves a macro F1 score of 0.91 on a "
            "held out test set, and detects the most frequent diseases with over 95 "
            "percent precision. Inference runs fast enough for mobile use. "
            "Limitations: All photographs were taken in laboratory conditions with "
            "uniform backgrounds, and the model has not been tested on field images "
            "where lighting and background vary.",
        ],
    ),
]


def _wrap(text: str, width: int = 90) -> str:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return "\n".join(lines)


def build(filename: str, title: str, pages: list[str]) -> None:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 96), _wrap(text), fontsize=11)
    doc.set_metadata({"title": title})
    OUT_DIR.mkdir(exist_ok=True)
    doc.save(OUT_DIR / filename)
    doc.close()


if __name__ == "__main__":
    for filename, title, pages in SAMPLE_PAPERS:
        build(filename, title, pages)
        print(f"wrote {OUT_DIR / filename}")
