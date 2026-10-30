#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
HWPX = ROOT / "graduation.hwpx"
TARGET = ROOT / "graduation.txt"
MARKDOWN = Path(sys.argv[1])
HP = "{http://www.hancom.co.kr/hwpml/2011/paragraph}"


TABLE_2 = """| 구분 | Acc. (%) | MAE | C0 (%) | C1 (%) | C2 (%) | C3 (%) | C4 (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 원 논문 | 83.80 | 0.2100 | 96.00 | 26.00 | 62.00 | 45.00 | 60.00 |
| 재현 모델 | 75.78 | 0.3196 | 90.94 | 13.52 | 43.48 | 42.05 | 21.13 |"""

TABLE_3 = """| 집계 방식 | Acc. (%) | MAE | C0 (%) | C1 (%) | C2 (%) | C3 (%) | C4 (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sum | 70.23 | 0.4169 | 91.40 | 11.48 | 10.59 | 23.86 | 4.23 |
| Mean | 72.40 | 0.3469 | 85.75 | 25.41 | 41.21 | 39.77 | 21.13 |"""

CAPTION_BY_HEADER = {
    "| 가중치 설정 | Acc. (%) | MAE | C0 (%) | C1 (%) | C2 (%) | C3 (%) | C4 (%) |": "표 4. 대조 손실 가중치 탐색 결과",
    "| Temperature | Acc. (%) | MAE | C0 (%) | C1 (%) | C2 (%) | C3 (%) | C4 (%) |": "표 5. Temperature에 따른 Projection 연결 실행의 검증 성능",
    "| Dropout | α/β | Acc. (%) | MAE | C0 (%) | C1 (%) | C2 (%) | C3 (%) | C4 (%) |": "표 6. Dropout 적용 및 손실 가중치에 따른 검증 성능",
    "| 입력 설정 | Acc. (%) | MAE | C0 (%) | C1 (%) | C2 (%) | C3 (%) | C4 (%) |": "표 7. ImageNet 사전학습과 입력 정규화에 따른 검증 성능",
    "| 회귀 표본·입력 | Acc. (%) | MAE | C0 (%) | C1 (%) | C2 (%) | C3 (%) | C4 (%) |": "표 8. 회귀 표본 구성과 입력 표현에 따른 검증 성능",
    "| 클래스 가중치 | Acc. (%) | MAE | C0 (%) | C1 (%) | C2 (%) | C3 (%) | C4 (%) |": "표 9. 클래스 가중치 적용에 따른 Projection 연결 실행의 검증 성능",
    "| 학습률·거리 | Acc. (%) | MAE | C0 (%) | C1 (%) | C2 (%) | C3 (%) | C4 (%) |": "표 10. 학습률과 순서 거리 조합별 보존 검증 결과",
    "| 구분 | 0–1 | 1–2 | 2–3 | 3–4 |": "표 11. 고정 라벨 거리와 Two-phase Learning의 인접 경계 간격 비교",
    "| α/β | Acc. (%) | Macro | QWK | C0 (%) | C1 (%) | C2 (%) | C3 (%) | C4 (%) |": "표 12. PCOL·SCOL 손실 가중치에 따른 검증 성능",
    "| Spatial | Acc. (%) | Macro | MAE | C0 (%) | C1 (%) | C2 (%) | C3 (%) | C4 (%) |": "표 13. Spatial Attention 적용 여부에 따른 검증 성능",
}


def clean_lines(lines: list[str]) -> str:
    output: list[str] = []
    blank = False
    for line in lines:
        value = line.strip().lstrip("\u3000").strip()
        if not value:
            if output and not blank:
                output.append("")
            blank = True
            continue
        output.append(value)
        blank = False
    return "\n".join(output).strip()


def restore_captions(text: str) -> str:
    for header, caption in CAPTION_BY_HEADER.items():
        text = text.replace(header, caption + "\n\n" + header, 1)
    return text


def own_paragraph_text(paragraph: ET.Element) -> str:
    values: list[str] = []

    def visit(element: ET.Element) -> None:
        for child in element:
            if child.tag == HP + "p":
                continue
            if child.tag == HP + "t" and child.text:
                values.append(child.text)
            visit(child)

    visit(paragraph)
    return re.sub(r"[ \t]+", " ", "".join(values)).strip()


def extract_middle_section() -> str:
    with ZipFile(HWPX) as archive:
        root = ET.fromstring(archive.read("Contents/section0.xml"))
    container = root.findall("./" + HP + "p")[2]
    paragraphs = [own_paragraph_text(p) for p in container.iter(HP + "p")]
    start = paragraphs.index("2.2 기준 모델 구성")
    end = paragraphs.index("표 3. 손실 집계 방식에 따른 초기 검증 성능집계 방식Acc. (%)MAEC0 (%)C1 (%)C2 (%)C3 (%)C4 (%)Sum70.230.416991.4011.4810.5923.864.23Mean72.400.346985.7525.4141.2139.7721.13")
    output: list[str] = []
    for paragraph in paragraphs[start : end + 1]:
        if paragraph.startswith("표 2. 원 논문과 재현 Baseline"):
            output.extend(["표 2. 원 논문과 재현 Baseline의 최종 성능 및 클래스별 정확도 비교", "", TABLE_2])
        elif paragraph.startswith("표 3. 손실 집계 방식"):
            output.extend(["표 3. 손실 집계 방식에 따른 초기 검증 성능", "", TABLE_3])
        elif paragraph:
            output.append(paragraph)
        output.append("")
    return "\n".join(output).strip()


def main() -> None:
    lines = MARKDOWN.read_text(encoding="utf-8").splitlines()

    section_21_start = next(i for i, line in enumerate(lines) if line.strip() == "2.1 데이터셋 구성")
    section_21_end = next(
        i for i in range(section_21_start + 1, len(lines))
        if lines[i].strip().startswith("앞 단계의 Mean Reduction")
    )
    section_21 = clean_lines(lines[section_21_start:section_21_end])

    section_233_start = next(i for i, line in enumerate(lines) if line.strip() == "2.3.3 손실 가중치")
    section_233 = restore_captions(clean_lines(lines[section_233_start:]))

    chapter = "\n\n".join([section_21, extract_middle_section(), section_233]).strip() + "\n"
    existing = TARGET.read_text(encoding="utf-8")
    match = re.search(r"(?m)^2\.1 데이터셋 구성\s*$", existing)
    if match:
        existing = existing[: match.start()].rstrip() + "\n"
    TARGET.write_text(existing.rstrip() + "\n\n" + chapter, encoding="utf-8")


if __name__ == "__main__":
    main()
