#!/usr/bin/env python3
from __future__ import annotations

import html
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "graduation_source_backup.txt"
OUTPUT = ROOT / "graduation.docx"
EDITED_TEXT = ROOT / "graduation_edited.txt"


SECTION_11 = [
    "1.1 당뇨망막병증의 개요",
    "당뇨망막병증은 장기간 지속되는 고혈당이 망막의 미세혈관과 혈액-망막 장벽을 손상시키면서 발생하는 대표적인 당뇨병성 미세혈관 합병증이다. 혈관 투과성이 증가하면 미세혈관류, 망막출혈, 삼출물 및 황반부종이 나타나고, 모세혈관 폐쇄가 확대되어 망막 허혈이 심화되면 이를 보상하기 위한 비정상적인 신생혈관이 형성되는데, 이러한 혈관은 구조적으로 취약하여 유리체출혈이나 견인망막박리를 일으키고 결국 회복하기 어려운 시력 저하와 실명으로 이어질 수 있다. 더욱이 초기에는 병변이 존재하더라도 중심 시력이 유지되어 환자가 뚜렷한 증상을 느끼지 못하는 경우가 많으므로, 자각 증상이 나타난 뒤 검사를 시행하는 방식만으로는 적절한 치료 시기를 놓칠 위험이 크다. 따라서 동공을 통해 망막과 망막혈관의 상태를 직접 관찰하는 안저검사는 미세혈관류, 출혈, 삼출물 및 신생혈관과 같은 주요 병변을 비침습적으로 확인하고 질환의 유무와 중증도를 평가할 수 있다는 점에서 매우 효과적이다. 특히 안저 촬영으로 획득한 영상은 검사 결과를 객관적으로 기록하여 시간에 따른 병변 변화를 비교하고 전문의 간 판독과 추적 관찰을 지원할 수 있으므로, 무증상 단계의 조기 발견부터 치료 개입 시점의 결정 및 진행 억제에 이르기까지 당뇨망막병증으로 인한 시력 손실을 예방하는 데 필수적인 임상적 근거를 제공한다.",
]


INSERT_BEFORE = {
    "EyePACS 학습 데이터는 총 35,126장의": "1.4.1 EyePACS 데이터셋과 중증도 등급",
    "이 등급 체계는 단순히 서로 다른 질환을": "1.4.2 인접 등급 간 판별의 어려움",
    "EyePACS의 35,126장 가운데 Grade 0은": "1.4.3 클래스 불균형과 소수 클래스 학습의 한계",
    "이런 문제를 해결하기 위해서 통상적으로": "1.4.4 데이터 불균형 완화 방법의 한계",
    "초기 딥러닝 기반 당뇨망막병증 연구는": "1.5.1 당뇨망막병증 자동 분류와 순서형 예측",
    "대조학습은 의미가 유사한 표본의 특징 벡터는": "1.5.2 대조학습 기반 표현 학습",
    "순서형 의료 영상에서는 같은 클래스인지 여부만으로": "1.5.3 순서형 대조학습과 질병 중증도 평가",
    "안저 영상에 포함된 미세한 병리적 단서를": "1.5.4 미세 병변 특징 강화를 위한 Attention과 Pooling",
    "ResNet은 입력 특징을 출력에 직접 더하는": "1.5.5 Backbone 네트워크: ResNet50과 EfficientNetV2",
}


def normalized_paragraphs(text: str) -> list[str]:
    raw = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    paragraphs = [line for line in raw if line]
    start = next(i for i, p in enumerate(paragraphs) if p.startswith("1.1 "))
    end = next(i for i, p in enumerate(paragraphs) if p.startswith("1.2 "))
    paragraphs[start:end] = SECTION_11

    edited: list[str] = []
    for paragraph in paragraphs:
        for prefix, heading in INSERT_BEFORE.items():
            if paragraph.startswith(prefix):
                edited.append(heading)
                break
        edited.append(paragraph)
    return edited


def heading_level(text: str) -> int | None:
    match = re.match(r"^(\d+(?:\.\d+)+)\s", text)
    if not match:
        return None
    return 2 if match.group(1).count(".") >= 2 else 1


def paragraph_xml(text: str) -> str:
    level = heading_level(text)
    escaped = html.escape(text)
    if level == 1:
        ppr = '<w:pPr><w:pStyle w:val="Heading1"/><w:keepNext/></w:pPr>'
    elif level == 2:
        ppr = '<w:pPr><w:pStyle w:val="Heading2"/><w:keepNext/></w:pPr>'
    else:
        ppr = '<w:pPr><w:jc w:val="both"/><w:ind w:firstLineChars="200"/><w:spacing w:line="408" w:lineRule="auto" w:after="160"/></w:pPr>'
    return f'<w:p>{ppr}<w:r><w:t xml:space="preserve">{escaped}</w:t></w:r></w:p>'


def build_document(paragraphs: list[str]) -> bytes:
    body = "".join(paragraph_xml(p) for p in paragraphs)
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <w:body>{body}<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/></w:sectPr></w:body>
</w:document>'''.encode("utf-8")


CONTENT_TYPES = b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>'''

ROOT_RELS = b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''

DOCUMENT_RELS = b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>'''

STYLES = b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Batang" w:hAnsi="Batang" w:eastAsia="\xeb\xb0\x94\xed\x83\x95"/><w:sz w:val="22"/><w:szCs w:val="22"/><w:lang w:val="ko-KR" w:eastAsia="ko-KR"/></w:rPr></w:rPrDefault></w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/><w:pPr><w:jc w:val="both"/><w:spacing w:line="408" w:lineRule="auto" w:after="160"/></w:pPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="360" w:after="240"/><w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:b/><w:sz w:val="30"/><w:szCs w:val="30"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="280" w:after="160"/><w:outlineLvl w:val="1"/></w:pPr><w:rPr><w:b/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr></w:style>
</w:styles>'''

CORE = b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>graduation</dc:title><dc:creator>Codex</dc:creator><cp:lastModifiedBy>Codex</cp:lastModifiedBy></cp:coreProperties>'''

APP = b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Microsoft Office Word</Application></Properties>'''


def main() -> None:
    paragraphs = normalized_paragraphs(SOURCE.read_text(encoding="utf-8-sig"))
    EDITED_TEXT.write_text("\n\n".join(paragraphs) + "\n", encoding="utf-8")
    with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", CONTENT_TYPES)
        archive.writestr("_rels/.rels", ROOT_RELS)
        archive.writestr("word/document.xml", build_document(paragraphs))
        archive.writestr("word/styles.xml", STYLES)
        archive.writestr("word/_rels/document.xml.rels", DOCUMENT_RELS)
        archive.writestr("docProps/core.xml", CORE)
        archive.writestr("docProps/app.xml", APP)


if __name__ == "__main__":
    main()
