# Stage 3 Extraction Test Report (Classified Charts)

| Property | Value |
|----------|-------|
| Generated | 2026-01-29 02:35:42 |
| Total Charts Tested | 60 |
| Successful Processing | 60 (100.0%) |
| Failed | 0 |
| Overall Classification Accuracy | **100.0%** |
| Average Processing Time | 11297.3 ms |
| Total Processing Time | 677.8 s |

## Classification Accuracy by Chart Type

| Chart Type | Correct | Total | Accuracy |
|------------|---------|-------|----------|
| area | 10 | 10 | 100.0% |
| bar | 10 | 10 | 100.0% |
| histogram | 10 | 10 | 100.0% |
| line | 10 | 10 | 100.0% |
| pie | 10 | 10 | 100.0% |
| scatter | 10 | 10 | 100.0% |

## Confidence Scores

| Metric | Average |
|--------|---------|
| Overall Confidence | 92.6% |
| Classification Confidence | 93.3% |
| OCR Confidence | 91.5% |

## Confusion Matrix

| Ground Truth | area | bar | histogram | line | pie | scatter |
|----|----|----|----|----|----|----|
| **area** | 10 | 0 | 0 | 0 | 0 | 0 |
| **bar** | 0 | 10 | 0 | 0 | 0 | 0 |
| **histogram** | 0 | 0 | 10 | 0 | 0 | 0 |
| **line** | 0 | 0 | 0 | 10 | 0 | 0 |
| **pie** | 0 | 0 | 0 | 0 | 10 | 0 |
| **scatter** | 0 | 0 | 0 | 0 | 0 | 10 |

## Detailed Results

<details>
<summary>Click to expand individual results</summary>

| # | Image | Ground Truth | Predicted | Correct | Confidence | Time (ms) |
|---|-------|--------------|-----------|---------|------------|-----------|
| 1 | arxiv_2104_14962v2_page_7_img_3.png... | bar | bar | Yes | 100% | 7692 |
| 2 | arxiv_2410_01499v2_page_55_img_7.png... | bar | bar | Yes | 76% | 3126 |
| 3 | arxiv_2601_13382v1_page_6_img_1.png... | histogram | histogram | Yes | 99% | 13811 |
| 4 | arxiv_2308_08361v1_page_19_img_8.png... | scatter | scatter | Yes | 80% | 33267 |
| 5 | arxiv_2601_10343v2_page_3_img_21.png... | pie | pie | Yes | 98% | 1150 |
| 6 | arxiv_2411_01001v1_page_31_img_1.png... | scatter | scatter | Yes | 78% | 32888 |
| 7 | arxiv_2512_22293v1_p12_img00.png... | area | area | Yes | 100% | 2000 |
| 8 | arxiv_2601_08668v1_page_23_img_11.png... | area | area | Yes | 100% | 26317 |
| 9 | arxiv_2510_11835v1_page_4_img_1.png... | bar | bar | Yes | 100% | 15061 |
| 10 | arxiv_1811_12817v3_page_14_img_8.png... | line | line | Yes | 99% | 1158 |
| 11 | arxiv_2511_13221v1_page_78_img_1.png... | histogram | histogram | Yes | 78% | 17294 |
| 12 | arxiv_2512_22473v2_page_8_img_1.png... | line | line | Yes | 100% | 9760 |
| 13 | arxiv_2512_20011v1_page_2_img_1.png... | pie | pie | Yes | 100% | 4077 |
| 14 | arxiv_2405_07001v4_page_20_img_1.png... | scatter | scatter | Yes | 100% | 1232 |
| 15 | arxiv_1606_00092v2_page_44_img_8.png... | line | line | Yes | 99% | 1670 |
| 16 | arxiv_2012_07719v2_page_8_img_1.png... | scatter | scatter | Yes | 78% | 8093 |
| 17 | arxiv_2601_09473v1_page_2_img_6.png... | area | area | Yes | 99% | 7073 |
| 18 | arxiv_2502_01184v1_page_13_img_2.png... | histogram | histogram | Yes | 99% | 3241 |
| 19 | arxiv_2406_15352v2_page_2_img_12.png... | area | area | Yes | 99% | 4389 |
| 20 | arxiv_2506_19821v1_page_19_img_4.png... | area | area | Yes | 97% | 16353 |
| 21 | arxiv_2601_09250v1_page_7_img_3.png... | bar | bar | Yes | 94% | 41087 |
| 22 | arxiv_2312_16707v1_page_25_img_10.png... | line | line | Yes | 100% | 1638 |
| 23 | arxiv_2504_18540v1_page_29_img_2.png... | scatter | scatter | Yes | 85% | 2749 |
| 24 | arxiv_2510_04514v2_page_28_img_1.png... | pie | pie | Yes | 99% | 9673 |
| 25 | arxiv_2405_07001v4_page_15_img_4.png... | bar | bar | Yes | 96% | 7289 |
| 26 | arxiv_2010_02319v1_p14_img00.png... | bar | bar | Yes | 100% | 784 |
| 27 | arxiv_2412_19146v1_page_2_img_1.png... | pie | pie | Yes | 100% | 9496 |
| 28 | arxiv_2601_09100v2_page_9_img_1.png... | scatter | scatter | Yes | 79% | 844 |
| 29 | arxiv_1806_00799v1_page_9_img_2.png... | pie | pie | Yes | 92% | 930 |
| 30 | arxiv_2111_14330v2_page_21_img_7.png... | scatter | scatter | Yes | 80% | 8410 |
| 31 | arxiv_2512_18745v1_page_39_img_18.png... | pie | pie | Yes | 97% | 1570 |
| 32 | arxiv_2601_08668v1_page_20_img_3.png... | area | area | Yes | 100% | 13941 |
| 33 | arxiv_2509_10330v1_page_11_img_1.png... | histogram | histogram | Yes | 89% | 39678 |
| 34 | arxiv_2601_08668v1_p17_img10.png... | area | area | Yes | 100% | 24006 |
| 35 | arxiv_2512_22749v1_page_54_img_4.png... | histogram | histogram | Yes | 100% | 25843 |
| 36 | arxiv_2503_19186v1_page_12_img_7.png... | scatter | scatter | Yes | 77% | 3725 |
| 37 | arxiv_2511_20910v2_page_27_img_1.png... | line | line | Yes | 100% | 51212 |
| 38 | arxiv_2501_11305v2_page_20_img_1.png... | scatter | scatter | Yes | 100% | 8368 |
| 39 | arxiv_2008_12537v2_page_16_img_1.png... | histogram | histogram | Yes | 100% | 1492 |
| 40 | arxiv_2106_05738v1_page_18_img_3.png... | scatter | scatter | Yes | 98% | 13002 |
| 41 | arxiv_2601_18110v1_page_6_img_1.png... | area | area | Yes | 99% | 4705 |
| 42 | arxiv_2601_08668v1_page_8_img_2.png... | area | area | Yes | 100% | 26175 |
| 43 | arxiv_2401_04752v1_page_23_img_4.png... | line | line | Yes | 100% | 1754 |
| 44 | arxiv_2506_10116v1_page_4_img_19.png... | pie | pie | Yes | 99% | 5444 |
| 45 | arxiv_2601_11232v1_p29_img02.png... | histogram | histogram | Yes | 100% | 6764 |
| 46 | arxiv_1710_07300v2_p19_img01.png... | line | line | Yes | 100% | 1549 |
| 47 | arxiv_2512_23565v3_p06_img00.png... | pie | pie | Yes | 100% | 8121 |
| 48 | arxiv_2601_16473v1_page_5_img_1.png... | histogram | histogram | Yes | 87% | 22967 |
| 49 | arxiv_2510_23587v1_page_1_img_10.png... | pie | pie | Yes | 88% | 1096 |
| 50 | arxiv_2511_02415v1_page_16_img_11.png... | line | line | Yes | 71% | 4864 |
| 51 | arxiv_2601_13238v1_page_11_img_4.png... | pie | pie | Yes | 87% | 15052 |
| 52 | arxiv_2407_21038v1_page_12_img_2.png... | bar | bar | Yes | 75% | 3947 |
| 53 | arxiv_2601_08668v1_p16_img05.png... | area | area | Yes | 100% | 25592 |
| 54 | arxiv_2412_18798v2_page_7_img_8.png... | bar | bar | Yes | 85% | 28243 |
| 55 | arxiv_2309_00635v1_page_17_img_1.png... | line | line | Yes | 100% | 2363 |
| 56 | arxiv_2306_01841v1_page_8_img_1.png... | histogram | histogram | Yes | 82% | 1560 |
| 57 | arxiv_2509_09995v3_page_9_img_1.png... | line | line | Yes | 100% | 3128 |
| 58 | arxiv_2601_07327v1_page_63_img_3.png... | histogram | histogram | Yes | 93% | 11007 |
| 59 | arxiv_2511_14073v2_page_5_img_4.png... | bar | bar | Yes | 95% | 20569 |
| 60 | arxiv_2509_26126v1_page_19_img_1.png... | bar | bar | Yes | 79% | 7550 |

</details>

## Observations

### Strengths
- Processing pipeline handles diverse chart styles
- Confidence scores provide reliability indicators
- Vectorization compression is efficient

### Areas for Improvement
- Some chart types may need more training data
- OCR post-processing could be tuned for academic charts
- Consider ensemble methods for ambiguous cases

## Next Steps

1. Fine-tune classifier on misclassified examples
2. Add more chart type-specific features
3. Implement Stage 4 semantic reasoning for value extraction
