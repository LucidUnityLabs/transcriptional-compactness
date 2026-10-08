"""Read author-reported BCC responses from Yost 2019 Supplementary Table 1.

Response is the literal clinical Response column, not a reconstructed
RECIST threshold. In particular su002 is Yes despite its annotated 0*.
SCC lesions, including su010-S, are excluded from the BCC contrast.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/E7_clonality/Yost2019_SupplementaryTables.xlsx'
NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def extract(source=SOURCE):
    with zipfile.ZipFile(source) as z:
        shared = ET.fromstring(z.read('xl/sharedStrings.xml'))
        strings = [''.join(t.text or '' for t in n.findall('.//m:t', NS))
                   for n in shared.findall('m:si', NS)]
        sheet = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        values = {}
        for cell in sheet.findall('.//m:c', NS):
            value = cell.find('m:v', NS)
            if value is not None:
                values[cell.attrib['r']] = strings[int(value.text)] \
                    if cell.attrib.get('t') == 's' else value.text
    if [values.get(f'{c}4') for c in ('A', 'B', 'F')] != \
            ['Patient', 'Tumor Type', 'Response']:
        raise ValueError('author clinical table header changed')
    result = {}
    for row in range(5, 20):
        if values.get(f'B{row}') != 'BCC':
            continue
        patient, response = values[f'A{row}'], values[f'F{row}']
        if patient in result or response not in ('Yes', 'No'):
            raise ValueError('duplicate or uninterpretable author BCC response')
        result[patient] = 'Responder' if response == 'Yes' else 'Non-responder'
    if len(result) != 11:
        raise ValueError('author BCC cohort coverage changed')
    return result


if __name__ == '__main__':
    import json
    print(json.dumps(extract(), indent=2, sort_keys=True))
