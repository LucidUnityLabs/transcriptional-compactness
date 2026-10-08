"""Reproduce the complete lung specimen/donor map from the deposited XLSX.

Uses the OOXML cell identities directly; the source's malformed optional
custom-properties entry must not affect scientific table extraction.
"""
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
import csv
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/GSE131907_kim_nsclc/GSE131907_Lung_Cancer_Feature_Summary.xlsx'
NS={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
def extract():
    with zipfile.ZipFile(SOURCE) as z:
        shared=ET.fromstring(z.read('xl/sharedStrings.xml'))
        strings=[''.join(n.itertext()) for n in shared.findall('m:si',NS)]
        table=ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        values={}
        for c in table.findall('.//m:c',NS):
            v=c.find('m:v',NS)
            if v is not None: values[c.attrib['r']]=strings[int(v.text)] if c.attrib.get('t')=='s' else v.text
    pairs=[(values[f'C{i}'],values[f'B{i}']) for i in range(4,62)]
    if len({s for s,d in pairs})!=58 or len({d for s,d in pairs})!=44:
        raise ValueError('source crosswalk dimensions changed; review deposited workbook')
    return sorted(pairs)
def main():
    actual=extract();dest=ROOT/'config/GSE131907_specimen_donor.tsv'
    with dest.open() as f: prior=[(r['sample'],r['donor']) for r in csv.DictReader(f,delimiter='\t')]
    if prior!=actual:raise SystemExit('SOURCE_CROSSWALK_MISMATCH')
    print('verified exact 58 specimen / 44 donor crosswalk against deposited workbook')
if __name__=='__main__':main()
