"""Extracts the non-nutrition label fields (product info, companies, claims, warnings) from OCR text.
Re-exported by business/services/ocr.py."""

import re


def extract_label_fields(raw_text: str) -> dict:
    """Extract non-nutrition fields from the full OCR text into a dict (product info, claims, warnings, recycling)."""
    fields = {
        "product_name":    None,
        "brand":           None,
        "net_weight":      None,
        "expiry_info":     None,
        "storage":         None,
        "manufacturer":    None,
        "seller":          None,
        "importer":        None,
        "origin_country":  None,
        "factory":         None,
        "contact":         None,
        "claims_jp":       [],
        "claims_en":       [],
        "warnings":        [],
        "spice_level":     None,
        "trademark":       None,
        "age_restriction": None,
        "cooking_instructions":       None,
        "allergen_declaration_style": None,
    }

    if not raw_text:
        return fields

    text = raw_text

    _LABEL_WORDS = r'(?:販売者|製造者|輸入者|工場住所|製造所|発売元|販売元|製造工場)'

    def _clean_val(val: str) -> str:
        """Strip leading/trailing company-label words and placeholder values like 未記載."""
        val = re.sub(r'^(?:[/／]?\s*' + _LABEL_WORDS + r'[/／：:\s]*)+', '', val)
        val = re.sub(r'\s*[/／]' + _LABEL_WORDS + r'.*$', '', val)
        val = val.strip()
        if val in ('未記載', '不明', 'N/A', '-', '—'):
            return ''
        return val

    for pat in [
        r'品名[：:\s]+([^\n。]{2,50})',
        r'名称[：:\s]+([^\n。]{2,50})',
        r'品名/名称[：:\s]+([^\n。]{2,50})',
    ]:
        m = re.search(pat, text)
        if m:
            v = _clean_val(m.group(1))
            if v:
                fields["product_name"] = v
                break

    m = re.search(r'内容量[：:\s]+([^\n。]{1,30})', text)
    if m:
        v = _clean_val(m.group(1))
        if v:
            fields["net_weight"] = v

    _EXPIRY_REFERENCE_PATTERNS = [
        '枠外', '外枠', '底面', '側面', 'フタ', '別途', '本体',
        '上部', '下部', '袋', 'パッケージ', '記載', '参照',
        'see', 'refer', 'outside', 'bottom', 'panel',
    ]
    for pat in [
        r'賞味期限[：:\s]+([^\n。●]{1,60})',
        r'消費期限[：:\s]+([^\n。●]{1,60})',
        r'賞味期限/消費期限[：:\s]+([^\n。●]{1,60})',
    ]:
        m = re.search(pat, text)
        if m:
            v = _clean_val(m.group(1))
            if not v:
                continue
            v = re.sub(r'[\s　]+$', '', v)
            is_duration = bool(re.search(
                r'(?:製造|購入|出荷)(?:日|より).{0,10}(?:ヶ月|か月|年|日間)', v
            ))
            if v:
                fields["expiry_info"] = v
                break

    m = re.search(r'保存方法[：:\s]+([^\n。]{1,100})', text)
    if m:
        v = _clean_val(m.group(1))
        if v:
            fields["storage"] = v

    # line-anchored match first, then anywhere in the text
    for pat in [
        r'^製造者[：:\s]+([^\n/]{2,60})',
        r'(?<![^\n])製造者[：:\s]+([^\n/]{2,60})',
        r'製造者[：:\s]+([^\n/]{2,60})',
    ]:
        m = re.search(pat, text, re.MULTILINE)
        if m:
            v = _clean_val(m.group(1))
            if v and v not in ('未記載',):
                fields["manufacturer"] = v[:60]
                break

    for pat in [
        r'^販売者[：:\s]+([^\n/]{2,60})',
        r'(?<![^\n])販売者[：:\s]+([^\n/]{2,60})',
        r'販売者[：:\s]+([^\n/]{2,60})',
    ]:
        m = re.search(pat, text, re.MULTILINE)
        if m:
            v = _clean_val(m.group(1))
            if v and v not in ('未記載',):
                fields["seller"] = v[:60]
                break

    for pat in [
        r'^製造所[：:\s]+([^\n]{5,80})',
        r'製造所[：:\s]+([^\n]{5,80})',
    ]:
        m = re.search(pat, text, re.MULTILINE)
        if m:
            v = _clean_val(m.group(1))
            if v and re.search(r'[都道府県市区町村\d]', v):
                fields["factory"] = v[:80]
                break

    m = re.search(r'輸入者[：:\s]+([^\n/]{2,60})', text)
    if m:
        v = _clean_val(m.group(1))
        if v:
            fields["importer"] = v[:60]

    JAPANESE_PLACE_NAMES = {
        '東京', '大阪', '京都', '札幌', '名古屋', '福岡', '横浜',
        '神戸', '仙台', '広島', '日本', '国内', '未記載',
    }
    for pat in [
        r'原産国[：:\s]+([^\n。●]{1,30})',
        r'原産地[：:\s]+([^\n。●]{1,30})',
    ]:
        m = re.search(pat, text)
        if m:
            val = _clean_val(m.group(1))
            if (val
                    and val not in JAPANESE_PLACE_NAMES
                    and not re.search(r'[都道府県市区町村]', val)):
                fields["origin_country"] = val
                break

    for pat in [
        r'お問い合わせ[先]?[：:\s]*((?:0120|0570)[-\s・]?\d{2,3}[-\s・]?\d{3,4})',
        r'(0120[-\s・]?\d{2,3}[-\s・]?\d{3,4})',
        r'(0570[-\s・]?\d{2,3}[-\s・]?\d{3,4})',
    ]:
        m = re.search(pat, text)
        if m:
            pre = text[max(0, m.start() - 5):m.start()]
            if '〒' not in pre:
                fields["contact"] = m.group(1).strip()
                break

    claim_patterns = [
        r'100\s*%\s*植物性油(?:脂)?使用',
        r'遺伝子組換えでない',
        r'遺伝子組換え不使用',
        r'食物繊維豊富',
        r'えん麦100\s*%',
        r'国産[^\s。、]{0,10}使用',
        r'北海道(?:産|製造)[^\s。、]{0,15}',
        r'(?:ノンカフェイン|カフェインゼロ)',
        r'低糖質',
        r'高たんぱく',
        r'分別生産流通管理[^\n。]{0,50}',
        r'ポリフェノール[^\n。]{0,40}',
    ]
    for pat in claim_patterns:
        m = re.search(pat, text)
        if m and m.group(0) not in fields["claims_jp"]:
            fields["claims_jp"].append(m.group(0).strip())

    mkt_section = re.search(r'## マーケティング情報\n([\s\S]{0,800}?)(?=##|$)', text)
    if mkt_section:
        mkt_text = mkt_section.group(1)
        for line in mkt_text.splitlines():
            line = line.strip().lstrip("・●▶-—")
            if len(line) > 3 and line not in fields["claims_jp"]:
                if any(kw in line for kw in ["使用", "入り", "豊富", "100%", "製造", "産"]):
                    fields["claims_jp"].append(line)

    en_claim_patterns = [
        r'100\s*%\s*Whole\s*Grain[^\n.]{0,40}',
        r'Naturally\s*High\s*Fiber[^\n.]{0,30}',
        r'Product\s*of\s*[A-Z][a-z]+[^\n.]{0,20}',
        r'natural\s*mineral\s*water',
        r'Store\s*in\s*a\s*cool[^\n.]{0,40}',
        r'[A-Z][a-z]+\s+[Ff]ree[^\n.]{0,20}',
    ]
    for pat in en_claim_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m and m.group(0) not in fields["claims_en"]:
            fields["claims_en"].append(m.group(0).strip())

    warning_patterns = [
        r'泣くほど辛[^\n。]{0,50}',
        r'小さなお子様[^\n。]{0,60}',
        r'辛みが苦手な方[^\n。]{0,40}',
        r'くるみの殻[^\n。]{0,50}',
        r'まれに黒い粒[^\n。]{0,40}',
        r'馬鈴薯由来[^\n。]{0,50}',
        r'製造には十分注意[^\n。]{0,50}',
        r'移り香注意',
    ]
    for pat in warning_patterns:
        m = re.search(pat, text)
        if m:
            w = m.group(0).strip()
            if w not in fields["warnings"]:
                fields["warnings"].append(w)

    m = re.search(r'辛さ\s*(\d+)', text)
    if m:
        fields["spice_level"] = int(m.group(1))

    m = re.search(r'商標登録\s*第?\s*(\d+)\s*号?', text)
    if m:
        fields["trademark"] = f"第{m.group(1)}号"

    age_m = (
        re.search(r'対象年齢\s*(\d+\s*才\s*以上)', text) or
        re.search(r'(?:対象年齢|推奨年齢)[：:\s]*(\d+\s*才[以上下まで]{0,3})', text)
    )
    if age_m:
        fields["age_restriction"] = age_m.group(1).strip()

    if "アレルゲン28品目対象" in text or "28品目対象" in text:
        fields["allergen_declaration_style"] = "28品目対象"
    elif "28品目中" in text:
        fields["allergen_declaration_style"] = "28品目中"
    elif "特定原材料等" in text:
        fields["allergen_declaration_style"] = "特定原材料等"
    elif "特定原材料" in text:
        fields["allergen_declaration_style"] = "特定原材料"

    recycling_section = re.search(
        r'## リサイクルマーク\n([\s\S]{0,400}?)(?=##|$)', text
    )
    recycle_text = recycling_section.group(1).strip() if recycling_section else text
    fields["_recycling_raw"] = recycle_text

    cooking_m = re.search(
        r'(?:## 調理方法|調理方法|召し上がり方|ご使用方法)\n([\s\S]{0,600}?)(?=##|$)',
        text,
    )
    if cooking_m:
        cooking_text = cooking_m.group(1).strip()
        if len(cooking_text) > 10:
            fields["cooking_instructions"] = cooking_text[:400]

    return fields