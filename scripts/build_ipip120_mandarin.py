#!/usr/bin/env python3
"""Build instruments/ipip120_mandarin.json and ipip120_mandarin_facet_map.json
from (a) the Mandarin item texts on IPIP's distribution page (Xu, n.d.) and
(b) Johnson's IPIP-NEO-120 Admin 120 scoring sheet from OSF.

The Mandarin items follow Johnson's canonical order (item k of the page is the
translation of Johnson item k), so the same trait/facet/keying applies.

Sources:
  https://ipip.ori.org/Mandarin%20translation%20of%20IPIP-NEO-120.htm
  https://osf.io/ycvdk/  (IPIP-NEO-120 Scoring Tool.xls, Admin 120 sheet)
"""

import json
from pathlib import Path

import pandas as pd

# Mandarin items 1..120 in Johnson's canonical order, verbatim from the IPIP
# page. The page renders these without leading "I "; we preserve them as-is
# because the Mandarin items have no "I " convention to match.
MANDARIN_ITEMS = [
    "为很多事情感到担心", "很容易交到朋友", "有丰富的想象力", "信任其他人",
    "能成功完成任务", "很容易生气", "喜欢参加大型聚会", "认为艺术是很重要的",
    "会利用他人来达成自己的目的", "喜欢整洁", "时常感到忧郁", "承担责任",
    "能强烈感受到自己的情绪", "乐于帮助他人", "信守承诺", "感觉自己很难接近别人",
    "总是很忙碌", "相比墨守成规，我更喜欢有变化。", "喜欢和别人比试", "工作努力",
    "大吃大喝寻欢作乐", "喜欢追求刺激", "喜欢阅读有挑战性的材料", "相信自己比别人更好",
    "我总是有所准备", "容易慌张", "我的快乐会感染周围的人", "认为自己在政治方面非常自由开放",
    "同情流浪汉", "会不经过思考就开始行动", "担心会发生最糟糕的情况", "和其他人待在一起时，我觉得很自在",
    "喜欢天马行空地幻想", "相信他人的出发点是好的", "在自己所从事的领域出类拔萃", "容易被激怒",
    "在聚会上，我会和很多不同的人交谈", "我能发现别人看不到的美", "为了领先别人，我会采用作弊的方式", "经常忘记把东西放回原处",
    "不喜欢自己", "我会尝试去领导他人", "能感受到他人的情绪", "关心他人",
    "我会实话实说", "害怕引起他人的注意", "总是在奔波", "我更喜欢做自己熟悉的事情",
    "会冲人大喊大叫", "我会超出预期地完成任务或工作", "很少过度放纵自己", "喜欢冒险",
    "会避免参与关于哲学的讨论", "觉得自己很了不起", "能执行自己制定的计划", "感到被各种事情压得喘不过气来",
    "我的生活很快乐", "相信没有绝对的对错之分", "我同情那些处境不如自己的人", "会做出草率的决定",
    "对很多事情感到害怕", "避免和他人接触", "喜欢做白日梦", "相信他人说的话",
    "能轻松完成任务", "发脾气", "更喜欢一个人独处", "不喜欢诗歌",
    "占别人的便宜", "我的房间很乱", "经常情绪低落", "掌控局面",
    "很少察觉到自己的情绪反应", "对他人的感受漠不关心", "破坏规则", "只有和朋友在一起的时候我才会感到自在",
    "我的闲暇时间非常充实", "不喜欢改变", "侮辱他人", "在工作或学习上，我不会多做，过得去就行",
    "能轻松抵御诱惑", "喜欢不计后果地行事", "理解抽象概念对我来说有些困难", "对自己的评价很高",
    "浪费时间", "感觉自己处理事情力不从心", "热爱生活", "我在政治上比较保守",
    "对别人遇到的麻烦事不感兴趣", "仓促行事", "我很容易觉得压力大", "和他人保持距离",
    "喜欢陷入沉思", "不相信别人", "知道如何完成任务", "不会轻易被惹恼",
    "避开人多的地方", "不喜欢去美术馆", "给别人使坏", "我的东西放得到处都是",
    "对自己感到满意", "等着别人来带头", "不理解那些情绪化的人", "不愿为他人花费时间",
    "违背自己的承诺", "不会被复杂的社交情境所困扰", "喜欢慢慢来", "我喜欢传统的方式",
    "报复别人", "对工作或学业不怎么投入时间和精力", "可以控制自己的欲望", "我的行为狂放不羁",
    "对理论性的讨论不感兴趣", "吹嘘自己的美德", "拖很久才开始做一件事", "在压力下能保持冷静",
    "看到生活中好的一面", "认为我们应该严惩犯罪", "我尽量不去想那些需要帮助的人", "做事不经过思考",
]
assert len(MANDARIN_ITEMS) == 120, f"got {len(MANDARIN_ITEMS)}"

# Load Johnson's Admin 120 scoring key.
ADMIN = pd.read_excel("/tmp/ipip120_scoring.xls", sheet_name="Admin 120")
assert ADMIN.shape[0] == 120

TRAIT_CODE = {"N": "N", "E": "E", "O": "O", "A": "A", "C": "C"}
TRAIT_NAME = {
    "N": "Neuroticism", "E": "Extraversion", "O": "Openness",
    "A": "Agreeableness", "C": "Conscientiousness",
}
TRAIT_ORDER = ["N", "E", "O", "A", "C"]

# Sanity-check first 5 items match expected English texts.
expected_first = [
    "Worry about things", "Make friends easily", "Have a vivid imagination",
    "Trust others", "Complete tasks successfully",
]
for i, expected in enumerate(expected_first):
    actual = ADMIN.iloc[i]["Text"]
    assert actual == expected, f"item {i+1}: expected {expected!r}, got {actual!r}"

items_flat = {}              # ipip120m_1 → "为很多事情..."
items_meta = {}              # ipip120m_1 → {trait, facet, pole, text, text_en}
trait_to_ids = {t: [] for t in TRAIT_ORDER}
trait_to_reverse = {t: [] for t in TRAIT_ORDER}
facet_to_ids = {}
facet_to_reverse = {}

for i, row in ADMIN.iterrows():
    k = int(row["M5-120 #"])
    assert k == i + 1, f"row {i} has M5-120 # {k}"
    facet_code = str(row["Facet"]).strip()           # e.g. "N1"
    facet_name = str(row["Facet Name"]).strip()      # e.g. "Anxiety"
    fr = str(row["Score F/R"]).strip()                # "F" or "R"
    text_en = str(row["Text"]).strip()
    reverse = (fr == "R")
    trait = facet_code[0]                              # leading letter
    assert trait in TRAIT_ORDER, f"unknown trait letter {trait!r} in {facet_code!r}"

    item_id = f"ipip120m_{k}"
    mandarin = MANDARIN_ITEMS[k - 1]
    items_flat[item_id] = mandarin
    items_meta[item_id] = {
        "trait": trait,
        "facet": facet_name,
        "facet_code": facet_code,
        "pole": "rev" if reverse else "fwd",
        "text": mandarin,
        "text_en": text_en,
    }
    trait_to_ids[trait].append(item_id)
    if reverse:
        trait_to_reverse[trait].append(item_id)
    facet_to_ids.setdefault(facet_name, []).append(item_id)
    if reverse:
        facet_to_reverse.setdefault(facet_name, []).append(item_id)

# Build the main instrument JSON.
SOURCE_NOTE = (
    "Mandarin items: Xu, Z. (n.d.). Mandarin translation of the IPIP-NEO-120. "
    "International Personality Item Pool. "
    "https://ipip.ori.org/Mandarin translation of IPIP-NEO-120.htm "
    "(reliability/validity: https://ipip.ori.org/ChineseIPIP-120reliability.htm; "
    "Cronbach's alphas .80-.92 on the 5 majors, convergent r=.52-.83 with the "
    "Big Five Personality Scale 10-item form). "
    "Scoring key: Johnson, J. A. (2014). Development of the IPIP-NEO-120. "
    "Journal of Research in Personality, 51, 78-89; "
    "Admin 120 sheet from https://osf.io/ycvdk/. "
    "Mandarin items follow Johnson's canonical 1..120 order, so the same "
    "trait/facet/reverse-keying assignments apply unchanged."
)

instrument = {
    "_meta": {
        "name": "IPIP-NEO-120 Mandarin Chinese",
        "source": SOURCE_NOTE,
        "trait_scales": [f"IPIP120M-{t}" for t in TRAIT_ORDER],
        "n_items": 120,
        "item_id_format": "ipip120m_{1..120}, matching Johnson IPIP-NEO-120 admin order",
        "facet_map_path": "instruments/ipip120_mandarin_facet_map.json",
    },
    "user_readable_name": "IPIP-NEO-120 (Mandarin)",
    "items": items_flat,
    "scales": {
        f"IPIP120M-{t}": {
            "user_readable_name": f"IPIP-NEO-120 {TRAIT_NAME[t]} (Mandarin)",
            "item_ids": trait_to_ids[t],
            "reverse_keyed_item_ids": trait_to_reverse[t],
        }
        for t in TRAIT_ORDER
    },
}

# Build the facet_map (per-item trait/facet/pole/text) — mirrors ipip300_facet_map.json.
facet_names_per_trait = {t: [] for t in TRAIT_ORDER}
seen = set()
for item_id, meta in items_meta.items():
    key = (meta["trait"], meta["facet"])
    if key not in seen:
        facet_names_per_trait[meta["trait"]].append(meta["facet"])
        seen.add(key)

facet_map = {
    "_method": {
        "source": ("Johnson IPIP-NEO-120 canonical scoring (Admin 120 sheet, "
                   "https://osf.io/ycvdk/). Mandarin item texts: Xu (n.d.) on IPIP."),
        "trait_cycle": ["N", "E", "O", "A", "C"],
        "facet_names_per_trait": facet_names_per_trait,
    },
    "items": items_meta,
}

INSTR_OUT = Path("instruments/ipip120_mandarin.json")
FACET_OUT = Path("instruments/ipip120_mandarin_facet_map.json")
INSTR_OUT.write_text(json.dumps(instrument, ensure_ascii=False, indent=2))
FACET_OUT.write_text(json.dumps(facet_map, ensure_ascii=False, indent=2))

# Sanity summary
print(f"wrote {INSTR_OUT} ({INSTR_OUT.stat().st_size:,} bytes)")
print(f"wrote {FACET_OUT} ({FACET_OUT.stat().st_size:,} bytes)")
print(f"\ntrait counts (items / reverse-keyed):")
for t in TRAIT_ORDER:
    print(f"  {t} ({TRAIT_NAME[t]:<18s}): {len(trait_to_ids[t]):>3d} items, "
          f"{len(trait_to_reverse[t]):>3d} reverse-keyed")
total = sum(len(v) for v in trait_to_ids.values())
total_rev = sum(len(v) for v in trait_to_reverse.values())
print(f"  TOTAL                     : {total:>3d} items, {total_rev:>3d} reverse-keyed")

print(f"\nfacets ({len(facet_to_ids)} total, expected 30):")
for t in TRAIT_ORDER:
    print(f"  {t}: {facet_names_per_trait[t]}")
