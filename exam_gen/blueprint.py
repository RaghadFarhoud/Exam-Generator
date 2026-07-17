"""
blueprint.py — جدول المواصفات (Table of Specifications).

يُبنى المخطط أولاً قبل توليد أي سؤال، ليضمن أن التوزيع الإحصائي
للامتحان المولّد يطابق بصمة الدكتور (cognitive level × difficulty × topics).
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List
import random


@dataclass
class Slot:
    """خانة واحدة في المخطط = سؤال واحد مطلوب توليده."""
    index: int
    topic: str
    cognitive_level: str
    difficulty: str


def _largest_remainder(weights: Dict[str, float], n: int) -> Dict[str, int]:
    """توزيع n عنصراً على فئات حسب أوزان، مع تصحيح بطريقة أكبر باقٍ."""
    total = sum(weights.values()) or 1.0
    raw = {k: (v / total) * n for k, v in weights.items()}
    floor = {k: int(v) for k, v in raw.items()}
    remaining = n - sum(floor.values())
    frac = sorted(weights, key=lambda k: raw[k] - floor[k], reverse=True)
    for k in frac[:remaining]:
        floor[k] += 1
    return floor


def profile_distributions(style_profile: dict):
    cog = style_profile.get("cognitive_level_distribution", {})
    dif = style_profile.get("difficulty_distribution", {})
    return cog, dif


def build_blueprint(corpus, n_questions: int, seed: int = 0,
                    topic_whitelist: List[str] | None = None,
                    coverage: bool = False) -> List[Slot]:
    """
    يبني قائمة خانات. المنطق:
      - توزيع cognitive_level و difficulty حسب بروفايل الدكتور.
      - توزيع المواضيع بحسب التغطية في بنك الأسئلة (مع خيار whitelist).
      - coverage=True: يضمن ظهور كل موضوع مرة على الأقل قبل التوزيع التناسبي
        (للامتحان الشامل). يتطلب n_questions >= عدد المواضيع للتغطية الكاملة.
    """
    rng = random.Random(seed)
    cog_dist, dif_dist = profile_distributions(corpus.style_profile)
    if not cog_dist:
        cog_dist = Counter(q["academic"]["cognitive_level"] for q in corpus.exam_questions)
    if not dif_dist:
        dif_dist = Counter(q["academic"]["difficulty"] for q in corpus.exam_questions)

    cog_alloc = _largest_remainder(dict(cog_dist), n_questions)
    dif_alloc = _largest_remainder(dict(dif_dist), n_questions)

    cog_pool = [c for c, k in cog_alloc.items() for _ in range(k)]
    dif_pool = [d for d, k in dif_alloc.items() for _ in range(k)]
    rng.shuffle(cog_pool)
    rng.shuffle(dif_pool)

    topics = topic_whitelist or corpus.topics
    topic_counts = Counter(q["topic"] for q in corpus.exam_questions)

    if coverage:
        # كل موضوع مرة أولاً، ثم املأ الباقي تناسبياً بالأهمية
        topic_pool = list(topics)[:n_questions]
        remaining = n_questions - len(topic_pool)
        if remaining > 0:
            weights = {t: topic_counts.get(t, 1) for t in topics}
            extra = _largest_remainder(weights, remaining)
            topic_pool += [t for t, k in extra.items() for _ in range(k)]
        if len(topics) > n_questions:
            # مواضيع أكثر من الأسئلة: خذ الأهم (الأكثر تكراراً) لضمان الأهم
            topic_pool = [t for t, _ in topic_counts.most_common()][:n_questions]
    else:
        weights = {t: topic_counts.get(t, 1) for t in topics}
        topic_alloc = _largest_remainder(weights, n_questions)
        topic_pool = [t for t, k in topic_alloc.items() for _ in range(k)]
    rng.shuffle(topic_pool)

    slots = []
    for i in range(n_questions):
        slots.append(Slot(
            index=i + 1,
            topic=topic_pool[i] if i < len(topic_pool) else rng.choice(topics),
            cognitive_level=cog_pool[i] if i < len(cog_pool) else rng.choice(list(cog_dist)),
            difficulty=dif_pool[i] if i < len(dif_pool) else rng.choice(list(dif_dist)),
        ))
    return slots


def summarize_blueprint(slots: List[Slot]) -> str:
    cog = Counter(s.cognitive_level for s in slots)
    dif = Counter(s.difficulty for s in slots)
    top = Counter(s.topic for s in slots)
    lines = [f"Blueprint: {len(slots)} أسئلة",
             f"  cognitive: {dict(cog)}",
             f"  difficulty: {dict(dif)}",
             f"  top topics: {dict(top.most_common(6))}"]
    return "\n".join(lines)
