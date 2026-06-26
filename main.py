import re
import json
import random
from pathlib import Path

import numpy as np
import jieba
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

# ── 全局常量 ──────────────────────────────────────────────

DATA_FILE = 'songci300.json'

COLORS = ['#E4C7C2', '#CACDCF', '#EBE0DD', '#B5A29E', '#F0CCBC']

FONT_CANDIDATES = [
    Path('simhei.ttf'),
    Path('msyh.ttc'),
    Path('C:/Windows/Fonts/simhei.ttf'),
    Path('C:/Windows/Fonts/msyh.ttc'),
]

STOP_WORDS = {
    '不知', '如何', '何处', '何事', '何年', '何时', '何日', '何如',
    '不是', '不得', '不能', '不会', '不应', '不堪', '不忍',
    '只是', '只有', '至今', '而今', '如今',
    '人间', '天上', '东风', '西风', '南风', '北风', '春风', '秋月',
    '一片', '一声', '一缕', '一抹', '一晌', '一霎', '一川',
    '无端', '无奈', '无情', '有缘', '有恨', '有意', '无心',
    '相思', '相忆', '相见', '相逢', '相伴', '相随',
    '依旧', '依然', '犹自', '犹然', '尚自',
    '多少', '几多', '几许', '许多', '这般',
    '莫是', '莫道', '莫不', '莫非',
    '应是', '料得', '料想',
    '正是', '恰好', '恰似', '恰如',
    '年来', '去时', '去后', '来时', '来后',
    '当时', '当日', '当年', '此时', '此日', '此际',
    '那堪', '那更', '那知', '怎奈', '怎知',
    '谁念', '谁道', '谁教', '谁信',
    '但见', '但闻', '但使',
    '又见', '又闻', '又听',
    '更那', '更堪', '更闻',
    '还有', '还似', '还如',
    '便是', '便做', '便教',
    '纵有', '纵然', '纵使',
    '若得', '若教', '若使',
    '向时', '向者', '向来',
}

SEASON_KEYWORDS = ['春', '夏', '秋', '冬', '风', '花', '雪', '月', '山', '水']

HANZI_RE = re.compile(r'[\u4e00-\u9fff]')


# ── 工具函数 ──────────────────────────────────────────────

def init_plot_style():
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False


def find_font_path():
    for fp in FONT_CANDIDATES:
        if fp.exists():
            return str(fp)
    return None


def cycle_color(i):
    return COLORS[i % len(COLORS)]


def extract_paragraphs(data):
    sentences = []
    for item in data:
        sentences.extend(item['paragraphs'])
    return sentences


def count_hanzi(text):
    return len(HANZI_RE.findall(text))


def save_chart(name):
    plt.tight_layout()
    plt.savefig(f'{name}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"已保存 {name}.png")


def draw_bar_labels(bars, fontsize=10, offset=0):
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., h + offset,
                 f'{int(h)}', ha='center', va='bottom', fontsize=fontsize)


# ── 数据加载 ──────────────────────────────────────────────

def load_data():
    data = json.loads(Path(DATA_FILE).read_text(encoding='utf-8'))
    return data


# ── 图表绘制 ──────────────────────────────────────────────

def plot_author_pie(authors_count):
    top = authors_count.most_common(10)
    labels, values = zip(*top)

    plt.figure(figsize=(10, 8))
    wedges, texts, autotexts = plt.pie(
        values, labels=labels, autopct='%1.1f%%',
        colors=[cycle_color(i) for i in range(len(top))],
        startangle=90, wedgeprops={'edgecolor': 'none'},
    )
    for t in texts:
        t.set_fontsize(10)
    for t in autotexts:
        t.set_fontsize(9)
        t.set_color('#555555')

    plt.title('前10位作者作品数量占比', fontsize=14, pad=20)
    save_chart('pie_chart')


def plot_rhythmic_bar(rhythmic_count):
    top = rhythmic_count.most_common(10)
    labels, values = zip(*top)

    plt.figure(figsize=(12, 8))
    bars = plt.bar(
        range(len(top)), values,
        color=[cycle_color(i) for i in range(len(top))],
        edgecolor='none',
    )
    plt.xticks(range(len(top)), labels, rotation=30, ha='right', fontsize=10)
    plt.ylabel('作品数量', fontsize=12)
    plt.title('前10个词牌名作品数量', fontsize=14, pad=20)
    draw_bar_labels(bars, fontsize=9)
    save_chart('bar_chart')


def _wordcloud_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return random.choice(COLORS)


def plot_wordcloud(data):
    font_path = find_font_path()
    if not font_path:
        print("未找到中文字体，跳过词云生成。")
        return

    text = ' '.join(extract_paragraphs(data))
    words = [
        w for w in jieba.lcut(text)
        if len(w) >= 2 and all('\u4e00' <= c <= '\u9fff' for c in w) and w not in STOP_WORDS
    ]

    wc = WordCloud(
        font_path=font_path, width=1600, height=900,
        background_color='white', color_func=_wordcloud_color_func,
        max_words=300, max_font_size=120, min_font_size=12,
        random_state=42, collocations=False,
    ).generate_from_frequencies(Counter(words))

    plt.figure(figsize=(16, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    save_chart('word_cloud')


def plot_length_hist(data):
    lengths = [
        sum(count_hanzi(s) for s in item['paragraphs'])
        for item in data
    ]
    lengths = [n for n in lengths if n > 0]

    bins = [0, 30, 60, 90, 120, max(lengths) + 1]
    bin_labels = ['0-30字', '30-60字', '60-90字', '90-120字', '120字以上']
    counts, _ = np.histogram(lengths, bins=bins)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        range(len(bin_labels)), counts,
        color=[cycle_color(i) for i in range(len(bin_labels))],
        edgecolor='none', width=0.7,
    )
    plt.xticks(range(len(bin_labels)), bin_labels, fontsize=11)
    plt.xlabel('字数区间', fontsize=12)
    plt.ylabel('词的数量', fontsize=12)
    plt.title('单首词总字数分布（小令/中调/长调）', fontsize=14, pad=20)
    draw_bar_labels(bars)
    save_chart('word_length_histogram')


def plot_season_imagery(data):
    text = ' '.join(extract_paragraphs(data))
    kw_counts = {kw: text.count(kw) for kw in SEASON_KEYWORDS}
    sorted_items = sorted(kw_counts.items(), key=lambda x: x[1], reverse=True)
    labels, values = zip(*sorted_items)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        range(len(labels)), values,
        color=[cycle_color(i) for i in range(len(labels))],
        edgecolor='none',
    )
    plt.xticks(range(len(labels)), labels, fontsize=12)
    plt.ylabel('出现次数', fontsize=12)
    plt.title('四季意象词频对比', fontsize=14, pad=20)
    draw_bar_labels(bars)
    save_chart('season_imagery_chart')


def plot_rhyme_wordcloud(data):
    font_path = find_font_path()
    if not font_path:
        print("未找到中文字体，跳过韵脚词云生成。")
        return

    rhyme_chars = []
    for sentence in extract_paragraphs(data):
        chars = HANZI_RE.findall(sentence)
        if chars:
            rhyme_chars.append(chars[-1])

    wc = WordCloud(
        font_path=font_path, width=1600, height=900,
        background_color='white', color_func=_wordcloud_color_func,
        max_words=200, max_font_size=120, min_font_size=14,
        random_state=42,
    ).generate_from_frequencies(Counter(rhyme_chars))

    plt.figure(figsize=(16, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    save_chart('rhyme_word_cloud')


# ── 入口 ──────────────────────────────────────────────────

if __name__ == '__main__':
    init_plot_style()

    data = load_data()

    authors_count = Counter(item['author'] for item in data)
    rhythmic_count = Counter(item['rhythmic'] for item in data)

    plot_author_pie(authors_count)
    plot_rhythmic_bar(rhythmic_count)
    plot_wordcloud(data)
    plot_length_hist(data)
    plot_season_imagery(data)
    plot_rhyme_wordcloud(data)

    print("六张图表已全部生成并保存！")
