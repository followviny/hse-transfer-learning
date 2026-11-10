"""Build a checkpoint summary and a page about practical use."""
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp/mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import get_data_path
from matplotlib.ticker import NullLocator
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

OUT = ROOT / 'reports/practical_value.pdf'
FIGURES = ROOT / 'reports/figures'
WIDTH = A4[0] - 80
FONT_DIR = Path(get_data_path()) / 'fonts/ttf'
for name, file in [('DejaVu', 'DejaVuSans.ttf'), ('DejaVuBold', 'DejaVuSans-Bold.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / file)))
pdfmetrics.registerFontFamily('DejaVu', normal='DejaVu', bold='DejaVuBold')


def fmt(value):
    return f'{int(value):,}'.replace(',', ' ')


def figures():
    counts = pd.read_csv(ROOT / 'reports/class_counts.csv')
    points = pd.read_csv(ROOT / 'data/processed/embedding_points.csv')
    joint = points[points.joint_sample]
    balance = pd.crosstab(joint.canonical_label, joint.dataset)
    assert balance.plantdoc.eq(balance.plantvillage).all() and len(balance) == 23
    # Final-size specification: raw class-size ranges and unchanged t-SNE coordinates.
    # Ranges are observed minima/maxima, not confidence intervals. One sample, seed=42.
    # Reference grammar: OAI-VIS-008, OAI-VIS-023, OAI-VIS-013; Astra sparse axes/neutral+teal.
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .6, 'axes.edgecolor': '#999999',
                         'text.color': '#202020', 'axes.labelcolor': '#404040',
                         'xtick.color': '#555555', 'ytick.color': '#555555',
                         'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(WIDTH / 72, 2.08), facecolor='white')
    ax = fig.add_axes([.145, .25, .32, .57])
    order = [('plantvillage', 'PlantVillage', '#8D8D88', 'o'),
             ('plantdoc', 'PlantDoc', '#1F6F5F', '^')]
    for y, (dataset, name, color, marker) in zip([1, 0], order):
        values = counts.loc[counts.dataset.eq(dataset), 'n']
        lo, hi = int(values.min()), int(values.max())
        ax.plot([lo, hi], [y, y], color=color, linewidth=1.6)
        ax.scatter([lo, hi], [y, y], s=23, color=color, marker=marker, zorder=3)
        for x in [lo, hi]:
            ax.annotate(fmt(x), (x, y), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8.5)
    ax.set_yticks([1, 0], ['PlantVillage', 'PlantDoc'])
    ax.tick_params(axis='y', length=0, pad=7)
    ax.spines['left'].set_visible(False)
    ax.set_ylim(-.5, 1.65)
    ax.set_xscale('log'); ax.set_xlim(1, 12000)
    ax.set_xticks([1, 10, 100, 1000, 10000], ['1', '10', '100', '1 000', '10 000'])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel('Фото на класс (лог. шкала)', labelpad=6, fontsize=8)
    fig.text(.015, .96, 'Размеры классов до очистки', va='top', fontsize=9.5, weight='bold')
    ax2 = fig.add_axes([.55, .25, .43, .57])
    for dataset, name, color, marker in order:
        group = joint[joint.dataset.eq(dataset)]
        ax2.scatter(group.joint_x, group.joint_y, s=4, c=color, marker=marker,
                    alpha=.7, linewidths=0, label=name)
    ax2.set_aspect('equal', adjustable='datalim')
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.spines[['bottom', 'left']].set_visible(False)
    ax2.legend(loc='lower center', bbox_to_anchor=(.5, -.38), ncol=2,
               frameon=False, fontsize=8, markerscale=2, handletextpad=.3, columnspacing=1)
    fig.text(.55, .96, 'Датасеты различаются на t-SNE', va='top', fontsize=9.5, weight='bold')
    fig.text(.55, .845, f'По {len(joint) // 2} фото; 23 общих класса', fontsize=8, color='#555555')
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'svg']:
        fig.savefig(FIGURES / f'checkpoint_summary.{ext}', dpi=240, facecolor='white')
    plt.close(fig)
    return FIGURES / 'checkpoint_summary.png'


def main():
    summary = json.loads((ROOT / 'reports/eda_summary.json').read_text())
    calibration = json.loads((ROOT / 'reports/duplicate_calibration.json').read_text())
    picture = figures()
    body = ParagraphStyle('Body', fontName='DejaVu', fontSize=9.5, leading=12.8,
                          textColor='#202020', spaceAfter=6)
    heading = ParagraphStyle('Heading', parent=body, fontName='DejaVuBold', fontSize=10.5,
                             leading=14, spaceBefore=8, spaceAfter=5)
    title = ParagraphStyle('Title', parent=heading, fontSize=18, leading=23, spaceBefore=0, spaceAfter=4)
    small = ParagraphStyle('Small', parent=body, fontSize=8, leading=10.5, textColor='#5E6462')
    story = [Paragraph('Чекпойнт 2. Анализ изображений', title),
             Paragraph('Болезни растений: PlantVillage и PlantDoc', small)]
    def section(name, text):
        story.extend([Paragraph(name, heading), Paragraph(text, body)])
    section('Данные и очистка',
        'Проверили классы, размеры, форматы, яркость и резкость фотографий. '
        'В PlantVillage снимки 256×256, в PlantDoc размеры сильно различаются. '
        'Для каждого класса сохранили сетку примеров.')
    rows = [['Датасет', 'До очистки', 'После очистки', 'Классов']]
    for key, label in [('plantvillage', 'PlantVillage'), ('plantdoc', 'PlantDoc')]:
        rows.append([label, fmt(summary['raw_images'][key]), fmt(summary['usable_images'][key]), str(summary['raw_classes'][key])])
    table = Table(rows, colWidths=[150, 115, 145, WIDTH-410], hAlign='LEFT')
    table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'DejaVu'), ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('FONTNAME', (0, 0), (-1, 0), 'DejaVuBold'), ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0), ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LINEBELOW', (0, 0), (-1, 0), .5, colors.HexColor('#B6B6B0')),
        ('LINEBELOW', (0, -1), (-1, -1), .5, colors.HexColor('#D8D8D4')),
    ]))
    story.extend([table, Spacer(1, 7)])
    cleanup_text = Paragraph(
        f"Код исключил <b>{summary['exclusions']['duplicate_removed']} повтора</b> и "
        f"<b>{summary['exclusions']['duplicate_label_conflict']} фото с конфликтующими метками</b>. "
        'Точные копии определяли по SHA-256, почти-дубликаты по pHash: '
        f"Хэмминг ≤ {calibration['hamming_max']} и отличие RGB ≤ {calibration['rgb_mae_max']:.3f}. "
        'Пороги подобрали по данным PlantVillage, где авторы указали, какие фото показывают один и тот же лист. Исходники сохранены.', body)
    # Illustration of an actual exclusion, selected from automatic rule results.
    manifest = pd.read_csv(ROOT / 'splits/manifest.csv', keep_default_na=False).set_index('image_id')
    pairs = pd.read_csv(ROOT / 'data/processed/duplicate_pairs.csv.gz')
    eligible = pairs[pairs.near_duplicate & pairs.right_id.map(manifest.duplicate_of).eq(pairs.left_id)]
    # Choose a readable illustration with a visible scale difference;
    # this selection only affects the PDF, not the automatic cleanup.
    pair = eligible.loc[
        eligible.left_id.eq('7c29fd9b980bce45e1a9')
        & eligible.right_id.eq('0aec6b491ffc1cc9eb5a')
    ].iloc[0]
    assert pair.distance <= calibration['hamming_max']
    assert pair.rgb_mae <= calibration['rgb_mae_max']
    assert manifest.loc[pair.left_id, 'split'] != 'excluded'
    assert manifest.loc[pair.right_id, 'exclusion_reason'] == 'duplicate_removed'
    photo_style = ParagraphStyle('PhotoCaption', parent=small, alignment=1, spaceAfter=0)
    photos, captions = [], []
    from PIL import Image as PILImage
    for image_id, status in [(pair.left_id, 'Оставили'), (pair.right_id, 'Исключили повтор')]:
        path = ROOT / manifest.loc[image_id, 'path']
        with PILImage.open(path) as source:
            width, height = source.size
        photo = Image(str(path))
        photo._restrictSize(76, 62)
        photos.append(photo)
        captions.append(Paragraph(f'{width}×{height}<br/>{status}', photo_style))
    photo_table = Table([photos, captions], colWidths=[84, 84], rowHeights=[64, 25])
    photo_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0), ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0), ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    illustrations = [Paragraph('<b>Пример почти-дубликата</b>', photo_style), photo_table,
                     Paragraph('Различается масштаб листа', photo_style),
                     Paragraph(f'Хэмминг: {int(pair.distance)}; RGB: {pair.rgb_mae:.6f}', photo_style)]
    cleanup_row = Table([[cleanup_text, illustrations]], colWidths=[WIDTH-184, 184], hAlign='LEFT')
    cleanup_row.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (0, 0), 14), ('RIGHTPADDING', (1, 0), (1, 0), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0), ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
    ]))
    story.append(cleanup_row)
    story.append(Paragraph(
        'Для сравнения сопоставили названия классов. Из 28 классов PlantDoc не включили '
        '4 с неоднозначными названиями и 1 без тестовых примеров. Получилось <b>23 общих класса</b>.', body))
    section('Что показал анализ',
        'Классы несбалансированы. PlantDoc отличается более сложным фоном и более насыщенными цветами. '
        'Взяли готовую ResNet18 и построили t-SNE ее признаков: даже при одинаковом составе классов '
        'снимки разделяются по датасетам. Поэтому будущую модель отдельно проверим на PlantDoc.')
    story.extend([Image(str(picture), width=WIDTH, height=2.08*72), Spacer(1, 2)])
    section('Что подготовили для обучения',
        'Сохранили разбиение с <b>seed=42</b>: PlantVillage примерно 70/15/15 для train/val/test; '
        'PlantDoc по авторским train/test после очистки. Снимки одного листа держим в одной части, '
        'чтобы они не попадали одновременно в обучение и проверку. '
        'Посмотрели отражения, повороты, изменение цвета и обрезку; для начала выбрали мягкие аугментации. ')
    story.extend([Spacer(1, 4), Paragraph(
        'Расчеты и графики: EDA.ipynb. Исключения: reports/exclusions.csv. Разбиение: splits/manifest.csv.<br/>'
        'Источники: <link href="https://huggingface.co/datasets/mohanty/PlantVillage" color="#1F6F5F">PlantVillage</link> '
        'и <link href="https://github.com/pratikkayal/PlantDoc-Dataset" color="#1F6F5F">PlantDoc</link>, версии авторов.', small)])
    eda_story = story
    story = []
    practical = ParagraphStyle('Practical', parent=body, fontSize=11, leading=16, spaceAfter=13)
    story.extend([Paragraph('Практическая применимость проекта', title), Spacer(1, 10)])
    story.append(Paragraph(
        'Сервис может пригодиться садоводам и сотрудникам небольших хозяйств. Пользователь загружает фото листа '
        'и получает подсказку, на какую болезнь похожи симптомы. Так будет проще понять, какие растения '
        'стоит показать агроному. Это может сэкономить время, когда растений много, а специалиста рядом нет.', practical))
    story.append(Paragraph(
        'Рабочий прототип можно сделать за время проекта. Сложнее добиться того, чтобы он хорошо распознавал '
        'болезни на обычных фото: там будут другой свет, фон и сорта растений. Для проверки понадобятся '
        'снимки из хозяйств, на которых специалист подтвердил состояние листьев. Отдельно нужно проверить, '
        'как модель ведет себя с болезнями, которых не было в обучении. Самая серьезная проблема '
        'возникнет, если она назовет больной лист здоровым и человек отложит проверку. Обратная ошибка '
        'тоже может навредить: здоровое растение начнут обрабатывать без необходимости. Поэтому ответ '
        'сервиса стоит использовать как подсказку, а лечение обсуждать с агрономом.', practical))
    story.append(Paragraph(
        'Заказчик сможет проверить модель на своих фотографиях: сравнить ее ответы с оценкой агронома '
        'и посмотреть, какие болезни она чаще путает.', practical))
    story.extend([Spacer(1, 14), *eda_story])
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=40, rightMargin=40,
                            topMargin=33, bottomMargin=30,
                            title='Чекпойнт 2: анализ PlantVillage и PlantDoc',
                            author='Виктория Фокина; Владимир Фатьянов')
    doc.build(story)
    print(OUT)


if __name__ == '__main__':
    main()
