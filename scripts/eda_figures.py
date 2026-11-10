"""Plotting helpers; figure data and explanations are in EDA.ipynb."""
from pathlib import Path
import math
import textwrap
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageOps
from eda_core import ROOT, SEED

FIGURES = ROOT / "reports/figures"
COLORS = {"plantvillage": "#858982", "plantdoc": "#1F6F5F"}
NAMES = {"plantvillage": "PlantVillage", "plantdoc": "PlantDoc"}


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 13, "axes.labelsize": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#999999", "axes.linewidth": 0.65,
                         "text.color": "#171717", "axes.labelcolor": "#171717",
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "figure.dpi": 100, "savefig.dpi": 160,
                         "svg.fonttype": "none"})


def finish(fig, name, vector=True):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight", facecolor="white")
    if vector:
        fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight", facecolor="white")
    if vector:
        plt.show()
    else:
        # Photograph-heavy grids are JPEG only in the notebook to keep its size
        # suitable for GitHub; lossless full-resolution PNGs remain in reports.
        from IPython.display import Image as DisplayImage, display
        stream = io.BytesIO()
        fig.savefig(stream, format="jpeg", dpi=100, bbox_inches="tight", facecolor="white", pil_kwargs={"quality":88})
        display(DisplayImage(data=stream.getvalue(), format="jpeg"))
        plt.close(fig)


def label(text):
    return text.replace("___", " · ").replace("_", " ").replace("(including sour)", "").strip()


def class_counts(frame):
    for dataset in NAMES:
        counts = frame[frame.dataset.eq(dataset)].label.value_counts()
        for start in range(0, len(counts), 19):
            section = counts.iloc[start:start+19].sort_values()
            fig, ax = plt.subplots(figsize=(12, max(4, len(section)*0.34)), layout="constrained")
            ax.barh([label(x) for x in section.index], section.values, color=COLORS[dataset], height=0.66)
            ax.set_xlim(0, section.max()*1.18)
            for y, n in enumerate(section.values):
                ax.text(n + section.max()*0.018, y, f"{n:,}".replace(",", " "), va="center", fontsize=10)
            ax.set_xlabel("Количество исходных изображений, шт.")
            ax.set_title(f"{NAMES[dataset]}: классы по числу изображений ({start+1}–{start+len(section)})", loc="left", pad=14)
            finish(fig, f"counts_{dataset}_{start//19+1}")


def thumbnail(path, size=(220, 180)):
    with Image.open(ROOT / path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return ImageOps.pad(image, size, color="#F5F5F2")


def class_galleries(frame):
    valid = frame[frame.error.eq("")]
    for dataset in NAMES:
        groups = list(valid[valid.dataset.eq(dataset)].groupby("label", sort=True))
        for start in range(0, len(groups), 8):
            subset = groups[start:start+8]
            fig, axes = plt.subplots(math.ceil(len(subset)/2), 2, figsize=(12, math.ceil(len(subset)/2)*2.5), squeeze=False, layout="constrained")
            for ax in axes.flat:
                ax.axis("off")
            for ax, (name, group) in zip(axes.flat, subset):
                sample = group.sample(n=min(2, len(group)), random_state=SEED)
                strip = Image.new("RGB", (440, 180), "white")
                for i, path in enumerate(sample.path):
                    strip.paste(thumbnail(path), (i*220, 0))
                ax.imshow(strip)
                ax.set_title(textwrap.fill(label(name), 44), fontsize=11, loc="left")
            fig.suptitle(f"{NAMES[dataset]} · по 2 примера на класс · seed={SEED}", x=0.01, ha="left", fontsize=14)
            finish(fig, f"examples_{dataset}_{start//8+1}", vector=False)


def dimension_plots(valid):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ds in NAMES:
        group = valid[valid.dataset.eq(ds)]
        for ax, col, title in zip(axes, ["width", "height", "aspect_ratio"], ["Ширина, пиксели", "Высота, пиксели", "Ширина / высота"]):
            x = np.sort(group[col].astype(float))
            ax.step(x, np.arange(1, len(x)+1)/len(x), where="post", color=COLORS[ds], label=NAMES[ds], linestyle="--" if ds=="plantvillage" else "-")
            ax.set_xlabel(title)
            ax.set_xscale("log")
            ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Доля изображений ≤ значения по оси X")
    axes[0].legend(frameon=False)
    fig.suptitle("Размеры PlantDoc неоднородны · оси X логарифмические", x=0.01, ha="left")
    finish(fig, "dimensions")


def shooting_conditions(valid):
    cols = ["brightness", "saturation", "border_std", "sharpness"]
    titles = ["Средняя яркость [0, 1]", "Насыщенность [0, 1]", "Разброс яркости рамки [0, 1]", "Резкость: variance(Laplacian)"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), layout="constrained")
    for ax, col, title in zip(axes, cols, titles):
        for ds in NAMES:
            values = np.sort(valid.loc[valid.dataset.eq(ds), col].astype(float))
            ax.step(values, np.arange(1, len(values)+1)/len(values), where="post", color=COLORS[ds], linestyle="--" if ds=="plantvillage" else "-", label=NAMES[ds])
        ax.set_xlabel(title, fontsize=10)
        ax.set_ylim(0, 1.02)
        if col == "sharpness":
            ax.set_xscale("log")
            ax.set_xlabel(title+"\nлогарифмическая шкала", fontsize=10)
    axes[0].set_ylabel("Накопленная доля изображений")
    axes[0].legend(frameon=False)
    fig.suptitle("Условия съемки: все декодируемые исходные изображения", x=0.01, ha="left")
    finish(fig, "shooting_conditions")


def color_by_class(valid):
    for ds in NAMES:
        means = valid[valid.dataset.eq(ds)].groupby("label")[["mean_r", "mean_g", "mean_b", "border_brightness"]].mean()
        means.to_csv(ROOT / "reports" / f"colors_by_class_{ds}.csv")
        for start in range(0, len(means), 19):
            part = means.iloc[start:start+19]
            fig, axes = plt.subplots(1, 2, figsize=(12, max(4, len(part)*0.30)), gridspec_kw={"width_ratios": [1, 2]}, layout="constrained")
            axes[0].imshow(part.iloc[:, :3].to_numpy()[:, None, :], aspect="auto")
            axes[0].set_yticks(np.arange(len(part)), [label(x) for x in part.index], fontsize=10)
            axes[0].set_xticks([0], ["Средний RGB"])
            axes[1].scatter(part.border_brightness, np.arange(len(part)), c=COLORS[ds], s=24)
            axes[1].set_ylim(len(part)-.5, -.5)
            axes[1].set_yticks([])
            axes[1].set_xlim(0, 1)
            axes[1].set_xlabel("Средняя яркость рамки [0, 1]")
            fig.suptitle(f"{NAMES[ds]}: цвет и края кадра различаются по классам", x=0.01, ha="left")
            finish(fig, f"color_{ds}_{start//19+1}")


def paired_domains(frame):
    labels = ["Apple___healthy", "Potato___Early_blight", "Tomato___Late_blight", "Grape___Black_rot"]
    fig, axes = plt.subplots(4, 4, figsize=(12, 10), layout="constrained")
    for row, name in enumerate(labels):
        for j, ds in enumerate(NAMES):
            group = frame[frame.dataset.eq(ds) & frame.canonical_label.eq(name) & frame.error.eq("")]
            for k, (_, image) in enumerate(group.sample(n=min(2, len(group)), random_state=SEED).iterrows()):
                ax = axes[row, j*2+k]
                ax.imshow(thumbnail(image.path))
                ax.axis("off")
                ax.set_title(NAMES[ds] if row==0 else "", fontsize=11)
        axes[row, 0].set_ylabel(textwrap.fill(label(name), 23))
        axes[row, 0].text(0, -0.08, label(name), transform=axes[row, 0].transAxes, fontsize=11)
    fig.suptitle("Одинаковые метки, разные условия съемки", x=0.01, ha="left")
    finish(fig, "domain_examples", vector=False)


def pair_gallery(frame, pairs, kind="near"):
    if kind == "near":
        selected = pairs[pairs.remove_candidate & ~pairs.exact_pixels & pairs.same_label].sort_values(["distance", "rgb_mae"], ascending=False).head(6)
    else:
        selected = pairs[pairs.remove_candidate & ~pairs.same_label].sort_values(["distance", "rgb_mae"]).head(6)
    if selected.empty:
        print(f"No pairs: {kind}")
        return
    fig, axes = plt.subplots(len(selected), 2, figsize=(10, len(selected)*2.25), squeeze=False, layout="constrained")
    for row, pair in enumerate(selected.itertuples()):
        for col, idx in enumerate([pair.left, pair.right]):
            item = frame.loc[idx]
            axes[row, col].imshow(thumbnail(item.path, size=(340, 190)))
            axes[row, col].axis("off")
            axes[row, col].set_title(textwrap.fill(f"{item.dataset}: {label(item.label)}", 50), fontsize=10)
        axes[row, 0].text(0, -0.08, f"pHash={pair.distance}/64; RGB MAE={pair.rgb_mae:.3f}", transform=axes[row, 0].transAxes, fontsize=10)
    fig.suptitle("Пары для проверки: " + ("почти-дубликаты" if kind=="near" else "конфликтующие метки"), x=0.01, ha="left")
    finish(fig, f"duplicates_{kind}", vector=False)


def quality_gallery(valid):
    chosen = []
    for ds in NAMES:
        group = valid[valid.dataset.eq(ds)]
        for col, ascending, description in [("width", True, "малая ширина"), ("sharpness", True, "низкая резкость"), ("brightness", True, "темное"), ("brightness", False, "светлое")]:
            row = group.sort_values(col, ascending=ascending).iloc[0]
            chosen.append((row, description, col))
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), layout="constrained")
    for ax, (row, description, col) in zip(axes.flat, chosen):
        ax.imshow(thumbnail(row.path))
        ax.axis("off")
        ax.set_title(f"{NAMES[row.dataset]} · {description}\n{col}={float(row[col]):.4g}", fontsize=10)
    fig.suptitle("Крайние значения — кандидаты на проверку, а не автоматическое удаление", x=0.01, ha="left", fontsize=13)
    finish(fig, "quality_examples", vector=False)
    return [(row.image_id, desc) for row, desc, _ in chosen]


def embedding_plots(points):
    joint = points[points.joint_sample]
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    for ds in NAMES:
        g = joint[joint.dataset.eq(ds)]
        ax.scatter(g.joint_x, g.joint_y, c=COLORS[ds], marker="o" if ds=="plantvillage" else "^", s=11, alpha=0.65, label=f"{NAMES[ds]} (n={len(g)})")
    ax.legend(frameon=False)
    ax.set(xlabel="t-SNE 1 (условные единицы)", ylabel="t-SNE 2 (условные единицы)")
    ax.set_title("Общие классы с одинаковым числом примеров в каждом домене", loc="left")
    finish(fig, "tsne_domains")
    # One class highlighted per panel avoids a 38-color legend and remains readable in grayscale.
    for ds, prefix in [("plantvillage", "pv"), ("plantdoc", "pd")]:
        subset = points[points.dataset.eq(ds)]
        groups = list(subset.groupby("label", sort=True))
        for start in range(0, len(groups), 12):
            part = groups[start:start+12]
            fig, axes = plt.subplots(math.ceil(len(part)/3), 3, figsize=(12, math.ceil(len(part)/3)*2.5), squeeze=False, layout="constrained")
            for ax in axes.flat:
                ax.axis("off")
            for ax, (name, group) in zip(axes.flat, part):
                ax.scatter(subset[prefix+"_x"], subset[prefix+"_y"], s=3, c="#D2D5D2", alpha=0.5)
                ax.scatter(group[prefix+"_x"], group[prefix+"_y"], s=10, c="#1F6F5F", marker="^")
                ax.set_title(textwrap.fill(label(name), 34), fontsize=10, loc="left")
            fig.suptitle(f"{NAMES[ds]}: один класс выделен на каждом t-SNE · только train/dev", x=0.01, ha="left")
            finish(fig, f"tsne_{ds}_{start//12+1}")


def augmentation_gallery(frame):
    import torch
    from torchvision import transforms as T
    torch.manual_seed(SEED)
    selected = []
    for ds in NAMES:
        g = frame[frame.dataset.eq(ds) & frame.canonical_label.eq("Tomato___Early_blight") & frame.split.isin(["train", "field_dev"])]
        selected.append(g.sample(1, random_state=SEED).iloc[0])
    ops = [("Оригинал", lambda x:x), ("Отражение", ImageOps.mirror),
           ("Поворот 15°", lambda x:x.rotate(15, resample=Image.Resampling.BILINEAR, expand=True)),
           ("Мягкий ColorJitter", T.ColorJitter(brightness=.15, contrast=.15, saturation=.1, hue=.02)),
           ("Crop ≥ 85% площади", T.RandomResizedCrop((224, 224), scale=(.85, 1), ratio=(.9, 1.1))),
           ("Сильный hue — риск", T.ColorJitter(hue=.45))]
    fig, axes = plt.subplots(2, len(ops), figsize=(14, 5.3), layout="constrained")
    for i, row in enumerate(selected):
        with Image.open(ROOT / row.path) as src:
            image = ImageOps.exif_transpose(src).convert("RGB")
        for j, (title, op) in enumerate(ops):
            axes[i,j].imshow(op(image))
            axes[i,j].axis("off")
            axes[i,j].set_title(title if i==0 else "", fontsize=10)
            if j==0:
                axes[i,j].text(0, -.1, NAMES[row.dataset], transform=axes[i,j].transAxes, fontsize=11)
    fig.suptitle("Визуальная проверка аугментаций: симптомы должны сохраняться", x=.01, ha="left")
    finish(fig, "augmentations", vector=False)
