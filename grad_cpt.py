import argparse
import csv
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist, mean, pstdev
from typing import Dict, List, Sequence

import pandas as pd
from psychopy import core, data, event, gui, visual


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SEX_OPTIONS = ["男", "女"]
GRADE_OPTIONS = ["大一", "大二", "大三", "大四", "研一", "研二", "研三", "博一", "博二", "博三", "博四"]


@dataclass
class TrialSpec:
    trial_index: int
    condition: str  # city / mountain
    stimulus_path: Path


@dataclass
class ParticipantInfo:
    name: str
    participant_id: str
    sex: str = ""
    grade: str = ""


class ExperimentAborted(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPT attention task manager (PsychoPy).")
    parser.add_argument(
        "--mode",
        choices=["menu", "practice", "initial", "retest", "manager"],
        default="menu",
        help="Run mode.",
    )
    parser.add_argument("--name", default="", help="Participant name.")
    parser.add_argument(
        "--participant-id",
        "--participant",
        dest="participant_id",
        default="",
        help="Participant ID.",
    )
    parser.add_argument("--sex", choices=SEX_OPTIONS, default="", help="Sex (for initial test).")
    parser.add_argument("--grade", choices=GRADE_OPTIONS, default="", help="Grade (for initial test).")
    parser.add_argument(
        "--materials-root",
        default="",
        help="Material root containing 初测/后测 or pre/post folders.",
    )
    parser.add_argument("--data-dir", default="data", help="Output data directory.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--main-trials", type=int, default=497, help="Formal task trial count.")
    parser.add_argument("--practice-seconds", type=float, default=60.0, help="Practice duration in seconds.")
    parser.add_argument("--trial-duration", type=float, default=1.2, help="Single-trial duration in seconds.")
    parser.add_argument("--fade-duration", type=float, default=0.8, help="Cross-fade duration in seconds.")
    parser.add_argument("--go-ratio", type=float, default=0.9, help="Go ratio for city stimuli.")
    parser.add_argument("--response-key", default="space", help="Response key for city trials.")
    parser.add_argument("--windowed", action="store_true", help="Run in windowed mode.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build and validate trials, no visual experiment runtime.",
    )
    parser.add_argument(
        "--prepare-materials",
        action="store_true",
        help="Extract supplementary zip files and exit.",
    )
    return parser.parse_args()


def info_dialog(title: str, message: str) -> None:
    dlg = gui.Dlg(title=title)
    dlg.addText(message)
    dlg.show()


def error_dialog(message: str) -> None:
    info_dialog("错误", message)


def decode_zip_name(name: str) -> str:
    try:
        raw = name.encode("cp437")
    except UnicodeEncodeError:
        return name
    for enc in ("utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return name


def extract_zip_safe(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            decoded_name = decode_zip_name(info.filename)
            target = destination / decoded_name
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def prepare_materials(base_dir: Path) -> Path | None:
    root = base_dir / "supplementary_info_extracted" / "psychopy" / "cpt_material" / "cpt-材料"
    if root.exists():
        return root

    sup_zip = base_dir / "supplementary_info.zip"
    if not sup_zip.exists():
        return None

    extracted = base_dir / "supplementary_info_extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    extract_zip_safe(sup_zip, extracted)

    inner_zip_candidates = list(extracted.rglob("cpt-material.zip"))
    if not inner_zip_candidates:
        return None

    inner_zip = inner_zip_candidates[0]
    material_unpack = inner_zip.parent / "cpt_material"
    material_unpack.mkdir(parents=True, exist_ok=True)
    extract_zip_safe(inner_zip, material_unpack)

    root = material_unpack / "cpt-材料"
    if root.exists():
        return root
    return None


def resolve_material_root(base_dir: Path, materials_root_arg: str) -> Path:
    if materials_root_arg:
        candidate = Path(materials_root_arg)
        if not candidate.exists():
            raise FileNotFoundError(f"指定素材路径不存在: {candidate}")
        return candidate

    prepared = prepare_materials(base_dir)
    if prepared is not None:
        return prepared

    fallback = base_dir / "stimuli"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        "未找到素材目录。请确保 supplementary_info.zip 在项目根目录，或使用 --materials-root 指定路径。"
    )


def resolve_phase_folder_map(material_root: Path) -> Dict[str, str]:
    if (material_root / "初测").exists() and (material_root / "后测").exists():
        return {"initial": "初测", "retest": "后测"}
    if (material_root / "pre").exists() and (material_root / "post").exists():
        return {"initial": "pre", "retest": "post"}
    raise FileNotFoundError(
        f"素材目录结构不符合要求: {material_root}\n需包含 初测/后测 或 pre/post。"
    )


def load_stimuli(material_root: Path, phase_folder_name: str) -> Dict[str, List[Path]]:
    phase_dir = material_root / phase_folder_name
    if not phase_dir.exists():
        raise FileNotFoundError(f"缺少阶段素材目录: {phase_dir}")

    pools: Dict[str, List[Path]] = {}
    for condition in ("city", "mountain"):
        cond_dir = phase_dir / condition
        if not cond_dir.exists():
            raise FileNotFoundError(f"缺少类别目录: {cond_dir}")
        files = sorted(
            p for p in cond_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if len(files) == 0:
            raise FileNotFoundError(f"未找到图片文件: {cond_dir}")
        pools[condition] = files
    return pools


def build_condition_sequence(n_trials: int, go_ratio: float, rng: random.Random) -> List[str]:
    n_trials = max(1, n_trials)
    n_go = int(round(n_trials * go_ratio))
    if n_trials > 1:
        n_go = min(max(n_go, 1), n_trials - 1)
    else:
        n_go = 1
    n_nogo = n_trials - n_go
    sequence = ["city"] * n_go + ["mountain"] * n_nogo
    rng.shuffle(sequence)
    return sequence


def choose_stimulus(candidates: Sequence[Path], last_stimulus: Path | None, rng: random.Random) -> Path:
    if len(candidates) == 1:
        return candidates[0]
    if last_stimulus is None:
        return rng.choice(list(candidates))
    valid = [p for p in candidates if p != last_stimulus]
    if not valid:
        return rng.choice(list(candidates))
    return rng.choice(valid)


def build_trials(n_trials: int, pools: Dict[str, List[Path]], go_ratio: float, rng: random.Random) -> List[TrialSpec]:
    conditions = build_condition_sequence(n_trials, go_ratio, rng)
    trials: List[TrialSpec] = []
    last_stimulus: Path | None = None
    for i, condition in enumerate(conditions, start=1):
        stimulus = choose_stimulus(pools[condition], last_stimulus, rng)
        trials.append(TrialSpec(trial_index=i, condition=condition, stimulus_path=stimulus))
        last_stimulus = stimulus
    return trials


def show_text(win: visual.Window, message: str, wait_keys: Sequence[str] = ("space",)) -> None:
    text = visual.TextStim(
        win=win,
        text=message,
        color="white",
        height=0.04,
        wrapWidth=1.4,
        alignText="left",
    )
    text.draw()
    win.flip()
    keys = event.waitKeys(keyList=list(wait_keys) + ["escape"])
    if keys and keys[0] == "escape":
        raise ExperimentAborted("用户按下 ESC 终止实验")

def run_trial(
    win: visual.Window,
    img_prev: visual.ImageStim,
    img_curr: visual.ImageStim,
    trial: TrialSpec,
    prev_stimulus: Path | None,
    response_key: str,
    trial_duration: float,
    fade_duration: float,
) -> dict:
    event.clearEvents(eventType="keyboard")
    trial_clock = core.Clock()
    img_curr.image = str(trial.stimulus_path)
    responded = False
    rt = None
    response = ""

    while trial_clock.getTime() < trial_duration:
        t = trial_clock.getTime()
        if t <= fade_duration:
            alpha = max(0.0, min(1.0, t / fade_duration if fade_duration > 0 else 1.0))
            if prev_stimulus is not None:
                img_prev.image = str(prev_stimulus)
                img_prev.opacity = 1.0 - alpha
                img_prev.draw()
            img_curr.opacity = alpha
        else:
            img_curr.opacity = 1.0

        img_curr.draw()
        win.flip()

        keys = event.getKeys(keyList=[response_key, "escape"], timeStamped=trial_clock)
        for key_name, key_time in keys:
            if key_name == "escape":
                raise ExperimentAborted("用户按下 ESC 终止实验")
            if (not responded) and key_name == response_key:
                responded = True
                response = key_name
                rt = key_time

    correct = responded if trial.condition == "city" else (not responded)
    if trial.condition == "mountain" and responded:
        error_type = "commission"
    elif trial.condition == "city" and not responded:
        error_type = "omission"
    else:
        error_type = ""

    return {
        "trial_index": trial.trial_index,
        "condition": trial.condition,
        "stimulus_file": trial.stimulus_path.name,
        "response_key": response,
        "responded": int(responded),
        "rt_s": "" if rt is None else round(rt, 6),
        "correct": int(correct),
        "error_type": error_type,
        "trial_duration_s": trial_duration,
    }


def clamp_rate(rate: float, n: int) -> float:
    if n <= 0:
        return 0.5
    eps = 0.5 / n
    if rate <= 0:
        return eps
    if rate >= 1:
        return 1 - eps
    return rate


def compute_behavior_metrics(rows: Sequence[dict]) -> dict:
    if not rows:
        return {}

    city_rows = [r for r in rows if r["condition"] == "city"]
    mountain_rows = [r for r in rows if r["condition"] == "mountain"]

    total_city = len(city_rows)
    total_mountain = len(mountain_rows)
    city_hit = sum(1 for r in city_rows if r["responded"] == 1)
    mountain_false_alarm = sum(1 for r in mountain_rows if r["responded"] == 1)
    city_omission = sum(1 for r in city_rows if r["responded"] == 0)

    cer = (mountain_false_alarm / total_mountain) if total_mountain > 0 else 0.0
    oer = (city_omission / total_city) if total_city > 0 else 0.0

    correct_city_rts = [float(r["rt_s"]) for r in city_rows if r["correct"] == 1 and r["rt_s"] != ""]
    mean_rt = mean(correct_city_rts) if correct_city_rts else None
    rtcv = pstdev(correct_city_rts) if len(correct_city_rts) >= 2 else (0.0 if len(correct_city_rts) == 1 else None)

    hit_rate = (city_hit / total_city) if total_city > 0 else 0.0
    false_alarm_rate = (mountain_false_alarm / total_mountain) if total_mountain > 0 else 0.0
    hit_rate_adj = clamp_rate(hit_rate, total_city)
    false_alarm_rate_adj = clamp_rate(false_alarm_rate, total_mountain)
    nd = NormalDist()
    d_prime = nd.inv_cdf(hit_rate_adj) - nd.inv_cdf(false_alarm_rate_adj)

    accuracy = mean(r["correct"] for r in rows)

    return {
        "accuracy": round(accuracy, 6),
        "cer": round(cer, 6),
        "oer": round(oer, 6),
        "mean_rt": "" if mean_rt is None else round(mean_rt, 6),
        "rtcv": "" if rtcv is None else round(rtcv, 6),
        "d_prime": round(d_prime, 6),
        "city_trials": total_city,
        "mountain_trials": total_mountain,
    }


def save_rows_csv(rows: Sequence[dict], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with out_file.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["empty"])
        return

    with out_file.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_result_row(row: dict, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not file_path.exists()
    with file_path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def rebuild_total_results(initial_file: Path, retest_file: Path, total_file: Path) -> None:
    init_df = read_table(initial_file)
    retest_df = read_table(retest_file)

    if init_df.empty and retest_df.empty:
        total_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(total_file, index=False, encoding="utf-8-sig")
        return

    key_cols = ["编号", "姓名"]

    def latest_record(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        if "测试时间" in df.columns:
            df = df.copy()
            df["_time_sort"] = pd.to_datetime(df["测试时间"], errors="coerce")
            df = df.sort_values("_time_sort")
        return df.drop_duplicates(subset=key_cols, keep="last")

    init_latest = latest_record(init_df)
    retest_latest = latest_record(retest_df)

    init_keep = [
        "编号",
        "姓名",
        "性别",
        "年级",
        "测试时间",
        "虚报率CER",
        "漏报率OER",
        "平均反应时RT",
        "反应时变异RTCV",
        "辨别力d'",
    ]
    retest_keep = [
        "编号",
        "姓名",
        "测试时间",
        "虚报率CER",
        "漏报率OER",
        "平均反应时RT",
        "反应时变异RTCV",
        "辨别力d'",
    ]

    for col in init_keep:
        if col not in init_latest.columns:
            init_latest[col] = ""
    for col in retest_keep:
        if col not in retest_latest.columns:
            retest_latest[col] = ""

    init_view = init_latest[init_keep].rename(
        columns={
            "测试时间": "初测时间",
            "虚报率CER": "初测CER",
            "漏报率OER": "初测OER",
            "平均反应时RT": "初测RT",
            "反应时变异RTCV": "初测RTCV",
            "辨别力d'": "初测d'",
        }
    )
    retest_view = retest_latest[retest_keep].rename(
        columns={
            "测试时间": "复测时间",
            "虚报率CER": "复测CER",
            "漏报率OER": "复测OER",
            "平均反应时RT": "复测RT",
            "反应时变异RTCV": "复测RTCV",
            "辨别力d'": "复测d'",
        }
    )

    total_df = pd.merge(init_view, retest_view, on=key_cols, how="outer")
    ordered_cols = [
        "编号",
        "姓名",
        "性别",
        "年级",
        "初测时间",
        "初测CER",
        "初测OER",
        "初测RT",
        "初测RTCV",
        "初测d'",
        "复测时间",
        "复测CER",
        "复测OER",
        "复测RT",
        "复测RTCV",
        "复测d'",
    ]
    for col in ordered_cols:
        if col not in total_df.columns:
            total_df[col] = ""
    total_df = total_df[ordered_cols]
    total_file.parent.mkdir(parents=True, exist_ok=True)
    total_df.to_csv(total_file, index=False, encoding="utf-8-sig")


def preview_csv(title: str, file_path: Path) -> None:
    if not file_path.exists():
        error_dialog(f"{title} 文件不存在:\n{file_path}")
        return
    df = read_table(file_path)
    if df.empty:
        info_dialog(title, f"文件存在但暂无记录:\n{file_path}")
        return

    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception:
        print(f"\n===== {title} =====")
        print(df.tail(20).to_string(index=False))
        message = f"{title}\n记录数: {len(df)}\n文件: {file_path}\n\n已在控制台打印最近20条记录。"
        info_dialog(title, message)
        return

    root = tk.Tk()
    root.title(title)
    root.geometry("1200x700")

    top = ttk.Frame(root, padding=8)
    top.pack(fill=tk.X)
    ttk.Label(top, text=f"文件: {file_path}", anchor="w").pack(fill=tk.X)
    ttk.Label(top, text=f"记录数: {len(df)}", anchor="w").pack(fill=tk.X, pady=(4, 0))

    table_frame = ttk.Frame(root, padding=(8, 0, 8, 8))
    table_frame.pack(fill=tk.BOTH, expand=True)

    cols = list(df.columns)
    tree = ttk.Treeview(table_frame, columns=cols, show="headings")
    x_scroll = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=tree.xview)
    y_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)

    tree.grid(row=0, column=0, sticky="nsew")
    y_scroll.grid(row=0, column=1, sticky="ns")
    x_scroll.grid(row=1, column=0, sticky="ew")
    table_frame.columnconfigure(0, weight=1)
    table_frame.rowconfigure(0, weight=1)

    max_sample_rows = min(len(df), 50)
    for col in cols:
        tree.heading(col, text=col)
        sample_values = [str(v) for v in df[col].iloc[:max_sample_rows].tolist()]
        max_len = max([len(col)] + [len(v) for v in sample_values])
        width = min(max(80, max_len * 12), 320)
        tree.column(col, width=width, stretch=True, anchor="center")

    for row in df.itertuples(index=False):
        values = [str(v) for v in row]
        tree.insert("", tk.END, values=values)

    bottom = ttk.Frame(root, padding=(8, 0, 8, 8))
    bottom.pack(fill=tk.X)
    ttk.Button(bottom, text="关闭", command=root.destroy).pack(side=tk.RIGHT)

    root.mainloop()


def export_excel(results_dir: Path, raw_dir: Path) -> Path:
    initial_file = results_dir / "初测结果.csv"
    retest_file = results_dir / "复测结果.csv"
    total_file = results_dir / "总结果.csv"

    timestamp = data.getDateStr(format="%Y%m%d_%H%M%S")
    out_file = results_dir / f"CPT数据导出_{timestamp}.xlsx"

    init_df = read_table(initial_file)
    retest_df = read_table(retest_file)
    total_df = read_table(total_file)

    raw_files = sorted(raw_dir.glob("*.csv"))
    raw_df = pd.concat([read_table(p) for p in raw_files], ignore_index=True) if raw_files else pd.DataFrame()

    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        init_df.to_excel(writer, sheet_name="初测结果", index=False)
        retest_df.to_excel(writer, sheet_name="复测结果", index=False)
        total_df.to_excel(writer, sheet_name="总结果", index=False)
        raw_df.to_excel(writer, sheet_name="原始试次", index=False)

    return out_file


def find_initial_record(initial_file: Path, participant_id: str, name: str) -> dict | None:
    if not initial_file.exists():
        return None
    df = read_table(initial_file)
    if df.empty:
        return None
    pid = str(participant_id).strip()
    pname = str(name).strip()
    matched = df[
        (df["编号"].astype(str).str.strip() == pid)
        & (df["姓名"].astype(str).str.strip() == pname)
    ]
    if matched.empty:
        return None
    rec = matched.iloc[-1].to_dict()
    return {k: ("" if pd.isna(v) else v) for k, v in rec.items()}


def prompt_initial_info(default_name: str = "", default_pid: str = "") -> ParticipantInfo | None:
    dlg = gui.Dlg(title="初测信息录入")
    dlg.addField("姓名:", initial=default_name)
    dlg.addField("性别:", choices=SEX_OPTIONS)
    dlg.addField("年级:", choices=GRADE_OPTIONS)
    dlg.addField("编号:", initial=default_pid)
    values = dlg.show()
    if values is None:
        return None

    name = str(values[0]).strip()
    sex = str(values[1]).strip()
    grade = str(values[2]).strip()
    participant_id = str(values[3]).strip()
    if not name or not participant_id:
        error_dialog("姓名和编号不能为空。")
        return None
    return ParticipantInfo(name=name, participant_id=participant_id, sex=sex, grade=grade)


def prompt_retest_info(default_name: str = "", default_pid: str = "") -> ParticipantInfo | None:
    dlg = gui.Dlg(title="复测信息录入")
    dlg.addField("姓名:", initial=default_name)
    dlg.addField("编号:", initial=default_pid)
    values = dlg.show()
    if values is None:
        return None

    name = str(values[0]).strip()
    participant_id = str(values[1]).strip()
    if not name or not participant_id:
        error_dialog("姓名和编号不能为空。")
        return None
    return ParticipantInfo(name=name, participant_id=participant_id)


def run_phase(
    phase_key: str,  # practice / initial / retest
    participant: ParticipantInfo,
    pools: Dict[str, List[Path]],
    args: argparse.Namespace,
    rng: random.Random,
    raw_dir: Path,
) -> tuple[Path, dict, bool]:
    if args.fade_duration > args.trial_duration:
        raise ValueError("--fade-duration must be <= --trial-duration")

    if phase_key == "practice":
        n_trials = max(1, int(args.practice_seconds / args.trial_duration))
        phase_title = "练习模式"
        intro = (
            "练习模式（1分钟）\n\n"
            "规则：\n"
            f"1. 城市图片按 {args.response_key}\n"
            "2. 山地图片不按键\n\n"
            "按空格开始。"
        )
    elif phase_key == "initial":
        n_trials = args.main_trials
        phase_title = "初测"
        intro = (
            "初测正式任务\n\n"
            "规则：\n"
            f"1. 城市图片按 {args.response_key}\n"
            "2. 山地图片不按键\n"
            "3. 请尽量又快又准\n\n"
            "按空格开始。"
        )
    else:
        n_trials = args.main_trials
        phase_title = "复测"
        intro = (
            "复测正式任务\n\n"
            "规则与初测一致：\n"
            f"1. 城市图片按 {args.response_key}\n"
            "2. 山地图片不按键\n"
            "3. 请尽量又快又准\n\n"
            "按空格开始。"
        )

    trials = build_trials(n_trials, pools, args.go_ratio, rng)
    if args.dry_run:
        print(f"[dry-run] {phase_title}")
        print(f"city={len(pools['city'])}, mountain={len(pools['mountain'])}, trials={len(trials)}")
        dummy_raw = raw_dir / f"DRYRUN_{phase_key}.csv"
        return dummy_raw, {}, True

    win = visual.Window(
        size=[1280, 720],
        fullscr=not args.windowed,
        color="black",
        units="height",
    )
    img_prev = visual.ImageStim(
        win=win,
        image=str(pools["city"][0]),
        size=(0.8, 0.8),
        units="height",
        mask="circle",
        interpolate=True,
    )
    img_curr = visual.ImageStim(
        win=win,
        image=str(pools["city"][0]),
        size=(0.8, 0.8),
        units="height",
        mask="circle",
        interpolate=True,
    )

    rows: List[dict] = []
    previous: Path | None = None
    completed = True
    start_time = data.getDateStr(format="%Y-%m-%d %H:%M:%S")
    try:
        show_text(win, intro)
        for trial in trials:
            row = run_trial(
                win=win,
                img_prev=img_prev,
                img_curr=img_curr,
                trial=trial,
                prev_stimulus=previous,
                response_key=args.response_key,
                trial_duration=args.trial_duration,
                fade_duration=args.fade_duration,
            )
            row.update(
                {
                    "phase": phase_key,
                    "name": participant.name,
                    "participant_id": participant.participant_id,
                    "sex": participant.sex,
                    "grade": participant.grade,
                    "start_time": start_time,
                }
            )
            rows.append(row)
            previous = trial.stimulus_path
    except ExperimentAborted:
        completed = False
    finally:
        metrics = compute_behavior_metrics(rows) if rows else {}
        if completed:
            end_msg = f"{phase_title}结束。\n\n按空格退出。"
            if metrics:
                end_msg = (
                    f"{phase_title}结束。\n\n"
                    f"CER: {metrics.get('cer', '')}\n"
                    f"OER: {metrics.get('oer', '')}\n"
                    f"RT: {metrics.get('mean_rt', '')}\n"
                    f"RTCV: {metrics.get('rtcv', '')}\n"
                    f"d': {metrics.get('d_prime', '')}\n\n"
                    "按空格退出。"
                )
            try:
                show_text(win, end_msg)
            except ExperimentAborted:
                pass
        win.close()
        core.wait(0.1)

    ts = data.getDateStr(format="%Y%m%d_%H%M%S")
    raw_file = raw_dir / f"{participant.participant_id}_{phase_key}_{ts}.csv"
    save_rows_csv(rows, raw_file)
    return raw_file, metrics, completed


def run_initial(
    participant: ParticipantInfo,
    pools: Dict[str, List[Path]],
    args: argparse.Namespace,
    rng: random.Random,
    raw_dir: Path,
    initial_file: Path,
    retest_file: Path,
    total_file: Path,
) -> None:
    raw_file, metrics, completed = run_phase(
        phase_key="initial",
        participant=participant,
        pools=pools,
        args=args,
        rng=rng,
        raw_dir=raw_dir,
    )
    if args.dry_run:
        return

    row = {
        "测试时间": data.getDateStr(format="%Y-%m-%d %H:%M:%S"),
        "阶段": "初测",
        "编号": participant.participant_id,
        "姓名": participant.name,
        "性别": participant.sex,
        "年级": participant.grade,
        "试次数": args.main_trials,
        "正确率": metrics.get("accuracy", ""),
        "虚报率CER": metrics.get("cer", ""),
        "漏报率OER": metrics.get("oer", ""),
        "平均反应时RT": metrics.get("mean_rt", ""),
        "反应时变异RTCV": metrics.get("rtcv", ""),
        "辨别力d'": metrics.get("d_prime", ""),
        "完成状态": "完成" if completed else "中断",
        "原始数据文件": str(raw_file),
    }
    append_result_row(row, initial_file)
    rebuild_total_results(initial_file, retest_file, total_file)
    info_dialog("初测完成", f"初测结果已保存。\n原始数据: {raw_file}\n结果文件: {initial_file}")


def run_retest(
    participant: ParticipantInfo,
    pools: Dict[str, List[Path]],
    args: argparse.Namespace,
    rng: random.Random,
    raw_dir: Path,
    initial_file: Path,
    retest_file: Path,
    total_file: Path,
) -> None:
    if not initial_file.exists():
        msg = "未检测到初测结果文件，请先完成一次初测后再进行复测。"
        if args.dry_run:
            print(msg)
        else:
            error_dialog(msg)
        return

    matched = find_initial_record(initial_file, participant.participant_id, participant.name)
    if matched is None:
        msg = (
            "复测失败：姓名和编号与初测记录不一致，无法进入复测。\n"
            "请确认与初测输入完全一致（包含前后空格与编号格式）。"
        )
        if args.dry_run:
            print(msg)
        else:
            error_dialog(msg)
        return

    participant.sex = str(matched.get("性别", ""))
    participant.grade = str(matched.get("年级", ""))
    raw_file, metrics, completed = run_phase(
        phase_key="retest",
        participant=participant,
        pools=pools,
        args=args,
        rng=rng,
        raw_dir=raw_dir,
    )
    if args.dry_run:
        return

    row = {
        "测试时间": data.getDateStr(format="%Y-%m-%d %H:%M:%S"),
        "阶段": "复测",
        "编号": participant.participant_id,
        "姓名": participant.name,
        "性别": participant.sex,
        "年级": participant.grade,
        "试次数": args.main_trials,
        "正确率": metrics.get("accuracy", ""),
        "虚报率CER": metrics.get("cer", ""),
        "漏报率OER": metrics.get("oer", ""),
        "平均反应时RT": metrics.get("mean_rt", ""),
        "反应时变异RTCV": metrics.get("rtcv", ""),
        "辨别力d'": metrics.get("d_prime", ""),
        "完成状态": "完成" if completed else "中断",
        "原始数据文件": str(raw_file),
    }
    append_result_row(row, retest_file)
    rebuild_total_results(initial_file, retest_file, total_file)
    info_dialog("复测完成", f"复测结果已保存。\n原始数据: {raw_file}\n结果文件: {retest_file}")

def run_practice(
    participant: ParticipantInfo,
    pools: Dict[str, List[Path]],
    args: argparse.Namespace,
    rng: random.Random,
    raw_dir: Path,
) -> None:
    raw_file, _, _ = run_phase(
        phase_key="practice",
        participant=participant,
        pools=pools,
        args=args,
        rng=rng,
        raw_dir=raw_dir,
    )
    if not args.dry_run:
        info_dialog("练习结束", f"练习原始数据已保存:\n{raw_file}")


def choose_main_menu() -> str | None:
    dlg = gui.Dlg(title="CPT程序入口")
    dlg.addField("选择模式:", choices=["练习模式", "正式测试", "管理者中心", "退出"])
    values = dlg.show()
    if values is None:
        return None
    return str(values[0])


def choose_formal_menu() -> str | None:
    dlg = gui.Dlg(title="正式测试")
    dlg.addField("选择阶段:", choices=["初测", "复测", "返回"])
    values = dlg.show()
    if values is None:
        return None
    return str(values[0])


def manager_center(results_dir: Path, raw_dir: Path) -> None:
    initial_file = results_dir / "初测结果.csv"
    retest_file = results_dir / "复测结果.csv"
    total_file = results_dir / "总结果.csv"

    while True:
        dlg = gui.Dlg(title="管理者中心")
        dlg.addField("功能:", choices=["初测结果", "复测结果", "总结果", "下载数据(Excel)", "返回"])
        values = dlg.show()
        if values is None:
            return
        action = str(values[0])

        if action == "返回":
            return
        if action == "初测结果":
            preview_csv("初测结果", initial_file)
        elif action == "复测结果":
            preview_csv("复测结果", retest_file)
        elif action == "总结果":
            preview_csv("总结果", total_file)
        elif action == "下载数据(Excel)":
            out = export_excel(results_dir, raw_dir)
            info_dialog("导出完成", f"Excel 已导出:\n{out}")


def ensure_dirs(data_dir: Path) -> tuple[Path, Path, Path]:
    raw_dir = data_dir / "raw_trials"
    results_dir = data_dir / "results"
    raw_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    total_file = results_dir / "总结果.csv"
    return raw_dir, results_dir, total_file


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    if args.prepare_materials:
        root = prepare_materials(base_dir)
        if root is None:
            raise FileNotFoundError("未找到 supplementary_info.zip，无法自动解压。")
        print(f"materials prepared at: {root}")
        return

    material_root = resolve_material_root(base_dir, args.materials_root)
    phase_map = resolve_phase_folder_map(material_root)

    rng = random.Random(args.seed)
    data_dir = Path(args.data_dir)
    raw_dir, results_dir, total_file = ensure_dirs(data_dir)
    initial_file = results_dir / "初测结果.csv"
    retest_file = results_dir / "复测结果.csv"

    initial_pools = load_stimuli(material_root, phase_map["initial"])
    retest_pools = load_stimuli(material_root, phase_map["retest"])

    if args.mode == "practice":
        participant = ParticipantInfo(
            name=args.name or "练习者",
            participant_id=args.participant_id or "PRACTICE",
        )
        run_practice(participant, initial_pools, args, rng, raw_dir)
        return

    if args.mode == "initial":
        participant = ParticipantInfo(
            name=args.name,
            participant_id=args.participant_id,
            sex=args.sex,
            grade=args.grade,
        )
        if not participant.name or not participant.participant_id:
            error_dialog("initial 模式需要提供 --name 和 --participant-id。")
            return
        if not participant.sex or not participant.grade:
            error_dialog("initial 模式需要提供 --sex 和 --grade。")
            return
        run_initial(
            participant=participant,
            pools=initial_pools,
            args=args,
            rng=rng,
            raw_dir=raw_dir,
            initial_file=initial_file,
            retest_file=retest_file,
            total_file=total_file,
        )
        return

    if args.mode == "retest":
        participant = ParticipantInfo(name=args.name, participant_id=args.participant_id)
        if not participant.name or not participant.participant_id:
            error_dialog("retest 模式需要提供 --name 和 --participant-id。")
            return
        run_retest(
            participant=participant,
            pools=retest_pools,
            args=args,
            rng=rng,
            raw_dir=raw_dir,
            initial_file=initial_file,
            retest_file=retest_file,
            total_file=total_file,
        )
        return

    if args.mode == "manager":
        manager_center(results_dir, raw_dir)
        return

    while True:
        action = choose_main_menu()
        if action is None or action == "退出":
            break

        if action == "练习模式":
            participant = ParticipantInfo(name="练习者", participant_id="PRACTICE")
            run_practice(participant, initial_pools, args, rng, raw_dir)
            continue

        if action == "正式测试":
            sub = choose_formal_menu()
            if sub is None or sub == "返回":
                continue

            if sub == "初测":
                participant = prompt_initial_info(default_name=args.name, default_pid=args.participant_id)
                if participant is None:
                    continue
                run_initial(
                    participant=participant,
                    pools=initial_pools,
                    args=args,
                    rng=rng,
                    raw_dir=raw_dir,
                    initial_file=initial_file,
                    retest_file=retest_file,
                    total_file=total_file,
                )
            elif sub == "复测":
                participant = prompt_retest_info(default_name=args.name, default_pid=args.participant_id)
                if participant is None:
                    continue
                run_retest(
                    participant=participant,
                    pools=retest_pools,
                    args=args,
                    rng=rng,
                    raw_dir=raw_dir,
                    initial_file=initial_file,
                    retest_file=retest_file,
                    total_file=total_file,
                )
            continue

        if action == "管理者中心":
            manager_center(results_dir, raw_dir)


if __name__ == "__main__":
    main()
