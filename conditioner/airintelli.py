import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ── 한글 폰트 ──────────────────────────────────────────────────────────────────
_KOREAN_FONTS = ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'Noto Sans CJK KR']
_installed    = {f.name for f in font_manager.fontManager.ttflist}
for _f in _KOREAN_FONTS:
    if _f in _installed:
        plt.rcParams['font.family'] = _f
        break
plt.rcParams['axes.unicode_minus'] = False

# ── 상수 ───────────────────────────────────────────────────────────────────────
ZONE_COUNT  = 13
TEMP_LIMIT  = 30
HUMID_LIMIT = 90
DAY_KO      = ['월', '화', '수', '목', '금', '토', '일']
DAY_EN      = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

OPTIMAL_JUNE    = [22.0, 20.3, 21.0, 19.5, 23.0, 22.0, 20.0, 24.5, 21.5, 20.3, 23.5, 17.5, 23.0]
OPTIMAL_JUL_AUG = [23.0, 20.0, 20.0, 24.5, 22.0, 22.0, 23.3, 20.0, 21.5, 26.0, 17.5, 19.5, 23.0]

# 센서 OFF 구간: 23:00 ~ 06:29 (분 단위로 표현하면 1380~1439, 0~389)
SENSOR_OFF_START = 23 * 60       # 1380
SENSOR_OFF_END   = 6 * 60 + 30  #  390  (06:30부터 가동)

def _is_sensor_off(hour, minute):
    """해당 시각이 센서 OFF 구간(23:00~06:29)이면 True"""
    total = hour * 60 + minute
    return total >= SENSOR_OFF_START or total < SENSOR_OFF_END

# ── ANSI 색상 헬퍼 ─────────────────────────────────────────────────────────────
def _ansi(hex_color, text):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"

def _level(diff):
    abs_d = abs(diff)
    if abs_d < 2:
        return 2, '#55bb44'
    elif abs_d < 5:
        return (1, '#39c5bb') if diff < 0 else (3, '#ffb4cc')
    else:
        return (0, '#3355bb') if diff < 0 else (4, '#ff0045')

def _variance_level(var):
    if var < 0.2:
        return 0, '#3355bb'
    elif var < 0.5:
        return 1, '#39c5bb'
    elif var < 0.8:
        return 2, '#55bb44'
    elif var < 1.1:
        return 3, '#ffb4cc'
    else:
        return 4, '#ff0045'

_BAR_COLORS = {
    'load':  '#4fc3f7',
    'train': '#ff7e00',
}

# ── 전역 저장소 ────────────────────────────────────────────────────────────────
weekday_models = {}
weekend_models = {}
datasets       = {}
optimal_temps  = None


# ── 기간 선택 ──────────────────────────────────────────────────────────────────
def select_period():
    global optimal_temps
    print("\n자리배치 기간을 선택하세요.")
    print("  1: 6월")
    print("  2: 7~8월")
    while True:
        choice = input("선택 (1 또는 2): ").strip()
        if choice == '1':
            optimal_temps = OPTIMAL_JUNE
            print("6월 적정온도 기준으로 설정되었습니다.")
            break
        elif choice == '2':
            optimal_temps = OPTIMAL_JUL_AUG
            print("7~8월 적정온도 기준으로 설정되었습니다.")
            break
        else:
            print("1 또는 2를 입력해주세요.")


# ── 데이터 로드 & 학습 ────────────────────────────────────────────────────────
def load_and_train():
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(script_dir, "data_prep")

    print("\n[1/3] 파일 크기 확인 중...")
    total_cells = 0
    zone_cells  = {}
    for i in range(ZONE_COUNT):
        fp = os.path.join(data_folder, f"{i}.csv")
        if os.path.exists(fp):
            n = sum(1 for _ in open(fp, encoding='utf-8-sig')) - 1
            zone_cells[i] = n * 3
            total_cells  += zone_cells[i]

    print(f"\n[2/3] 데이터 로드 중... (총 {total_cells:,} 셀)")
    bar_load = tqdm(
        total=total_cells,
        unit='셀',
        unit_scale=True,
        bar_format='{l_bar}{bar}| {n:,.0f}/{total:,.0f} [{elapsed}<{remaining}]',
        colour=_BAR_COLORS['load'],
        file=sys.stdout,
    )
    raw_data = {}
    for i in range(ZONE_COUNT):
        fp = os.path.join(data_folder, f"{i}.csv")
        if not os.path.exists(fp):
            print(f"\n경고: {fp} 파일을 찾을 수 없습니다.")
            continue
        df = pd.read_csv(fp, encoding='utf-8-sig')
        df = df.iloc[:, [0, 1, 3]]
        df.columns = ['datetime', '온도', '습도']
        raw_data[i] = df

        cells = zone_cells.get(i, len(df) * 3)
        bar_load.update(cells)
        bar_load.set_description(f"구역 {i:>2} 로드완료")
    bar_load.close()

    print(f"\n[3/3] 전처리 및 모델 학습 중... (구역당 평일·주말 2개 모델)")
    bar_train = tqdm(
        total=ZONE_COUNT * 2,
        unit='모델',
        bar_format='{l_bar}{bar}| {n}/{total} [{elapsed}<{remaining}]',
        colour=_BAR_COLORS['train'],
        file=sys.stdout,
    )
    for i, df in raw_data.items():
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime'])
        df['요일'] = df['datetime'].dt.weekday
        df['시간'] = df['datetime'].dt.hour
        df['분']   = df['datetime'].dt.minute
        df['날짜'] = df['datetime'].dt.date
        df = df[(df['온도'] < TEMP_LIMIT) & (df['습도'] < HUMID_LIMIT)]
        datasets[i] = df.reset_index(drop=True)

        for is_weekend, mdict in [(False, weekday_models), (True, weekend_models)]:
            tag    = '주말' if is_weekend else '평일'
            subset = df[df['요일'] >= 5] if is_weekend else df[df['요일'] <= 4]
            bar_train.set_description(f"구역 {i:>2} {tag} 학습")
            if not subset.empty:
                m = RandomForestRegressor(n_estimators=100, random_state=42)
                m.fit(subset[['요일', '시간', '분']], subset['온도'])
                mdict[i] = m
            bar_train.update(1)
    bar_train.close()
    print("\n학습 완료")


# ── 온도 예측 출력 ─────────────────────────────────────────────────────────────
def _shift(h, m, delta):
    total = (h * 60 + m + delta) % 1440
    return total // 60, total % 60

def get_expected_temperature(yoil, hour, minut):
    try:
        yoil, hour, minut = int(yoil), int(hour), int(minut)
    except ValueError:
        print("요일/시간/분은 숫자로 입력해주세요.")
        return
    if not (0 <= yoil <= 6):
        print("요일은 0~6 사이의 값으로 입력해주세요.")
        return
    if not (0 <= hour <= 23):
        print("시간은 0~23 사이의 값으로 입력해주세요.")
        return
    if not (0 <= minut <= 59):
        print("분은 0~59 사이의 값으로 입력해주세요.")
        return

    # 센서 OFF 구간 체크
    if _is_sensor_off(hour, minut):
        print(f"\n  [{hour:02d}:{minut:02d}]은 센서 OFF 구간(23:00~06:29)입니다. 예측이 불가능합니다.")
        return

    is_weekend = yoil > 4
    model_dict = weekend_models if is_weekend else weekday_models
    tag        = "[주말]" if is_weekend else "[평일]"

    hour1,  min1  = _shift(hour, minut,  1)
    hour10, min10 = _shift(hour, minut, 10)

    # +1분 / +10분이 센서 OFF 구간에 걸치면 변화량 표시 불가 플래그
    off_1m  = _is_sensor_off(hour1,  min1)
    off_10m = _is_sensor_off(hour10, min10)

    input_now = pd.DataFrame({'요일': [yoil], '시간': [hour],   '분': [minut]})
    input_1m  = pd.DataFrame({'요일': [yoil], '시간': [hour1],  '분': [min1]})
    input_10m = pd.DataFrame({'요일': [yoil], '시간': [hour10], '분': [min10]})

    print(f"\n=== {tag} {DAY_KO[yoil]}요일 {hour:02d}시 {minut:02d}분 — 구역별 예상 온도 ===")
    print(f"    (현재 | +1분({hour1:02d}:{min1:02d}) | +10분({hour10:02d}:{min10:02d}) 변화량 기준)")

    preds = []
    for i in range(ZONE_COUNT):
        opt = optimal_temps[i]
        if i not in model_dict:
            print(f"  구역 {i:>2}  모델이 준비되지 않았습니다.")
            continue

        pred = model_dict[i].predict(input_now)[0]
        preds.append(pred)

        diff  = pred - opt
        sign  = '+' if diff >= 0 else ''
        lv, color = _level(diff)
        colored   = _ansi(color, f"차이: {sign}{diff:.2f}°C  Lv.{lv}")

        if off_1m:
            chg1 = "N/A(센서OFF)"
        else:
            d1  = model_dict[i].predict(input_1m)[0] - pred
            s1  = '+' if d1 >= 0 else ''
            chg1 = f"{s1}{d1:.2f}°C"

        if off_10m:
            chg10 = "N/A(센서OFF)"
        else:
            d10  = model_dict[i].predict(input_10m)[0] - pred
            s10  = '+' if d10 >= 0 else ''
            chg10 = f"{s10}{d10:.2f}°C"

        print(f"  구역 {i:>2}  예상: {pred:.2f}°C  적정: {opt}°C  {colored}  │ +1분: {chg1}  +10분: {chg10}")

    if len(preds) >= 2:
        var = float(np.var(preds))
        vlv, vcolor = _variance_level(var)
        colored = _ansi(vcolor, f"전체 구역 예상 온도 분산: {var:.3f}  Lv.{vlv}")
        print(f"\n  {colored}")


# ── 그래프 공통 헬퍼 ───────────────────────────────────────────────────────────
def _make_pred_df(zone, day_list, hour_range=None):
    """센서 ON 구간(06:30~22:59)의 분만 예측. OFF 구간은 NaN으로 채워 선을 끊는다."""
    import datetime as _dt
    base = pd.Timestamp(_dt.date(2000, 1, 1))
    rows = []
    offset = 0
    for yoil in day_list:
        is_weekend = yoil > 4
        mdict = weekend_models if is_weekend else weekday_models
        if zone not in mdict:
            continue
        if hour_range:
            sh, eh = hour_range
            minute_list = [h * 60 + m for h in range(sh, eh + 1) for m in range(60)]
        else:
            minute_list = list(range(1440))
        hours = [m // 60 for m in minute_list]
        mins  = [m % 60  for m in minute_list]

        # 센서 ON인 분만 배치 예측, OFF 구간은 NaN
        on_mask  = [not _is_sensor_off(h, mn) for h, mn in zip(hours, mins)]
        on_idx   = [j for j, v in enumerate(on_mask) if v]
        preds_all = [float('nan')] * len(minute_list)
        if on_idx:
            inp = pd.DataFrame({
                '요일': yoil,
                '시간': [hours[j] for j in on_idx],
                '분':   [mins[j]  for j in on_idx],
            })
            results = mdict[zone].predict(inp)
            for k, j in enumerate(on_idx):
                preds_all[j] = results[k]

        for j, (h, mn, p) in enumerate(zip(hours, mins, preds_all)):
            rows.append({
                'yoil':     yoil,
                'offset':   offset,
                'datetime': base + pd.Timedelta(minutes=minute_list[j]),
                '온도':      p,        # NaN이면 그래프에서 선이 끊김
                '시간':      h,
                '분':        mn,
                'sensor_on': on_mask[j],
            })
            offset += 1
    return pd.DataFrame(rows)


def _build_ticks(pred_df, tick_min):
    tick_pos, tick_lbl = [], []
    for _, row in pred_df.iterrows():
        ts        = row['datetime']
        total_min = ts.hour * 60 + ts.minute
        if total_min % tick_min == 0:
            tick_pos.append(int(row['offset']))
            if tick_min >= 60:
                tick_lbl.append(f"{ts.hour:02d}:00")
            else:
                tick_lbl.append(f"{ts.hour:02d}:{ts.minute:02d}")
    return tick_pos, tick_lbl


def _add_day_separators(ax, pred_df, day_list, total):
    ylim_top = ax.get_ylim()[1]
    for yoil in day_list:
        grp = pred_df[pred_df['yoil'] == yoil]
        if grp.empty:
            continue
        pos = int(grp['offset'].iloc[0])
        if yoil != day_list[0]:
            ax.axvline(x=pos, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.text(pos + total * 0.003, ylim_top,
                DAY_EN[yoil], fontsize=7, color='gray', va='top', clip_on=True)


def _shade_sensor_off(ax, pred_df):
    """센서 OFF 구간을 회색 음영으로 표시"""
    in_off  = False
    off_start = None
    for _, row in pred_df.iterrows():
        if not row['sensor_on'] and not in_off:
            in_off    = True
            off_start = row['offset']
        elif row['sensor_on'] and in_off:
            ax.axvspan(off_start, row['offset'], color='gray', alpha=0.15, label='_nolegend_')
            in_off = False
    if in_off:
        ax.axvspan(off_start, pred_df['offset'].iloc[-1], color='gray', alpha=0.15, label='_nolegend_')
    # 범례용 패치 한 개만
    from matplotlib.patches import Patch
    return Patch(facecolor='gray', alpha=0.3, label='Sensor OFF (23:00–06:29)')


def _draw_graph(zone, pred_df, day_list, tick_min, title, xlabel, multi_day=False):
    total = len(pred_df)
    tick_pos, tick_lbl = _build_ticks(pred_df, tick_min)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(pred_df['offset'], pred_df['온도'],
            label='Predicted Temp (°C)', color='red', linewidth=0.9)
    ax.axhline(y=optimal_temps[zone], color='orange', linestyle=':', linewidth=1.2,
               label=f'Optimal {optimal_temps[zone]}°C')

    off_patch = _shade_sensor_off(ax, pred_df)

    if multi_day:
        _add_day_separators(ax, pred_df, day_list, total)

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, fontsize=7, rotation=45)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [off_patch], labels + [off_patch.get_label()], loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ── 그래프 ─────────────────────────────────────────────────────────────────────
def plot_graph(zone, mode):
    if zone not in weekday_models and zone not in weekend_models:
        print(f"No model for zone {zone}.")
        return

    if mode == 'g+1':
        print("\nSelect a day of the week.")
        for k, name in enumerate(DAY_EN):
            print(f"  {k}: {name}")
        sel = input("Enter day number (0=Mon ~ 6=Sun): ").strip()
        try:
            sel_yoil = int(sel)
            if not (0 <= sel_yoil <= 6):
                raise ValueError
        except ValueError:
            print("Invalid input.")
            return

        pred_df = _make_pred_df(zone, [sel_yoil], hour_range=(13, 15))
        if pred_df.empty:
            print(f"Model for zone {zone} is not ready.")
            return
        # 13~15시는 센서 ON 구간이므로 음영 없음 (hour_range 한정)
        _draw_graph(zone, pred_df, [sel_yoil], tick_min=10,
                    title=f"Zone {zone} — {DAY_EN[sel_yoil]}  Predicted Temp (13:00–15:59)",
                    xlabel="Time (13:00–15:59, 10-min intervals)")

    elif mode == 'g+2':
        print("\nSelect a day of the week.")
        for k, name in enumerate(DAY_EN):
            print(f"  {k}: {name}")
        sel = input("Enter day number (0=Mon ~ 6=Sun): ").strip()
        try:
            sel_yoil = int(sel)
            if not (0 <= sel_yoil <= 6):
                raise ValueError
        except ValueError:
            print("Invalid input.")
            return

        pred_df = _make_pred_df(zone, [sel_yoil])
        if pred_df.empty:
            print(f"Model for zone {zone} is not ready.")
            return
        _draw_graph(zone, pred_df, [sel_yoil], tick_min=30,
                    title=f"Zone {zone} — {DAY_EN[sel_yoil]}  Predicted Temp (00:00–23:59)",
                    xlabel="Time (30-min intervals)")

    elif mode == 'g+3':
        day_list = [0, 1, 2]
        pred_df  = _make_pred_df(zone, day_list)
        if pred_df.empty:
            print(f"Model for zone {zone} is not ready.")
            return
        _draw_graph(zone, pred_df, day_list, tick_min=120,
                    title=f"Zone {zone} — Mon / Tue / Wed  Predicted Temp",
                    xlabel="Day / Time (2-hour intervals)", multi_day=True)

    elif mode == 'g+4':
        day_list = [0, 1, 2, 3, 4]
        pred_df  = _make_pred_df(zone, day_list)
        if pred_df.empty:
            print(f"Model for zone {zone} is not ready.")
            return
        _draw_graph(zone, pred_df, day_list, tick_min=360,
                    title=f"Zone {zone} — Mon–Fri  Predicted Temp",
                    xlabel="Day / Time (6-hour intervals)", multi_day=True)

    else:
        print("Invalid graph option.")


# ── 메인 ───────────────────────────────────────────────────────────────────────
select_period()
load_and_train()

while True:
    print("\n--- 인공지능 온도 예측 시스템 ---")
    cmd = input("진행하시려면 Enter, 종료하시려면 'q'를 입력하세요: ")
    if cmd.strip().lower() == 'q':
        print("프로그램을 종료합니다.")
        break

    zone_input = input("구역 번호 입력 (0~12) 또는 요일시간 예측을 위해 Enter: ")
    if zone_input.isdigit() and 0 <= int(zone_input) <= 12:
        zone_num = int(zone_input)
        g_option = input(
            "그래프 출력 (g+1: 13~15시/10분, g+2: 하루/30분, g+3: 월화수/2h, g+4: 일주일/6h) "
            "/ 스킵하려면 Enter: "
        )
        if g_option in ['g+1', 'g+2', 'g+3', 'g+4']:
            plot_graph(zone_num, g_option)
            continue

    yoil = input('요일 입력(0=월, 1=화, 2=수, 3=목, 4=금, 5=토, 6=일): ')
    if yoil.strip().lower() == 'q':
        break
    hour = input('24시간제 시간 입력: ')
    if hour.strip().lower() == 'q':
        break
    minut = input('분 입력: ')
    if minut.strip().lower() == 'q':
        break

    get_expected_temperature(yoil, hour, minut)