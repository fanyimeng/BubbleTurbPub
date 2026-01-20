import os
import glob
import random
import re

refdates = ['2025/03/21', '2025/03/22']
hourangles = ['+1.25h', '+3.94h']
skymodel = r"/Volumes/tergeo/directuv_data/simobs/CAR_B05_MP_C0402_1222test3/CAR_B05_MP_C0402_1222test3_jypix.fits"
workdir = r"/Volumes/tergeo/directuv_data/simobs/CAR_B05_MP_C0402_1222test3"
project_base = "CAR_B05_MP_C0402_1222test3"
antennalist = "vla.d.cfg"
incenter = "1.42GHz"
inwidth = "2kHz"
totaltime = "185.0s"
integration = "10s"
thermalnoise = "tsys-atm"
user_pwv = 0.5
seed = 11111
flag_ant_max = 5
flag_ant_seed = 3121806708
flag_ant_lists = [['EA23'], ['EA12EA25EA18EA08']]
ant_base = os.path.splitext(os.path.basename(antennalist))[0]
ms_list = []

os.makedirs(workdir, exist_ok=True)
os.chdir(workdir)

if True:
    old_dirs = glob.glob(os.path.join(workdir, f"{project_base}_d*"))
    for d in old_dirs:
        os.system('rm -rf "%s"' % d)

def safe_clear_tclean_products(imagename):
    pattern = imagename + ".*"
    paths = glob.glob(pattern)
    for path in paths:
        try:
            if os.path.isdir(path):
                os.system('rm -rf "%s"' % path)
            else:
                os.remove(path)
        except Exception as exc:
            print(f"[WARN] cannot remove {path}: {exc}")

def _parse_ant_names(listobs_path):
    names = []
    try:
        with open(listobs_path, "r") as fh:
            for line in fh:
                if line.strip().startswith("ID=") and "'" in line:
                    matches = re.findall(r"'([^']+)'", line)
                    names.extend(matches)
    except Exception as exc:
        print(f"[WARN] listobs parse failed for {listobs_path}: {exc}")
    seen = set()
    ordered = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered

def flag_random_antennas(ms_path, seed_offset, label):
    if flag_ant_max <= 0 and flag_ant_lists is None:
        return []
    if flag_ant_seed is None and flag_ant_lists is None:
        return []
    listobs_out = os.path.join(workdir, f"{label}_listobs.txt")
    try:
        listobs(vis=ms_path, listfile=listobs_out, overwrite=True, verbose=False)
        print(f"listobs -> {listobs_out}")
    except Exception as exc:
        print(f"[WARN] listobs failed for {ms_path}: {exc}")
        return []
    ant_names = _parse_ant_names(listobs_out)
    if not ant_names:
        print("No antennas found for flagging (listobs parsing).")
        return []
    if flag_ant_lists is not None:
        if (seed_offset - 1) >= len(flag_ant_lists):
            return []
        override = [a for a in flag_ant_lists[seed_offset - 1] if a]
        if not override:
            print(f"No antennas flagged for {ms_path} (override empty).")
            return []
        ant_upper = [a.upper() for a in ant_names]
        index_map = dict(zip(ant_upper, ant_names))
        flagged = []
        for token in override:
            tok = token.strip().upper()
            if not tok:
                continue
            if tok in index_map:
                flagged.append(index_map[tok])
                continue
            m = re.match(r"^EA(\d+)$", tok)
            if m:
                idx = int(m.group(1))
                if 1 <= idx <= len(ant_names):
                    flagged.append(ant_names[idx - 1])
                    continue
        if not flagged:
            print(f"No matching antennas to flag for {ms_path}.")
            return []
        seen = set()
        flag_ants = []
        for name in flagged:
            if name in seen:
                continue
            seen.add(name)
            flag_ants.append(name)
    else:
        if flag_ant_seed is None or flag_ant_max <= 0:
            return []
        rng = random.Random(flag_ant_seed + seed_offset)
        n_flag = rng.randint(0, min(flag_ant_max, len(ant_names)))
        if n_flag <= 0:
            print(f"No antennas flagged for {ms_path} (n_flag=0).")
            return []
        rng = random.Random(flag_ant_seed + seed_offset + 10000)
        flag_ants = rng.sample(ant_names, n_flag)
    flag_str = ";".join(flag_ants)
    try:
        flagdata(vis=ms_path, mode="manual", antenna=flag_str, action="apply", flagbackup=False)
        print("Flagged antennas (%d): %s" % (len(flag_ants), flag_str))
    except Exception as exc:
        print("[WARN] flagging antennas failed: %s" % exc)
        return []
    return flag_ants

for idx, refdate in enumerate(refdates, 1):
    proj = f"{project_base}_d{idx}"
    proj_dir = os.path.join(workdir, proj)
    ms_path = os.path.join(proj_dir, f"{proj}.{ant_base}.noisy.ms")
    alt_ms = os.path.join(proj_dir, f"{proj}.{ant_base}.ms")
    ha_val = hourangles[idx - 1] if len(hourangles) >= idx else "transit"

    if True and os.path.exists(proj_dir):
        os.system('rm -rf "%s"' % proj_dir)

    ms_exists = os.path.exists(ms_path) or os.path.exists(alt_ms)
    if ms_exists and not True:
        ms_list.append(os.path.abspath(ms_path if os.path.exists(ms_path) else alt_ms))
        continue

    simobserve(
        project=proj,
        skymodel=skymodel,
        inbright="",
        indirection="",
        incell="",
        incenter=incenter,
        inwidth=inwidth,
        complist="",
        compwidth="",
        comp_nchan=1,
        setpointings=True,
        ptgfile="$project.ptg.txt",
        integration=integration,
        direction=[],
        mapsize=["''", "''"],
        maptype="hexagonal",
        pointingspacing="",
        caldirection="",
        calflux="1Jy",
        obsmode="int",
        refdate=refdate,
        hourangle=ha_val,
        totaltime=totaltime,
        antennalist=antennalist,
        sdantlist="",
        sdant=0,
        outframe="LSRK",
        thermalnoise=thermalnoise,
        user_pwv=user_pwv,
        t_ground=269.0,
        t_sky=260.0,
        tau0=0.1,
        seed=seed + idx,
        leakage=0.0,
        graphics="both",
        verbose=False,
        overwrite=True,
    )
    ms_target = None
    if os.path.exists(ms_path):
        ms_target = ms_path
    elif os.path.exists(alt_ms):
        ms_target = alt_ms
    if ms_target:
        ms_list.append(os.path.abspath(ms_target))
        flag_random_antennas(ms_target, seed_offset=idx, label=proj)

if True and len(ms_list) > 1:
    concat_vis = os.path.join(workdir, "CAR_B05_MP_C0402_1222test3_concat.ms")
    if (not True) and os.path.exists(concat_vis):
        ms_list = [concat_vis]
    else:
        if os.path.exists(concat_vis):
            os.system('rm -rf "%s"' % concat_vis)
        concat(vis=ms_list, concatvis=concat_vis, freqtol="", dirtol="", copypointing=False)
        ms_list = [concat_vis]

if not ms_list:
    print("No MS produced; exiting.")
    raise SystemExit(0)

# Legacy flagging: fixed count on the final MS when no per-day seed is provided.
listobs_out = os.path.join(workdir, f"{project_base}_concat.listobs" if len(ms_list) == 1 else f"{project_base}_listobs.txt")
if flag_ant_seed is None and flag_ant_lists is None:
    ant_names = []
    try:
        listobs(vis=ms_list[0], listfile=listobs_out, overwrite=True, verbose=False)
        print(f"listobs -> {listobs_out}")
        ant_names = _parse_ant_names(listobs_out)
    except Exception as exc:
        print(f"[WARN] listobs (pre-flag) failed: {exc}")

    if flag_ant_max > 0 and ant_names:
        try:
            random.seed(seed + 999)
            n_flag = min(flag_ant_max, len(ant_names))
            flag_ants = random.sample(list(set(ant_names)), n_flag)
            flag_str = ";".join(flag_ants)
            flagdata(vis=ms_list[0], mode="manual", antenna=flag_str, action="apply", flagbackup=False)
            print("Flagged antennas (%d): %s" % (n_flag, flag_str))
        except Exception as exc:
            print("[WARN] flagging antennas failed: %s" % exc)
    elif flag_ant_max > 0:
        print("No antennas found for flagging (listobs parsing).")

# listobs again after flagging for bookkeeping
try:
    listobs(vis=ms_list[0], listfile=listobs_out, overwrite=True, verbose=False)
    print(f"listobs -> {listobs_out}")
except Exception as exc:
    print(f"[WARN] listobs (post-flag) failed: {exc}")

imagename = os.path.join(workdir, f"{project_base}_dirty")
if not True and os.path.exists(imagename + ".image"):
    print(f"tclean output exists and overwrite=False; skip tclean for {project_base}.")
else:
    if True:
        safe_clear_tclean_products(imagename)
    tclean(
        vis=ms_list,
        imagename=imagename,
        field="",
        spw="",
        specmode="mfs",
        imsize=[96, 96],
        cell="5arcsec",
        weighting="briggs",
        robust=0.5,
        niter=0,
        threshold="0mJy",
        nterms=1,
        gridder="standard",
        deconvolver="hogbom",
        savemodel="none",
        pbcor=False,
    )

    exportfits(imagename=f"{imagename}.image", fitsimage=f"{imagename}.image.fits", overwrite=True, dropstokes=True)
    exportfits(imagename=f"{imagename}.psf", fitsimage=f"{imagename}.psf.fits", overwrite=True, dropstokes=True)
    if os.path.exists(f"{imagename}.pb"):
        exportfits(imagename=f"{imagename}.pb", fitsimage=f"{imagename}.pb.fits", overwrite=True, dropstokes=True)
    if os.path.exists(f"{imagename}.pbcor"):
        exportfits(imagename=f"{imagename}.pbcor", fitsimage=f"{imagename}.pbcor.fits", overwrite=True, dropstokes=True)
    print("CASA simobserve+tclean done.")