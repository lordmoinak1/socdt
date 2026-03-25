from __future__ import annotations
import os, argparse
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import json

# Optional NIfTI
try:
    import nibabel as nib
except Exception:
    nib = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold


def set_seed(seed=42):
    import random
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _maybe_resize_binary(a: np.ndarray, img_size: Optional[int]):
    if img_size is None: return a
    if a.ndim == 2:
        im = Image.fromarray((a > 0.5).astype(np.uint8) * 255)
        im = im.resize((img_size, img_size), resample=Image.NEAREST)
        return (np.array(im) > 127).astype(np.float32)
    elif a.ndim == 3:
        out = []
        for z in range(a.shape[-1]):
            im = Image.fromarray((a[..., z] > 0.5).astype(np.uint8) * 255)
            im = im.resize((img_size, img_size), resample=Image.NEAREST)
            out.append((np.array(im) > 127).astype(np.float32))
        return np.stack(out, -1).astype(np.float32)
    return a

def load_binary_mask(path: str | Path, img_size: Optional[int] = None) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == ".npy":
        a = np.load(p)
    elif p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        a = np.array(Image.open(p).convert("L"))
    elif str(p).endswith(".nii") or str(p).endswith(".nii.gz"):
        if nib is None: raise RuntimeError(f"Need nibabel to read NIfTI: {p}")
        a = np.asanyarray(nib.load(str(p)).get_fdata())
    else:
        raise ValueError(f"Unsupported mask format: {p}")
    a = a.astype(np.float32)
    a = (a > 0).astype(np.float32) if a.max() <= 3 else (a >= 128).astype(np.float32)
    # Squeeze degenerate axes (keep 2D or 3D)
    if a.ndim == 3 and 1 in a.shape:
        a = np.squeeze(a)
        if a.ndim == 1:
            a = a[None, None]
    a = _maybe_resize_binary(a, img_size)
    return np.ascontiguousarray(a, dtype=np.float32)

def resolve_under_root(root: Path, mp: str | Path) -> Path:
    mp = str(mp).strip()
    p = Path(mp)
    if p.is_absolute() and p.exists(): return p
    cand = (root / p)
    if cand.exists(): return cand
    base = Path(mp).name
    hits = [Path(x) for x in glob(str(root / "**" / base), recursive=True)]
    if len(hits) == 1: return hits[0]
    if len(hits) > 1:
        hits.sort(key=lambda x: len(str(x)))
        return hits[0]
    return cand

def series_mae(gt_series_np, pred_series_np, include_baseline=True, threshold=0.5, spacing=None):
    gt = [np.asarray(x) for x in gt_series_np]
    pr = [np.asarray(x) for x in pred_series_np]
    if include_baseline and len(pr) == len(gt) - 1:
        pr = [gt[0]] + pr
    def to_bin(x): return (x >= threshold).astype(np.uint8)
    def to_bin_gt(x): return (x.astype(np.float32) > 0).astype(np.uint8)
    gt_bin = [to_bin_gt(x) for x in gt]; pr_bin = [to_bin(x) for x in pr]
    scale = 1.0 if spacing is None else float(np.prod(spacing))
    Vg = [float(x.sum())*scale for x in gt_bin]; Vp = [float(x.sum())*scale for x in pr_bin]
    return float(np.mean([abs(a-b) for a,b in zip(Vg, Vp)]))


MGMT_VOC = ["positive", "negative", "unknown"]
IDH_VOC  = ["mutated", "wildtype", "unknown"]
P1919_VOC= ["codeleted", "relative", "intact", "unknown"]

def _norm(x, scale, clip=5.0):
    x = pd.to_numeric(x, errors="coerce")
    if not np.isfinite(x): return 0.0
    return float(np.tanh(np.clip(x/scale, -clip, clip)))

def _onehot(tok: str, vocab: List[str]) -> List[float]:
    t = (tok or "").strip().lower()
    if not t:
        return [0.0]*len(vocab)
    # map variants
    if any(k in t for k in ["mutated", "idh1", "r132", "r172"]):
        t = "mutated" if "mutated" in vocab else t
    if t in ("wt", "wild type"): t = "wildtype"
    if "co-del" in t or "codelet" in t or "relative co-deletion" in t:
        t = "relative" if "relative" in vocab else "codeleted"
    if t not in vocab:
        if "pos" in t: t = "positive"
        elif "neg" in t: t = "negative"
    out = [0.0]*len(vocab)
    try:
        i = vocab.index(t)
        out[i] = 1.0
    except ValueError:
        out[-1] = 1.0  # unknown bucket
    return out

class UCSFTraj(Dataset):
    """
    metadata_images.csv: patient_id, day, path_mask (2D PNG/JPG or 3D NIfTI/NPY)
    patients_csv: UCSF combined (SubjectID + columns listed in the user's sheet)
    """
    def __init__(self, root: str | Path, img_size: Optional[int] = None,
                 patients_csv: str = "metadata_patient.csv",
                 pid_col_img="patient_id", day_col="day", mask_col="path_mask",
                 pid_col_pat="SubjectID"):
        root = Path(root)
        dfi = pd.read_csv(root / "metadata_images.csv")
        needed = {pid_col_img, day_col, mask_col}
        miss = [c for c in needed if c not in dfi.columns]
        assert not miss, f"metadata_images.csv missing columns: {miss}"

        cov = pd.read_csv(patients_csv) if Path(patients_csv).exists() else None

        self.examples: List[Dict] = []
        for pid, grp in dfi.groupby(pid_col_img):
            g = grp.copy()
            g[day_col] = pd.to_numeric(g[day_col], errors="coerce")
            g = g[np.isfinite(g[day_col])].sort_values(day_col)
            if len(g) < 3: continue

            masks, days = [], []
            ok = True
            for _, r in g.iterrows():
                mp = str(r[mask_col]).strip()
                if mp in ("", "nan", "None"): ok=False; break
                m = load_binary_mask(resolve_under_root(root, mp), img_size=img_size)
                if m.ndim not in (2,3):
                    m = np.squeeze(m)
                    if m.ndim not in (2,3): ok=False; break
                masks.append(m)
                days.append(int(round(float(r[day_col]))))
            if not ok: continue

            # ---- attach patient-level info from UCSF combined ----
            pulses: List[int] = []
            half_life: Optional[float] = None
            chemo_scale: float = 0.0
            cov_gen: List[float] = []
            rt_start_day: float = np.nan

            if cov is not None and not cov.empty:
                row = cov[cov[pid_col_pat].astype(str) == str(pid)]
                if not row.empty:
                    r0 = row.iloc[0]
                    # (a) chemo pulse from Days from 1st chemo start to 1st scan
                    d = pd.to_numeric(r0.get("Days from 1st chemo start to 1st scan"), errors="coerce")
                    if np.isfinite(d):
                        if d > 0: pulses = [-int(round(float(d)))]
                        elif d < 0: pulses = [int(round(abs(float(d))))]
                        else: pulses = [0]
                    # (b) chemo type → scale & default half-life
                    typ = str(r0.get("1st Chemo type", "")).lower()
                    chemo_scale = 1.0 if ("tmz" in typ) else 0.0
                    if "tmz" in typ: half_life = 7.0
                    elif "pcv" in typ: half_life = 10.0
                    else: half_life = None
                    # (c) RT start day
                    rt = pd.to_numeric(r0.get("Days from 1st scan to 1st RT start (neg = RT first)"), errors="coerce")
                    rt_start_day = float(rt) if np.isfinite(rt) else np.nan
                    # (d) genomics/features → cov_gen
                    age = _norm(r0.get("Age at MRI"), scale=100.0)
                    grade = _norm(r0.get("WHO CNS Grade"), scale=4.0)
                    mgmt = _onehot(str(r0.get("MGMT status", "")), MGMT_VOC)
                    idh  = _onehot(str(r0.get("IDH", "")), IDH_VOC)
                    p19  = _onehot(str(r0.get("1p/19q", "")), P1919_VOC)
                    cov_gen = [age, grade] + mgmt + idh + p19

            self.examples.append(dict(
                pid=str(pid), masks=masks, days=days,
                chemo_pulses=pulses, chemo_half_life=half_life, chemo_scale=chemo_scale,
                cov_gen=np.asarray(cov_gen, dtype=np.float32) if cov_gen else np.zeros(2+len(MGMT_VOC)+len(IDH_VOC)+len(P1919_VOC), dtype=np.float32),
                rt_start_day=rt_start_day,
            ))

    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return self.examples[i]

class GlobalScalars(nn.Module):
    def __init__(self, K=1.0, D_bounds=(1e-5, 1.0), k_bounds=(1e-5, 1.0),
                 aCT_bounds=(0.0, 1.0), aRT_bounds=(0.0, 0.5), bRT_bounds=(0.0, 0.05)):
        super().__init__()
        self.K = float(K)
        self.D_lo, self.D_hi = map(float, D_bounds)
        self.k_lo, self.k_hi = map(float, k_bounds)
        self.c_lo, self.c_hi = map(float, aCT_bounds)
        self.r_lo, self.r_hi = map(float, aRT_bounds)
        self.b_lo, self.b_hi = map(float, bRT_bounds)
        # Raw parameters (unconstrained)
        self.raw_D  = nn.Parameter(torch.tensor(0.0))
        self.raw_k  = nn.Parameter(torch.tensor(0.0))
        self.raw_aC = nn.Parameter(torch.tensor(-2.0))
        self.raw_aR = nn.Parameter(torch.tensor(-3.0))
        self.raw_bR = nn.Parameter(torch.tensor(-5.0))

    def forward(self):
        sig = torch.sigmoid
        D  = self.D_lo + (self.D_hi - self.D_lo) * sig(self.raw_D)
        k  = self.k_lo + (self.k_hi - self.k_lo) * sig(self.raw_k)
        aC = self.c_lo + (self.c_hi - self.c_lo) * sig(self.raw_aC)
        aR = self.r_lo + (self.r_hi - self.r_lo) * sig(self.raw_aR)
        bR = self.b_lo + (self.b_hi - self.b_lo) * sig(self.raw_bR)
        return D, k, aC, aR, bR

class GenomicMod(nn.Module):
    """Maps genomic covariates -> multiplicative multipliers for (D,k,aCT)."""
    def __init__(self, cov_dim: int, width: int = 16, lo: float = 0.5, hi: float = 1.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cov_dim, width), nn.GELU(),
            nn.Linear(width, 3)
        )
        self.lo, self.hi = float(lo), float(hi)

    def forward(self, cov: torch.Tensor):
        s = torch.sigmoid(self.net(cov))  # (B,3)
        return self.lo + (self.hi - self.lo) * s  # (B,3)

class IMEXETD(nn.Module):
    def __init__(self, K=1.0, dt=0.1, dim=2, cg_tol=1e-5, cg_maxiter=200):
        super().__init__()
        self.K = float(K)
        self.dt = float(dt)
        self.dim = int(dim)
        self.cg_tol = float(cg_tol)
        self.cg_maxiter = int(cg_maxiter)

    @staticmethod
    def _pad_rep(u):
        if u.ndim == 4:  # (B,1,H,W)
            return F.pad(u, (1,1,1,1), mode='replicate')
        elif u.ndim == 5:  # (B,1,H,W,D)
            return F.pad(u, (1,1,1,1,1,1), mode='replicate')
        else:
            raise ValueError("u must be 4D or 5D")

    def laplace_neumann(self, u):
        if u.ndim == 4:
            up = self._pad_rep(u)
            c = up[..., 1:-1, 1:-1]
            l = up[..., 1:-1, :-2]
            r = up[..., 1:-1, 2:]
            t = up[..., :-2, 1:-1]
            b = up[..., 2:, 1:-1]
            return (-4.0 * c + l + r + t + b)
        else:
            up = self._pad_rep(u)
            c = up[..., 1:-1, 1:-1, 1:-1]
            xm = up[..., 1:-1, 1:-1, :-2]
            xp = up[..., 1:-1, 1:-1, 2:]
            ym = up[..., 1:-1, :-2, 1:-1]
            yp = up[..., 1:-1, 2:, 1:-1]
            zm = up[..., :-2, 1:-1, 1:-1]
            zp = up[..., 2:, 1:-1, 1:-1]
            return (-6.0 * c + xm + xp + ym + yp + zm + zp)

    def _apply_A(self, u, D):
        return u - self.dt * D * self.laplace_neumann(u)

    @torch.no_grad()
    def _cg(self, b, D, x0=None):
        A = lambda x: self._apply_A(x, D)
        x = b.clone() if x0 is None else x0.clone()
        r = b - A(x)
        p = r.clone()
        rs_old = torch.sum(r*r)
        for _ in range(self.cg_maxiter):
            Ap = A(p)
            alpha = rs_old / (torch.sum(p*Ap) + 1e-12)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.sum(r*r)
            if torch.sqrt(rs_new) < self.cg_tol:
                break
            p = r + (rs_new/rs_old) * p
            rs_old = rs_new
        return x

    def step(self, u, D, k, aC, C_t):
        # Implicit diffusion
        u_tilde = self._cg(u, D)
        # Closed-form reaction/chemo per voxel (Riccati)
        a = k - aC * C_t
        b = k / self.K
        while a.ndim < u_tilde.ndim: a = a.unsqueeze(0)
        while b.ndim < u_tilde.ndim: b = b.unsqueeze(0)
        e = torch.exp(torch.clamp(a * self.dt, min=-60.0, max=60.0))
        num = a * u_tilde * e
        den = b * u_tilde * (e - 1.0) + a
        u_next = torch.where(torch.abs(den) > 1e-12, num/den, u_tilde)
        return u_next.clamp(0.0, self.K)

class ChemoSignal:
    def __init__(self, pulses: List[int] | None, half_life: Optional[float]):
        self.pulses = sorted(pulses or [])
        self.hl = float(half_life) if half_life is not None else None
        self.ln2 = np.log(2.0)

    def C(self, t: float) -> float:
        if not self.pulses or self.hl is None or self.hl <= 0:
            return 0.0
        s = 0.0
        for t0 in self.pulses:
            if t >= t0:
                s += np.exp(- self.ln2 * (t - t0) / self.hl)
        return float(s)

@torch.no_grad()
def dsc_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = (y_true.astype(np.uint8) > 0)
    b = (y_pred.astype(np.uint8) > 0)
    inter = float((a & b).sum())
    return (2*inter) / (a.sum() + b.sum() + 1e-6)

def dice_loss_binary(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.clamp(0,1); target = target.clamp(0,1)
    inter = (pred * target).sum()
    return 1.0 - (2*inter + 1e-6) / (pred.sum() + target.sum() + 1e-6)

def mixed_loss(pred: torch.Tensor, target: torch.Tensor, bce_w: float = 0.0) -> torch.Tensor:
    dice = dice_loss_binary(pred, target)
    if bce_w <= 0: return dice
    p = pred.clamp(1e-4, 1-1e-4)
    bce = F.binary_cross_entropy(p, target)
    return (1 - bce_w) * dice + bce_w * bce

def simulate_patient(ex: Dict, params: GlobalScalars, genmod: GenomicMod, solver: IMEXETD,
                     device: torch.device, steps_per_day: float, training: bool,
                     assimilate: bool, alpha: float,
                     D_bounds: Tuple[float,float], k_bounds: Tuple[float,float], aCT_bounds: Tuple[float,float]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Condition on PRE and POST1; predict POST2 only.
    Requires at least 3 timepoints (t0=pre, t1=post1, t2=post2).
    """
    masks_np: List[np.ndarray] = ex['masks']
    days: List[int] = ex['days']

    if len(masks_np) < 3:
        return [], []  # not enough TPs to do post2 prediction

    dim = 3 if masks_np[0].ndim == 3 else 2
    D0,k0,aC0,aRT,bRT = params()
    D0 = D0.to(device); k0 = k0.to(device); aC0 = aC0.to(device); aRT = aRT.to(device); bRT = bRT.to(device)
    solver.dim = dim

    to_t = lambda a: torch.from_numpy(a).float().to(device).unsqueeze(0).unsqueeze(0)
    # Initial state is the OBSERVED post1 mask (conditioning on pre is implicit via absolute time & chemo pulses)
    u = to_t(masks_np[1])

    # Genomic multipliers
    cov = torch.from_numpy(ex.get('cov_gen', np.zeros(12, np.float32))).float().to(device).unsqueeze(0)
    mD, mK, mC = genmod(cov).squeeze(0)
    D = (D0 * mD).clamp(*D_bounds)
    k = (k0 * mK).clamp(*k_bounds)
    aC = (aC0 * mC).clamp(*aCT_bounds)

    # Chemo signal
    chemo = ChemoSignal(ex.get('chemo_pulses', []), ex.get('chemo_half_life', None))
    cscale = float(ex.get('chemo_scale', 1.0))

    # RT schedule (30 × 2 Gy) if start > 0
    rt_start = ex.get('rt_start_day', np.nan)
    rt_events = []
    d_frac = 2.0
    if np.isfinite(rt_start) and (rt_start > 0) and (rt_start < 5000):
        rt_events = [float(rt_start) + i for i in range(30)]

    preds: List[torch.Tensor] = []
    gts  : List[torch.Tensor] = [to_t(masks_np[2])]

    # Simulate only t1 -> t2
    t_curr = float(days[1])
    t_next = float(days[2])
    delta_days = max(1.0, t_next - t_curr)
    nsteps = int(np.ceil(delta_days * steps_per_day))
    for s in range(nsteps):
        t_mid = t_curr + (s + 0.5) * (delta_days / nsteps)
        C_t = torch.tensor([cscale * chemo.C(t_mid)], device=device, dtype=u.dtype)
        u = solver.step(u, D, k, aC, C_t)
        if rt_events:
            t_lo = t_curr + s * (delta_days / nsteps)
            t_hi = t_curr + (s + 1) * (delta_days / nsteps)
            hit = any((t_lo <= te < t_hi) for te in rt_events)
            if hit:
                S = torch.exp(-aRT * d_frac - bRT * (d_frac ** 2))
                while S.ndim < u.ndim: S = S.unsqueeze(0)
                u = u * S
    preds.append(u.clone())

    return preds, gts

def train_one_epoch(dl, params: GlobalScalars, genmod: GenomicMod, solver: IMEXETD, optim, device, args,
                    D_bounds, k_bounds, aCT_bounds) -> float:
    params.train(); genmod.train()
    running, n = 0.0, 0
    for ex in dl:
        optim.zero_grad(set_to_none=True)
        preds, gts = simulate_patient(ex, params, genmod, solver, device,
                                      steps_per_day=args.steps_per_day,
                                      training=True, assimilate=args.assimilate, alpha=args.alpha,
                                      D_bounds=D_bounds, k_bounds=k_bounds, aCT_bounds=aCT_bounds)
        if preds:
            pred = torch.cat(preds, 0)
            gt   = torch.cat(gts, 0)
            loss = mixed_loss(pred, gt, bce_w=args.bce_w)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(list(params.parameters()) + list(genmod.parameters()), args.clip)
            optim.step()
            running += float(loss.detach().cpu()); n += 1
    return running / max(1, n)


@torch.no_grad()
def evaluate_subset(subset, params: GlobalScalars, genmod: GenomicMod, solver: IMEXETD, device, fold: int,
                    steps_per_day: float, D_bounds, k_bounds, aCT_bounds, save_masks_dir: Optional[Path] = None) -> pd.DataFrame:
    params.eval(); genmod.eval()
    rows = []
    # helper to save 2D/3D masks
    def _save_mask(prefix: Path, arr: np.ndarray, is_3d: bool):
        prefix.parent.mkdir(parents=True, exist_ok=True)
        if is_3d:
            np.save(str(prefix) + ".npy", arr.astype(np.float32))
        else:
            Image.fromarray((arr * 255).astype(np.uint8)).save(str(prefix) + ".png")
            np.save(str(prefix) + ".npy", arr.astype(np.float32))

    for ex in subset:
        preds, gts = simulate_patient(ex, params, genmod, solver, device,
                                      steps_per_day=steps_per_day,
                                      training=False, assimilate=False, alpha=1.0,
                                      D_bounds=D_bounds, k_bounds=k_bounds, aCT_bounds=aCT_bounds)
        # Save masks if requested
        out_patient_dir = None
        is_3d = bool(ex['masks'][0].ndim == 3)
        if save_masks_dir is not None:
            out_patient_dir = Path(save_masks_dir) / str(ex['pid'])
            # baseline (pre) and post1 GTs
            base = (ex['masks'][0] > 0).astype(np.uint8)
            _save_mask(out_patient_dir / 'baseline_gt_t0', base, is_3d)
            if len(ex['masks']) >= 2:
                post1 = (ex['masks'][1] > 0).astype(np.uint8)
                _save_mask(out_patient_dir / 'post1_gt_t1', post1, is_3d)

        if not preds:
            D0,k0,aC0,aR,bR = params()
            rows.append(dict(fold=fold, patient=ex['pid'], DSC=np.nan, VolMAE=np.nan,
                             D0=float(D0), k0=float(k0), alpha_CT0=float(aC0), alpha_RT=float(aR), beta_RT=float(bR)))
            continue

        # Save predicted POST2 and its GT
        if out_patient_dir is not None:
            prob = preds[0].squeeze().detach().cpu().numpy().astype(np.float32)
            pmask = (prob >= 0.2).astype(np.uint8)
            gmask = gts[0].squeeze().cpu().numpy().astype(np.uint8)
            _save_mask(out_patient_dir / f'pred_prob_t2', prob, is_3d)
            _save_mask(out_patient_dir / f'pred_mask_t2', pmask, is_3d)
            _save_mask(out_patient_dir / f'gt_mask_t2', gmask, is_3d)
            # metadata dump (days, rt/chemo info)
            meta = dict(days=[int(x) for x in ex['days']],
                        rt_start_day=(float(ex.get('rt_start_day')) if np.isfinite(ex.get('rt_start_day', np.nan)) else None),
                        chemo_pulses=ex.get('chemo_pulses', []),
                        chemo_half_life=ex.get('chemo_half_life', None),
                        chemo_scale=float(ex.get('chemo_scale', 1.0)))
            try:
                with open(out_patient_dir / 'meta.json', 'w') as f:
                    json.dump(meta, f, indent=2)
            except Exception:
                pass

        # Metrics
        p_last = (preds[0].squeeze().cpu().numpy() >= 0.2).astype(np.uint8)
        g_last = (gts[0].squeeze().cpu().numpy()   >= 0.2).astype(np.uint8)
        dsc = dsc_binary(g_last, p_last)

        # VolMAE over [baseline, post2] only (since we condition on post1)
        baseline = ex['masks'][0].astype(np.uint8)
        gt2      = ex['masks'][2].astype(np.uint8)
        pred_series = [baseline, p_last]
        gt_series   = [baseline, gt2]
        vol_mae = series_mae(gt_series, pred_series, include_baseline=False)

        D0,k0,aC0,aR,bR = params()
        rows.append(dict(fold=fold, patient=ex['pid'], DSC=float(dsc), VolMAE=float(vol_mae),
                         D0=float(D0), k0=float(k0), alpha_CT0=float(aC0), alpha_RT=float(aR), beta_RT=float(bR)))
    return pd.DataFrame(rows)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, required=True)
    ap.add_argument('--patients_csv', type=str, required=True)
    ap.add_argument('--save_root', type=str, default='runs/socdt_ucsf_1')

    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--img_size', type=int, default=None)

    # Columns
    ap.add_argument('--pid_col_img', type=str, default='patient_id')
    ap.add_argument('--day_col', type=str, default='day')
    ap.add_argument('--mask_col', type=str, default='path_mask')
    ap.add_argument('--pid_col_pat', type=str, default='SubjectID')

    # PDE / solver
    ap.add_argument('--K', type=float, default=1.0)
    ap.add_argument('--dt', type=float, default=0.1)
    ap.add_argument('--steps_per_day', type=float, default=10.0)
    ap.add_argument('--cg_tol', type=float, default=1e-5)
    ap.add_argument('--cg_maxiter', type=int, default=200)

    # Parameter bounds
    ap.add_argument('--D_min', type=float, default=1e-5)
    ap.add_argument('--D_max', type=float, default=1.0)
    ap.add_argument('--k_min', type=float, default=1e-5)
    ap.add_argument('--k_max', type=float, default=1.0)
    ap.add_argument('--aCT_min', type=float, default=0.0)
    ap.add_argument('--aCT_max', type=float, default=1.0)
    ap.add_argument('--aRT_min', type=float, default=0.0)
    ap.add_argument('--aRT_max', type=float, default=0.5)
    ap.add_argument('--bRT_min', type=float, default=0.0)
    ap.add_argument('--bRT_max', type=float, default=0.05)

    # Genomics modulator
    ap.add_argument('--gen_lo', type=float, default=0.5)
    ap.add_argument('--gen_hi', type=float, default=1.5)
    ap.add_argument('--gen_width', type=int, default=16)

    # Training
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-6)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--bce_w', type=float, default=0.0)

    # Assimilation
    ap.add_argument('--assimilate', action='store_true')
    ap.add_argument('--alpha', type=float, default=0.9)

    # Debug
    ap.add_argument('--detect_anomaly', action='store_true')

    return ap.parse_args()

def main():
    args = parse_args()

    set_seed(args.seed)
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')

    # Dataset
    ds = UCSFTraj(root=args.root, img_size=args.img_size,
                  patients_csv=args.patients_csv,
                  pid_col_img=args.pid_col_img, day_col=args.day_col, mask_col=args.mask_col,
                  pid_col_pat=args.pid_col_pat)
    if len(ds) == 0:
        raise RuntimeError("No valid patients with ≥2 timepoints.")

    ref = ds[0]['masks'][0]
    dim = 3 if ref.ndim == 3 else 2

    n_splits = min(args.folds, max(2, len(ds)//2))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

    outdir = Path(args.save_root); outdir.mkdir(parents=True, exist_ok=True)

    D_bounds = (args.D_min, args.D_max)
    k_bounds = (args.k_min, args.k_max)
    aCT_bounds = (args.aCT_min, args.aCT_max)

    # Model
    cov_dim = len(ds[0]['cov_gen']) if 'cov_gen' in ds[0] else (2+len(MGMT_VOC)+len(IDH_VOC)+len(P1919_VOC))

    all_rows: List[Dict] = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(range(len(ds))), 1):
        print(f"=== Fold {fold}/{n_splits} (dim={dim}D) ===")
        tr = Subset(ds, tr_idx); va = Subset(ds, va_idx)
        tr_dl = DataLoader(tr, batch_size=16, shuffle=True, collate_fn=lambda b: b[0])

        params = GlobalScalars(K=args.K,
                               D_bounds=D_bounds, k_bounds=k_bounds,
                               aCT_bounds=aCT_bounds,
                               aRT_bounds=(args.aRT_min, args.aRT_max),
                               bRT_bounds=(args.bRT_min, args.bRT_max)).to(device)
        genmod = GenomicMod(cov_dim=cov_dim, width=args.gen_width, lo=args.gen_lo, hi=args.gen_hi).to(device)
        solver = IMEXETD(K=args.K, dt=args.dt, dim=dim, cg_tol=args.cg_tol, cg_maxiter=args.cg_maxiter).to(device)

        def count_params(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        print("GlobalScalars:", count_params(params))   # -> 5
        print("GenomicMod:  ", count_params(genmod))   # -> 259
        print("IMEXETD:     ", count_params(solver))   # -> 0
        print("Total:       ", count_params(params) + count_params(genmod) + count_params(solver))  # -> 264

        optim = torch.optim.Adam(list(params.parameters()) + list(genmod.parameters()),
                                 lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)

        for ep in range(1, args.epochs+1):
            loss = train_one_epoch(tr_dl, params, genmod, solver, optim, device, args,
                                   D_bounds, k_bounds, aCT_bounds)
            if ep % max(1, args.epochs//10) == 0 or ep == args.epochs:
                with torch.no_grad():
                    D0,k0,aC0,aR,bR = params()
                    print(f"  ep {ep:03d} | loss {loss:.4f} | D0 {float(D0):.5f} k0 {float(k0):.5f} a_CT0 {float(aC0):.5f} a_RT {float(aR):.5f} b_RT {float(bR):.5f}")

        # Validate
        fold_df = evaluate_subset(va, params, genmod, solver, device, fold,
                                  steps_per_day=args.steps_per_day,
                                  D_bounds=D_bounds, k_bounds=k_bounds, aCT_bounds=aCT_bounds,
                                  save_masks_dir=outdir / f"fold{fold}_masks")
        out_csv = outdir / f"fold{fold}_metrics.csv"
        fold_df.to_csv(out_csv, index=False)
        print(f"Saved per-patient metrics → {out_csv}")

        mu_dsc = float(fold_df['DSC'].mean(skipna=True)) if not fold_df.empty else float('nan')
        mu_mae = float(fold_df['VolMAE'].mean(skipna=True)) if not fold_df.empty else float('nan')
        print(f"Fold {fold} — DSC mean: {mu_dsc:.4f} | VolMAE mean: {mu_mae:.1f}")

        all_rows.extend(fold_df.to_dict(orient='records'))

    # Aggregate
    agg = pd.DataFrame(all_rows)
    agg.to_csv(outdir / 'all_folds_metrics.csv', index=False)
    if not agg.empty:
        perfold = agg.groupby('fold').agg(DSC_mean=('DSC','mean'), VolMAE_mean=('VolMAE','mean'))
        dsc_mu = float(np.mean(perfold['DSC_mean'].to_numpy())) if not perfold.empty else float('nan')
        dsc_sd = float(np.std(perfold['DSC_mean'].to_numpy(), ddof=1)) if len(perfold) > 1 else float('nan')
        mae_mu = float(np.mean(perfold['VolMAE_mean'].to_numpy())) if not perfold.empty else float('nan')
        mae_sd = float(np.std(perfold['VolMAE_mean'].to_numpy(), ddof=1)) if len(perfold) > 1 else float('nan')
        print("================== K-FOLD SUMMARY ==================")
        print(f"DSC (mean over folds):     {dsc_mu:.4f} ± {dsc_sd:.4f}")
        print(f"VolMAE (mean over folds):  {mae_mu:.1f} ± {mae_sd:.1f}")
        print("==================================================")


if __name__ == '__main__':
    main()

    # python3 train.py --root /path/to/SocDT_UCSF --patients_csv "/path/to/ucsf_combined.csv" --lr 1e-5 --weight_decay 1e-5 --pid_col_img patient_id --day_col day --mask_col path_mask --pid_col_pat SubjectID --assimilate --D_max 0.05 --k_max 0.05 --alpha 0.6 --device cuda
