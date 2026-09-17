# made by - Karthik
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from sklearn.ensemble import ExtraTreesRegressor

SEED = 1234
NFOLD = 6
EPOCHS = 12
BATCH = 48
LR = 6e-4
WD = 3e-4
PRIOR_LR = 6e-3
RES_SCALE = 0.006
IMG_SIZE = 96
NTHREAD = 8
BETA = 2.0
ET_TREES = 300
ET_LEAF = 6
ET_SEEDS = (0, 1, 2, 3, 4)
DIR_W = 0.3
FLOOR_Q = 0.40
ALPHA = 1.0
EPS = 1e-9
HOR = np.arange(1, 9, dtype=np.float32)
WGT = np.arange(1, 9, dtype=np.float32) / 36.0
DXC = [f"future_correction_dx_{h}" for h in range(1, 9)]
DYC = [f"future_correction_dy_{h}" for h in range(1, 9)]

torch.set_num_threads(NTHREAD)
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cpu")

public_dir = Path(sys.argv[1])
submission_out = Path(sys.argv[2])

train = pd.read_csv(public_dir / "train.csv")
test = pd.read_csv(public_dir / "test.csv")
sample = pd.read_csv(public_dir / "sample_submission.csv")
OUTCOLS = [c for c in sample.columns if c != "id"]
submission_out.parent.mkdir(parents=True, exist_ok=True)


def load_images(df):
    out = np.zeros((len(df), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    for i, rel in enumerate(df["image_path"].values):
        im = Image.open(public_dir / rel).convert("RGB")
        if im.size != (IMG_SIZE, IMG_SIZE):
            im = im.resize((IMG_SIZE, IMG_SIZE))
        out[i] = np.asarray(im)
    return out


XTR = load_images(train)
XTE = load_images(test)
YTR = np.stack([train[DXC].values, train[DYC].values], axis=2).astype(np.float32)
TR_COEF = (YTR * HOR[None, :, None]).sum(1) / float((HOR ** 2).sum())
TR_CMAG = np.linalg.norm(TR_COEF, axis=1)
TR_CDIR = TR_COEF / np.maximum(TR_CMAG, EPS)[:, None]
MED_CMAG = float(np.median(TR_CMAG))

seed_dir = np.median(TR_COEF, axis=0)
seed_dir = seed_dir / (float(np.linalg.norm(seed_dir)) + 1e-12)
seed_vec = seed_dir * MED_CMAG
holder = pd.DataFrame({"id": test["id"].values})
for h in range(1, 9):
    holder[f"future_correction_dx_{h}"] = float(seed_vec[0]) * h
    holder[f"future_correction_dy_{h}"] = float(seed_vec[1]) * h
holder[["id"] + OUTCOLS].to_csv(submission_out, index=False)


def image_features(IMG):
    n, H, W, _ = IMG.shape
    f = IMG.astype(np.float32) / 255.0
    R, G, B = f[..., 0], f[..., 1], f[..., 2]
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    ys = (np.arange(H, dtype=np.float32) - (H - 1) / 2.0) / (H / 2.0)
    xs = (np.arange(W, dtype=np.float32) - (W - 1) / 2.0) / (W / 2.0)
    XX = xs[None, :]
    YY = ys[:, None]
    out, nm = [], []

    def add(v, name):
        out.append(np.asarray(v, dtype=np.float32).reshape(n))
        nm.append(name)

    gm = gray.reshape(n, -1)
    add(gm.mean(1), "img_mean")
    add(gm.std(1), "img_std")
    add(R.reshape(n, -1).mean(1) - B.reshape(n, -1).mean(1), "img_r_minus_b")
    add(G.reshape(n, -1).mean(1) - B.reshape(n, -1).mean(1), "img_g_minus_b")
    add((f.max(3).reshape(n, -1) - f.min(3).reshape(n, -1)).mean(1), "img_sat")
    thr = np.quantile(gm, 0.90, axis=1)[:, None, None]
    wmask = (gray > thr).astype(np.float32)
    wsum = wmask.sum((1, 2)) + EPS
    fcx = (wmask * XX).sum((1, 2)) / wsum
    fcy = (wmask * YY).sum((1, 2)) / wsum
    add(fcx, "foam_cx")
    add(fcy, "foam_cy")
    add(np.hypot(fcx, fcy), "foam_r")
    thr2 = np.quantile(gm, 0.10, axis=1)[:, None, None]
    dmask = (gray < thr2).astype(np.float32)
    dsum = dmask.sum((1, 2)) + EPS
    dcx = (dmask * XX).sum((1, 2)) / dsum
    dcy = (dmask * YY).sum((1, 2)) / dsum
    add(dcx, "dark_cx")
    add(dcy, "dark_cy")
    add(fcx - dcx, "foam_minus_dark_x")
    add(fcy - dcy, "foam_minus_dark_y")
    med = np.median(gm, axis=1)[:, None, None]
    sal = np.abs(gray - med)
    ssum = sal.sum((1, 2)) + EPS
    scx = (sal * XX).sum((1, 2)) / ssum
    scy = (sal * YY).sum((1, 2)) / ssum
    add(scx, "sal_cx")
    add(scy, "sal_cy")
    mxx = (sal * (XX - scx[:, None, None]) ** 2).sum((1, 2)) / ssum
    myy = (sal * (YY - scy[:, None, None]) ** 2).sum((1, 2)) / ssum
    mxy = (sal * (XX - scx[:, None, None]) * (YY - scy[:, None, None])).sum((1, 2)) / ssum
    add(mxx - myy, "sal_m2_diff")
    add(2 * mxy, "sal_m2_cross")
    tr_ = mxx + myy + EPS
    add((mxx - myy) / tr_, "sal_cos2t")
    add(2 * mxy / tr_, "sal_sin2t")
    add(np.sqrt(tr_), "sal_scale")
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, :, 1:-1] = gray[:, :, 2:] - gray[:, :, :-2]
    gy[:, 1:-1, :] = gray[:, 2:, :] - gray[:, :-2, :]
    for tag, sl in [("all", (slice(None), slice(None))), ("ctr", (slice(24, 72), slice(24, 72)))]:
        a = gx[:, sl[0], sl[1]]
        b = gy[:, sl[0], sl[1]]
        Jxx = (a * a).mean((1, 2))
        Jyy = (b * b).mean((1, 2))
        Jxy = (a * b).mean((1, 2))
        t = Jxx + Jyy + EPS
        add(np.sqrt(t), f"grad_energy_{tag}")
        add((Jxx - Jyy) / t, f"grad_cos2t_{tag}")
        add(2 * Jxy / t, f"grad_sin2t_{tag}")
        add(np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2) / t, f"grad_coh_{tag}")
    add(gray[:, :, :48].reshape(n, -1).mean(1) - gray[:, :, 48:].reshape(n, -1).mean(1), "asym_lr")
    add(gray[:, :24, :].reshape(n, -1).mean(1) - gray[:, 72:, :].reshape(n, -1).mean(1), "asym_tb")
    add(gray[:, :24, :].reshape(n, -1).mean(1), "strip_top")
    add(gray[:, 72:, :].reshape(n, -1).mean(1), "strip_bot")
    add(gray[:, 36:60, 36:60].reshape(n, -1).mean(1) - gm.mean(1), "centre_minus_all")
    add(gray[:, 36:60, 36:60].reshape(n, -1).std(1), "centre_std")
    add(gray[:, 24:48, :].reshape(n, -1).std(1) - gray[:, 48:72, :].reshape(n, -1).std(1), "std_ab_diff")
    return np.stack(out, 1).astype(np.float32), nm


class FeatureBuilder:
    def fit(self, df):
        self.ar = float(np.median(df["aspect_ratio"].values * df["bbox_height"].values
                                  / np.maximum(df["bbox_width"].values, EPS)))
        top = df["center_y"].values - df["bbox_height"].values / 2.0
        self.y_hor = float(np.quantile(top, 0.01)) - 0.02
        self.types = sorted(df["object_type"].astype(str).unique().tolist())
        return self

    def velocity(self, df):
        bw = df["bbox_width"].values
        bh = df["bbox_height"].values
        bs = np.sqrt(np.maximum(bw * bh, 1e-12))
        vx = df["past_dx"].values * np.sqrt(self.ar) / (8.0 * bs)
        vy = df["past_dy"].values / (np.sqrt(self.ar) * 8.0 * bs)
        return vx, vy, bs

    def transform(self, df, IMG):
        ar = self.ar
        vx, vy, bs = self.velocity(df)
        sp = np.hypot(vx, vy)
        inv = 1.0 / (sp + EPS)
        ux, uy = vx * inv, vy * inv
        bw = df["bbox_width"].values
        bh = df["bbox_height"].values
        cx = df["center_x"].values - 0.5
        cy = df["center_y"].values - 0.5
        asp = np.maximum(df["aspect_ratio"].values, 1e-6)
        lasp = np.log(asp)
        larea = np.log(np.maximum(df["area_ratio"].values, 1e-12))
        lbs = np.log(np.maximum(bs, 1e-12))
        pmr = df["past_motion_ratio"].values
        isU = (df["object_type"].astype(str).values == self.types[-1]).astype(np.float64)
        pdx = df["past_dx"].values * np.sqrt(ar)
        pdy = df["past_dy"].values / np.sqrt(ar)
        spf = np.hypot(pdx, pdy) / 8.0
        r_iso = pmr / np.maximum(8.0 * sp, EPS)
        dy_hor = np.maximum(df["center_y"].values - self.y_hor, 1e-3)
        lrng = np.log(1.0 / dy_hor)
        lreal = lbs + lrng
        F = []

        def add(v):
            F.append(np.asarray(v, dtype=np.float64).ravel())

        add(vx * 20); add(vy * 20); add(sp * 20); add(np.log1p(sp * 50)); add(np.sqrt(sp))
        add(ux); add(uy); add(ux * ux - uy * uy); add(2 * ux * uy)
        add(np.abs(ux)); add(np.abs(uy))
        add(pdx * 100); add(pdy * 100); add(spf * 100); add(np.log1p(spf * 500))
        add(cx); add(cy); add(np.abs(cx)); add(np.abs(cy)); add(np.hypot(cx, cy))
        add(bw); add(bh); add(bs); add(lbs); add(larea); add(asp); add(lasp)
        add(bw * ar); add(np.hypot(bw * ar, bh))
        add(cy + bh / 2.0); add(cy - bh / 2.0)
        add(isU)
        add(pmr); add(np.log1p(pmr * 5)); add(r_iso); add(np.log(np.maximum(r_iso, 1e-3)))
        add(lrng); add(dy_hor); add(lreal); add(np.exp(lreal) * 100)
        add(sp * 20 * isU); add(sp * 20 * larea); add(sp * 20 * cy); add(sp * 20 * np.abs(cy))
        add(sp * 20 * lrng); add(sp * 20 * lreal)
        add(vx * 20 * lasp); add(vy * 20 * lasp)
        add(vx * 20 * cy); add(vy * 20 * cy); add(vx * 20 * cx); add(vy * 20 * cx)
        add(vx * 20 * isU); add(vy * 20 * isU)
        add(vx / np.maximum(bs, 1e-6) * 0.05); add(vy / np.maximum(bs, 1e-6) * 0.05)
        add(sp * bs * 300); add(sp * 20 * asp); add(sp * 20 * np.hypot(cx, cy))
        add((ux * ux - uy * uy) * lasp)
        add(ux * cx + uy * cy); add(-uy * cx + ux * cy)
        add(ux * lrng); add(uy * lrng); add(np.log1p(sp * 50) * lreal)
        Xt = np.stack(F, 1).astype(np.float32)
        IF, INM = image_features(IMG)

        def col(name):
            return IF[:, INM.index(name)].astype(np.float64)

        extra = []
        for bx, by in [("foam_cx", "foam_cy"), ("sal_cx", "sal_cy"),
                       ("foam_minus_dark_x", "foam_minus_dark_y")]:
            ax, ay = col(bx), col(by)
            extra.append(ax * ux + ay * uy)
            extra.append(-ax * uy + ay * ux)
        mc2, ms2 = ux * ux - uy * uy, 2 * ux * uy
        for tag in ["all", "ctr"]:
            c2, s2 = col(f"grad_cos2t_{tag}"), col(f"grad_sin2t_{tag}")
            extra.append(c2 * mc2 + s2 * ms2)
            extra.append(-c2 * ms2 + s2 * mc2)
        c2, s2 = col("sal_cos2t"), col("sal_sin2t")
        extra.append(c2 * mc2 + s2 * ms2)
        extra.append(-c2 * ms2 + s2 * mc2)
        Xt = np.concatenate([Xt, IF, np.stack(extra, 1).astype(np.float32)], 1)
        return np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)


fb = FeatureBuilder().fit(train)
WTR = fb.transform(train, XTR)
WTE = fb.transform(test, XTE)
TR_VX, TR_VY, _ = fb.velocity(train)
TE_VX, TE_VY, _ = fb.velocity(test)
TR_VX = TR_VX.astype(np.float32)
TR_VY = TR_VY.astype(np.float32)
TE_VX = TE_VX.astype(np.float32)
TE_VY = TE_VY.astype(np.float32)
TR_CX = train.center_x.values.astype(np.float32)
TR_CY = train.center_y.values.astype(np.float32)
TE_CX = test.center_x.values.astype(np.float32)
TE_CY = test.center_y.values.astype(np.float32)
STAT = np.column_stack([
    np.log(np.maximum(train.area_ratio.values, 1e-9)),
    np.log(np.maximum(train.aspect_ratio.values, 1e-6)),
    train.bbox_width.values, train.bbox_height.values,
    (train.object_type.astype(str).values == fb.types[-1]).astype(float)]).astype(np.float32)
STAT_T = np.column_stack([
    np.log(np.maximum(test.area_ratio.values, 1e-9)),
    np.log(np.maximum(test.aspect_ratio.values, 1e-6)),
    test.bbox_width.values, test.bbox_height.values,
    (test.object_type.astype(str).values == fb.types[-1]).astype(float)]).astype(np.float32)
SM, SS = STAT.mean(0), STAT.std(0) + 1e-6


def net_feats(vx, vy, cx, cy, stat):
    sp = np.hypot(vx, vy)
    inv = 1.0 / (sp + 1e-9)
    return np.column_stack([vx * 20, vy * 20, sp * 20, vx * inv, vy * inv, np.log1p(sp * 50),
                            cx - 0.5, cy - 0.5, (stat - SM) / SS]).astype(np.float32)


NFEAT = net_feats(TR_VX, TR_VY, TR_CX, TR_CY, STAT).shape[1]
IMNET_M = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_S = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_tensor(arr):
    x = torch.from_numpy(np.ascontiguousarray(arr)).float().permute(0, 3, 1, 2) / 255.0
    return (x - IMNET_M) / IMNET_S


class TrackNet(nn.Module):
    def __init__(self, nfeat):
        super().__init__()
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.visual = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool,
                                    m.layer1, m.layer2, m.layer3, m.layer4,
                                    nn.AdaptiveAvgPool2d(1), nn.Flatten())
        for p in list(m.conv1.parameters()) + list(m.bn1.parameters()) + list(m.layer1.parameters()):
            p.requires_grad = False
        self.tabular = nn.Sequential(nn.Linear(nfeat, 96), nn.SiLU(),
                                     nn.Linear(96, 96), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(512 + 96, 128), nn.SiLU(), nn.Dropout(0.3),
                                  nn.Linear(128, 2))
        self.prior = nn.Linear(3, 2)
        nn.init.zeros_(self.prior.weight)
        nn.init.zeros_(self.prior.bias)

    def forward(self, img, tab, mot):
        z = torch.cat([self.visual(img), self.tabular(tab)], dim=1)
        return self.head(z) * RES_SCALE + self.prior(mot)


HT = torch.from_numpy(HOR).view(1, 8, 1)
WT = torch.from_numpy(WGT).view(1, 8)


def trajectory_loss(coef, target):
    pred = coef[:, None, :] * HT
    err = torch.linalg.norm(pred - target, dim=2)
    pm = torch.linalg.norm(pred, dim=2)
    tm = torch.linalg.norm(target, dim=2)
    return ((err + BETA * torch.relu(tm - pm)) * WT).sum(1).mean()


groups = train["group_key"].values
uniq = np.array(sorted(set(groups.tolist())))
gsize = np.array([(groups == u).sum() for u in uniq])
load = np.zeros(NFOLD)
gfold = {}
for gi in np.argsort(-gsize):
    k = int(np.argmin(load))
    gfold[uniq[gi]] = k
    load[k] += gsize[gi]
fold_of = np.array([gfold[g] for g in groups])

cnn_acc = np.zeros((len(test), 2), dtype=np.float64)
tab_te = net_feats(TE_VX, TE_VY, TE_CX, TE_CY, STAT_T)
tab_te_f = net_feats(-TE_VX, TE_VY, 1.0 - TE_CX, TE_CY, STAT_T)
mot_te = np.column_stack([TE_VX, TE_VY, np.ones(len(test))]).astype(np.float32)
mot_te_f = np.column_stack([-TE_VX, TE_VY, np.ones(len(test))]).astype(np.float32)

for fold in range(NFOLD):
    torch.manual_seed(SEED + fold)
    np.random.seed(SEED + fold)
    tri = np.where(fold_of != fold)[0]
    net = TrackNet(NFEAT).to(device)
    pri = list(net.prior.parameters())
    oth = [p for nm, p in net.named_parameters() if p.requires_grad and not nm.startswith("prior")]
    opt = torch.optim.AdamW([{"params": oth, "lr": LR, "weight_decay": WD},
                             {"params": pri, "lr": PRIOR_LR, "weight_decay": 0.0}], lr=LR)
    steps = int(np.ceil(len(tri) / BATCH)) * EPOCHS
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR, PRIOR_LR], total_steps=steps,
                                              pct_start=0.3)
    for ep in range(EPOCHS):
        net.train()
        perm = np.random.permutation(tri)
        for s in range(0, len(perm), BATCH):
            b = perm[s:s + BATCH]
            fx = np.random.rand() < 0.5
            fy = np.random.rand() < 0.5
            sx = -1.0 if fx else 1.0
            sy = -1.0 if fy else 1.0
            im = XTR[b]
            if fx:
                im = im[:, :, ::-1]
            if fy:
                im = im[:, ::-1]
            tb = net_feats(TR_VX[b] * sx, TR_VY[b] * sy,
                           0.5 + (TR_CX[b] - 0.5) * sx, 0.5 + (TR_CY[b] - 0.5) * sy, STAT[b])
            mb = np.column_stack([TR_VX[b] * sx, TR_VY[b] * sy, np.ones(len(b))]).astype(np.float32)
            yb = YTR[b].copy()
            yb[:, :, 0] *= sx
            yb[:, :, 1] *= sy
            opt.zero_grad()
            out = net(to_tensor(im), torch.from_numpy(tb), torch.from_numpy(mb))
            trajectory_loss(out, torch.from_numpy(yb)).backward()
            opt.step()
            sch.step()
    net.eval()
    with torch.no_grad():
        parts = []
        for s in range(0, len(test), 64):
            sl = slice(s, s + 64)
            o0 = net(to_tensor(XTE[sl]), torch.from_numpy(tab_te[sl]),
                     torch.from_numpy(mot_te[sl])).numpy()
            o1 = net(to_tensor(XTE[sl][:, :, ::-1]), torch.from_numpy(tab_te_f[sl]),
                     torch.from_numpy(mot_te_f[sl])).numpy()
            o1[:, 0] *= -1.0
            parts.append((o0 + o1) / 2.0)
        cnn_acc += np.concatenate(parts).astype(np.float64)
    print("cnn fold", fold, "complete", flush=True)

cnn_coef = cnn_acc / NFOLD

et_dir = np.zeros((len(test), 2), dtype=np.float64)
nfit = 0
for fold in range(NFOLD):
    tri = np.where(fold_of != fold)[0]
    for s in ET_SEEDS:
        md = ExtraTreesRegressor(n_estimators=ET_TREES, min_samples_leaf=ET_LEAF,
                                 max_features=0.5, n_jobs=NTHREAD, random_state=s)
        md.fit(WTR[tri], TR_CDIR[tri])
        et_dir += md.predict(WTE)
        nfit += 1
    print("et fold", fold, "complete", flush=True)
et_dir /= nfit


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


FLOOR = float(np.quantile(TR_CMAG, FLOOR_Q))
magnitude = np.maximum(np.linalg.norm(cnn_coef, axis=1), FLOOR)
direction = unit(DIR_W * unit(et_dir) + (1.0 - DIR_W) * unit(cnn_coef))
coef = direction * (magnitude * ALPHA)[:, None]
traj = coef[:, None, :] * HOR[None, :, None].astype(np.float64)
traj = np.nan_to_num(traj, nan=0.0, posinf=0.0, neginf=0.0)

submission = pd.DataFrame({"id": test["id"].values})
for j, h in enumerate(range(1, 9)):
    submission[f"future_correction_dx_{h}"] = traj[:, j, 0]
    submission[f"future_correction_dy_{h}"] = traj[:, j, 1]
submission[["id"] + OUTCOLS].to_csv(submission_out, index=False)
print("rows", len(submission), flush=True)
