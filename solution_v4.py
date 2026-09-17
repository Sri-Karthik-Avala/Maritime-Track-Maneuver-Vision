# made by - Karthik
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision

SEED = 1234
NFOLD = 6
SEEDS = (0, 1, 2)
EPOCHS = 12
BATCH = 48
LR = 6e-4
WD = 3e-4
IMG_SIZE = 96
NTHREAD = 8
BETA = 2.0
ALPHA = 1.0
FLOOR_Q = 0.4
PRIOR_LR = 6e-3
RES_SCALE = 0.006
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

AR = float(np.median(train["aspect_ratio"].values * train["bbox_height"].values
                     / np.maximum(train["bbox_width"].values, 1e-9)))
SQAR = float(np.sqrt(AR))


def raw_inputs(df):
    bw = df["bbox_width"].values.astype(np.float32)
    bh = df["bbox_height"].values.astype(np.float32)
    bs = np.sqrt(np.maximum(bw * bh, 1e-12))
    vx = (df["past_dx"].values * SQAR / (8.0 * bs)).astype(np.float32)
    vy = (df["past_dy"].values / (SQAR * 8.0 * bs)).astype(np.float32)
    cx = df["center_x"].values.astype(np.float32)
    cy = df["center_y"].values.astype(np.float32)
    stat = np.column_stack([
        np.log(np.maximum(df["area_ratio"].values, 1e-9)),
        np.log(np.maximum(df["aspect_ratio"].values, 1e-6)),
        bw, bh,
        (df["object_type"].values == "USV").astype(np.float64),
    ]).astype(np.float32)
    return vx, vy, cx, cy, stat


TR_VX, TR_VY, TR_CX, TR_CY, TR_ST = raw_inputs(train)
TE_VX, TE_VY, TE_CX, TE_CY, TE_ST = raw_inputs(test)
ST_M = TR_ST.mean(0)
ST_S = TR_ST.std(0) + 1e-6


def make_feats(vx, vy, cx, cy, stat):
    sp = np.hypot(vx, vy)
    inv = 1.0 / (sp + 1e-6)
    f = np.column_stack([
        vx * 20.0, vy * 20.0, sp * 20.0,
        vx * inv, vy * inv,
        np.log1p(sp * 50.0),
        cx - 0.5, cy - 0.5,
        (stat - ST_M) / ST_S,
    ])
    return f.astype(np.float32)


NFEAT = make_feats(TR_VX, TR_VY, TR_CX, TR_CY, TR_ST).shape[1]


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
seed_dir = np.median(TR_COEF, axis=0)
seed_dir = seed_dir / (float(np.linalg.norm(seed_dir)) + 1e-12)
seed_vec = seed_dir * float(np.median(TR_CMAG))
holder = pd.DataFrame({"id": test["id"].values})
for h in range(1, 9):
    holder[f"future_correction_dx_{h}"] = float(seed_vec[0]) * h
    holder[f"future_correction_dy_{h}"] = float(seed_vec[1]) * h
holder = holder[["id"] + OUTCOLS]
holder.to_csv(submission_out, index=False)

IMNET_M = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_S = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_tensor(arr):
    x = torch.from_numpy(np.ascontiguousarray(arr)).float().permute(0, 3, 1, 2) / 255.0
    return (x - IMNET_M) / IMNET_S


class TrackNet(nn.Module):
    def __init__(self, nfeat):
        super().__init__()
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.visual = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4,
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        for p in list(m.conv1.parameters()) + list(m.bn1.parameters()) + list(m.layer1.parameters()):
            p.requires_grad = False
        self.tabular = nn.Sequential(
            nn.Linear(nfeat, 96), nn.SiLU(),
            nn.Linear(96, 96), nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(512 + 96, 128), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(128, 2),
        )
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

dir_acc = np.zeros((len(test), 2), dtype=np.float64)
mag_acc = np.zeros(len(test), dtype=np.float64)
nmodel = 0
tab_te = make_feats(TE_VX, TE_VY, TE_CX, TE_CY, TE_ST)
tab_te_flip = make_feats(-TE_VX, TE_VY, 1.0 - TE_CX, TE_CY, TE_ST)
mot_te = np.column_stack([TE_VX, TE_VY, np.ones(len(test))]).astype(np.float32)
mot_te_flip = np.column_stack([-TE_VX, TE_VY, np.ones(len(test))]).astype(np.float32)

for run in SEEDS:
  for fold in range(NFOLD):
      torch.manual_seed(SEED + 100 * run + fold)
      np.random.seed(SEED + 100 * run + fold)
      tr_idx = np.where(fold_of != fold)[0]
      net = TrackNet(NFEAT).to(device)
      pri = list(net.prior.parameters())
      oth = [p for nm, p in net.named_parameters() if p.requires_grad and not nm.startswith("prior")]
      opt = torch.optim.AdamW(
          [{"params": oth, "lr": LR, "weight_decay": WD},
           {"params": pri, "lr": PRIOR_LR, "weight_decay": 0.0}], lr=LR)
      steps = int(np.ceil(len(tr_idx) / BATCH)) * EPOCHS
      sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR, PRIOR_LR], total_steps=steps, pct_start=0.3)
      for ep in range(EPOCHS):
          net.train()
          perm = np.random.permutation(tr_idx)
          for s in range(0, len(perm), BATCH):
              b = perm[s:s + BATCH]
              fx = np.random.rand() < 0.5
              fy = np.random.rand() < 0.5
              im = XTR[b]
              sx, sy = 1.0, 1.0
              if fx:
                  im = im[:, :, ::-1]
                  sx = -1.0
              if fy:
                  im = im[:, ::-1]
                  sy = -1.0
              tb = make_feats(TR_VX[b] * sx, TR_VY[b] * sy,
                              0.5 + (TR_CX[b] - 0.5) * sx,
                              0.5 + (TR_CY[b] - 0.5) * sy, TR_ST[b])
              yb = YTR[b].copy()
              yb[:, :, 0] *= sx
              yb[:, :, 1] *= sy
              mb = np.column_stack([TR_VX[b] * sx, TR_VY[b] * sy, np.ones(len(b))]).astype(np.float32)
              opt.zero_grad()
              out = net(to_tensor(im), torch.from_numpy(tb), torch.from_numpy(mb))
              loss = trajectory_loss(out, torch.from_numpy(yb))
              loss.backward()
              opt.step()
              sch.step()
      net.eval()
      with torch.no_grad():
          parts = []
          for s in range(0, len(test), 64):
              sl = slice(s, s + 64)
              o0 = net(to_tensor(XTE[sl]), torch.from_numpy(tab_te[sl]), torch.from_numpy(mot_te[sl])).numpy()
              o1 = net(to_tensor(XTE[sl][:, :, ::-1]), torch.from_numpy(tab_te_flip[sl]),
                       torch.from_numpy(mot_te_flip[sl])).numpy()
              o1[:, 0] *= -1.0
              parts.append((o0 + o1) / 2.0)
          pr = np.concatenate(parts).astype(np.float64)
          nrm = np.linalg.norm(pr, axis=1)
          dir_acc += pr / np.maximum(nrm, 1e-12)[:, None]
          mag_acc += nrm
          nmodel += 1
      print("run", run, "fold", fold, "complete", flush=True)

FLOOR = float(np.quantile(TR_CMAG, FLOOR_Q))
direction = dir_acc / np.maximum(np.linalg.norm(dir_acc, axis=1, keepdims=True), 1e-12)
magnitude = np.maximum(mag_acc / nmodel, FLOOR)
coef = direction * (magnitude * ALPHA)[:, None]
traj = coef[:, None, :] * HOR[None, :, None]
traj = np.nan_to_num(traj, nan=0.0, posinf=0.0, neginf=0.0)

submission = pd.DataFrame({"id": test["id"].values})
for j, h in enumerate(range(1, 9)):
    submission[f"future_correction_dx_{h}"] = traj[:, j, 0]
    submission[f"future_correction_dy_{h}"] = traj[:, j, 1]
submission = submission[["id"] + OUTCOLS]
submission.to_csv(submission_out, index=False)
print("rows", len(submission), flush=True)
