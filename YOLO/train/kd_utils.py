import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG


# ──────────────────────────────────────────────────────────
# 1. KD Loss
# ──────────────────────────────────────────────────────────
class YOLO11KDLoss(nn.Module):
    """
    두 가지 포맷 자동 처리:

    [구버전 / old format]
      list of (B, 4*reg_max+nc, H, W)  ← scale 별 4D 텐서

    [YOLO11 new format]  ← 이번에 확인된 포맷
      dict {
        'boxes' : (B, 4*reg_max, N)   ← DFL logits, 전 scale 합쳐진 flat
        'scores': (B, nc, N)          ← cls logits, 전 scale 합쳐진 flat
        'feats' : list(len=3)         ← box-only backbone feats (KD엔 미사용)
      }

    component-wise KD:
      total = (1 - γ*w_box - α*w_cls - β*w_dfl) * task_loss
            + α*cls_kd + β*dfl_kd + γ*box_kd
    """

    def __init__(
        self,
        reg_max: int       = 16,
        temperature: float = 4.0,
        alpha: float       = 0.3,   # cls KD 비중
        beta: float        = 0.2,   # dfl KD 비중
        gamma: float       = 0.1,   # box KD 비중
    ):
        super().__init__()
        self.reg_max = reg_max
        self.box_ch  = 4 * reg_max
        self.T       = temperature
        self.alpha   = alpha
        self.beta    = beta
        self.gamma   = gamma

    def _decode_box(self, dfl_logits):
        """DFL logits → soft-argmax 좌표  (N, 4*reg_max) → (N, 4)"""
        N    = dfl_logits.shape[0]
        dist = dfl_logits.reshape(N, 4, self.reg_max)
        prob = F.softmax(dist, dim=-1)
        bins = torch.arange(self.reg_max, device=dfl_logits.device).float()
        return (prob * bins).sum(dim=-1)

    # ── YOLO11 flat dict 포맷 ───────────────────────────────
    def _compute_from_flat(self, s_dict, t_dict):
        """
        boxes : (B, 4*reg_max, N)  ← DFL logits
        scores: (B, nc, N)         ← cls logits
        """
        s_boxes  = s_dict['boxes']   # (B, 64, N)
        t_boxes  = t_dict['boxes']
        s_scores = s_dict['scores']  # (B, nc, N)
        t_scores = t_dict['scores']

        B, nc, N = s_scores.shape

        # cls KD : (B*N, nc)
        s_cls = s_scores.permute(0, 2, 1).reshape(-1, nc)
        t_cls = t_scores.permute(0, 2, 1).reshape(-1, nc)
        cls_kd = F.kl_div(
            F.log_softmax(s_cls / self.T, dim=-1),
            F.softmax(t_cls  / self.T, dim=-1),
            reduction='batchmean',
        ) * (self.T ** 2)

        # dfl KD : (B, 64, N) → (B*N*4, reg_max)
        s_dfl = s_boxes.permute(0, 2, 1).reshape(-1, self.reg_max)
        t_dfl = t_boxes.permute(0, 2, 1).reshape(-1, self.reg_max)
        dfl_kd = F.kl_div(
            F.log_softmax(s_dfl / self.T, dim=-1),
            F.softmax(t_dfl  / self.T, dim=-1),
            reduction='batchmean',
        ) * (self.T ** 2)

        # box KD : decoded 좌표 MSE  (B*N, 4)
        s_box = self._decode_box(s_boxes.permute(0, 2, 1).reshape(-1, self.box_ch))
        t_box = self._decode_box(t_boxes.permute(0, 2, 1).reshape(-1, self.box_ch))
        box_kd = F.mse_loss(s_box, t_box)

        return cls_kd, dfl_kd, box_kd

    # ── 구버전 4D list 포맷 ─────────────────────────────────
    def _compute_from_4d(self, student_preds, teacher_preds):
        cls_total = dfl_total = box_total = 0.0
        for s_pred, t_pred in zip(student_preds, teacher_preds):
            nc = s_pred.shape[1] - self.box_ch
            s_dfl = s_pred[:, :self.box_ch, :, :]
            t_dfl = t_pred[:, :self.box_ch, :, :]
            s_cls = s_pred[:, self.box_ch:, :, :]
            t_cls = t_pred[:, self.box_ch:, :, :]

            s_cls_flat = s_cls.permute(0, 2, 3, 1).reshape(-1, nc)
            t_cls_flat = t_cls.permute(0, 2, 3, 1).reshape(-1, nc)
            cls_kd = F.kl_div(
                F.log_softmax(s_cls_flat / self.T, dim=-1),
                F.softmax(t_cls_flat  / self.T, dim=-1),
                reduction='batchmean',
            ) * (self.T ** 2)

            s_dfl_flat = s_dfl.permute(0, 2, 3, 1).reshape(-1, self.reg_max)
            t_dfl_flat = t_dfl.permute(0, 2, 3, 1).reshape(-1, self.reg_max)
            dfl_kd = F.kl_div(
                F.log_softmax(s_dfl_flat / self.T, dim=-1),
                F.softmax(t_dfl_flat  / self.T, dim=-1),
                reduction='batchmean',
            ) * (self.T ** 2)

            s_box = self._decode_box(s_dfl.permute(0, 2, 3, 1).reshape(-1, self.box_ch))
            t_box = self._decode_box(t_dfl.permute(0, 2, 3, 1).reshape(-1, self.box_ch))
            box_kd = F.mse_loss(s_box, t_box)

            cls_total += cls_kd
            dfl_total += dfl_kd
            box_total += box_kd

        n = len(student_preds)
        return cls_total / n, dfl_total / n, box_total / n

    # ── 포맷 자동 dispatch ──────────────────────────────────
    def compute_kd(self, s_raw, t_raw):
        if isinstance(s_raw, dict) and 'boxes' in s_raw:
            return self._compute_from_flat(s_raw, t_raw)
        return self._compute_from_4d(s_raw, t_raw)

    def forward(self, student_preds, teacher_preds):
        cls_kd, dfl_kd, box_kd = self.compute_kd(student_preds, teacher_preds)
        return self.alpha * cls_kd + self.beta * dfl_kd + self.gamma * box_kd


# ──────────────────────────────────────────────────────────
# 2. KD Trainer
# ──────────────────────────────────────────────────────────
class YOLO11KDTrainer(DetectionTrainer):
    """
    학습 흐름:
      training loop
        → model(batch_dict) → model.loss(batch)
        → model.forward(batch["img"])  ← s_hook 발동 → _s_raw 저장
        → model.criterion(preds, batch) ← 패치된 kd_criterion 호출
          → teacher(batch["img"]) → t_hook 발동 → _t_raw 저장
          → kd_fn.compute_kd(_s_raw, _t_raw) → KD loss 계산
    """

    def __init__(
        self,
        teacher_model,
        kd_loss_fn,
        cfg=DEFAULT_CFG,
        overrides=None,
        _callbacks=None,
    ):
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self.teacher    = teacher_model
        self.kd_loss_fn = kd_loss_fn

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self._s_raw   = None   # student Detect head 출력 (raw)
        self._t_raw   = None   # teacher Detect head 출력 (raw, detached)
        self._kd_step = 0

    # ── setup ──────────────────────────────────────────────
    def _setup_train(self, *args, **kwargs):
        super()._setup_train(*args, **kwargs)
        self.teacher = self.teacher.to(self.device)
        self._register_hooks()
        self._patch_criterion()

    # ── Detect head 탐색 ───────────────────────────────────
    @staticmethod
    def _find_detect_head(model):
        detect_names = {'Detect', 'v10Detect', 'OBB', 'Segment', 'Pose'}
        for m in reversed(list(model.model)):
            if type(m).__name__ in detect_names:
                return m
        return model.model[-1]

    # ── forward hook 등록 ──────────────────────────────────
    def _register_hooks(self):
        trainer  = self
        s_detect = self._find_detect_head(self.model)
        t_detect = self._find_detect_head(self.teacher)
        print(f"[KD] Student Detect: {type(s_detect).__name__}")
        print(f"[KD] Teacher Detect: {type(t_detect).__name__}")

        def s_hook(module, inp, out):
            # YOLO11 new: dict with 'boxes', 'scores'
            if isinstance(out, dict) and 'boxes' in out:
                trainer._s_raw = out   # gradient 유지
            # old: list/tuple of 4D tensors
            elif isinstance(out, (list, tuple)):
                if all(torch.is_tensor(x) and x.ndim == 4 for x in out):
                    trainer._s_raw = list(out)

        def t_hook(module, inp, out):
            raw = None
            # eval: (decoded_tensor, dict_or_list)
            if isinstance(out, tuple) and len(out) == 2:
                second = out[1]
                if isinstance(second, dict) and 'boxes' in second:
                    # tensor 값만 detach, 나머지(list 등)는 그대로
                    raw = {k: (v.detach() if torch.is_tensor(v) else v)
                           for k, v in second.items()}
                elif isinstance(second, (list, tuple)):
                    if all(torch.is_tensor(x) and x.ndim == 4 for x in second):
                        raw = [f.detach() for f in second]
            # old training: list of 4D tensors
            elif isinstance(out, (list, tuple)):
                if all(torch.is_tensor(x) and x.ndim == 4 for x in out):
                    raw = [f.detach() for f in out]
            trainer._t_raw = raw

        self._hook_s = s_detect.register_forward_hook(s_hook)
        self._hook_t = t_detect.register_forward_hook(t_hook)

    # ── model.criterion 패치 ───────────────────────────────
    def _patch_criterion(self):
        model    = self.model
        teacher  = self.teacher
        kd_fn    = self.kd_loss_fn
        trainer  = self

        if not hasattr(model, 'criterion') or model.criterion is None:
            model.criterion = model.init_criterion()
        orig_crit = model.criterion

        def kd_criterion(preds, batch):
            # [1] task loss (student forward는 이미 완료 → s_hook 발동됨)
            task_loss, task_loss_items = orig_crit(preds, batch)

            # task_loss가 scalar가 아닐 경우 (ultralytics 버전 차이) sum으로 정규화
            if isinstance(task_loss, torch.Tensor) and task_loss.numel() > 1:
                task_loss = task_loss.sum()

            # [2] teacher forward → t_hook 발동
            with torch.no_grad():
                teacher(batch["img"])

            s_raw = trainer._s_raw
            t_raw = trainer._t_raw

            # 캡처 실패 시 task loss만 반환
            if s_raw is None or t_raw is None:
                if trainer._kd_step < 3:
                    print(f"[KD] FAIL  s={type(s_raw).__name__ if s_raw is not None else None}, "
                          f"t={type(t_raw).__name__ if t_raw is not None else None}")
                trainer._kd_step += 1
                return task_loss, task_loss_items

            # [3] component-wise KD
            cls_kd, dfl_kd, box_kd = kd_fn.compute_kd(s_raw, t_raw)

            # task_loss_items shape 정규화: (1,3) → (3,) 등 대비
            items    = task_loss_items.detach().float().flatten()  # 항상 1D (3,)
            item_sum = items.sum().clamp(min=1e-6)
            w_box = (items[0] / item_sum).item()
            w_cls = (items[1] / item_sum).item()
            w_dfl = (items[2] / item_sum).item()

            a, b, g = kd_fn.alpha, kd_fn.beta, kd_fn.gamma
            scale   = 1.0 - g * w_box - a * w_cls - b * w_dfl
            total   = scale * task_loss + a * cls_kd + b * dfl_kd + g * box_kd

            if trainer._kd_step < 5 or trainer._kd_step % 200 == 0:
                print(
                    f"[KD {trainer._kd_step:4d}] "
                    f"task={task_loss.item():.4f}  "
                    f"cls_kd={cls_kd.item():.4f}  "
                    f"dfl_kd={dfl_kd.item():.4f}  "
                    f"box_kd={box_kd.item():.4f}  "
                    f"scale={scale:.3f}  "
                    f"total={total.item():.4f}"
                )
            trainer._kd_step += 1
            return total, task_loss_items

        model.criterion = kd_criterion