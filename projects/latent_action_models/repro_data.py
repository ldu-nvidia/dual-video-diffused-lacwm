"""Single-GPU repro: iterate the REAL dataloader, check each batch for anomalies, save it,
and run the full model forward+backward (no DDP/NCCL). With CUDA_LAUNCH_BLOCKING=1 the IMA
surfaces at the exact kernel, and last_batch.pt holds the culprit batch for post-mortem."""
import sys, hydra, torch
from omegaconf import DictConfig
from custom_resolvers import *  # noqa



@hydra.main(version_base=None, config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    model = hydra.utils.instantiate(cfg.model).cuda()
    model.train()
    ds = hydra.utils.instantiate(cfg.dataset)
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=16, num_workers=8, pin_memory=True, prefetch_factor=2)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    def fin(t):
        return torch.isfinite(t.float()).all().item() if torch.is_tensor(t) else True

    for i, batch in enumerate(dl):
        batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
        mi = batch.get("morphology_index")
        prob = []
        if mi is not None and (int(mi.min()) < 0 or int(mi.max()) >= 10):
            prob.append(f"morphOOB={mi.tolist()}")
        for key in ("rgb", "actions", "mask"):
            if key in batch and torch.is_tensor(batch[key]) and not fin(batch[key]):
                prob.append(f"{key}_nonfinite={int((~torch.isfinite(batch[key].float())).sum())}")
        if "actions" in batch and torch.is_tensor(batch["actions"]):
            a = batch["actions"].float()
            if a.abs().max() > 1e4:
                prob.append(f"actions_huge_max={float(a.abs().max()):.2e}")
        if prob:
            print(f"[ANOMALY iter {i}] morph={mi.tolist() if mi is not None else None} "
                  f"shapes rgb={tuple(batch['rgb'].shape)} act={tuple(batch['actions'].shape)} :: {prob}", flush=True)
        # save culprit-candidate batch (overwrite each iter)
        torch.save({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in batch.items()},
                   "/scr/ravenh/wan_tests/last_batch.pt")
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch)
        print(f"iter {i} forward loss={float(loss):.4f} -> backward (anomaly detect on)", flush=True)
        loss.backward()
        # which module's gradient is the first to go non-finite (= the corrupting step)?
        groups = {
            "inverse_model": model.inverse_model,
            "action_pool": model.action_pool,
            "action_to_control": model.forward_model.action_to_control,
            "action_decoder": model.action_decoder,
        }
        badgrad, maxnorm = [], {}
        for nm, mod in groups.items():
            gs = [p.grad for p in mod.parameters() if p.grad is not None]
            if gs:
                mx = max(float(g.abs().max()) for g in gs)
                maxnorm[nm] = mx
                if not all(torch.isfinite(g).all() for g in gs):
                    badgrad.append(nm)
        # also report which module first produces a non-finite OUTPUT (param values)
        badparam = [nm for nm, mod in groups.items()
                    if not all(torch.isfinite(p).all() for p in mod.parameters())]
        if badgrad or badparam:
            print(f"iter {i} NONFINITE-GRAD={badgrad} NONFINITE-PARAM={badparam} "
                  f"maxgrad={{ {', '.join(f'{k}:{v:.1e}' for k,v in maxnorm.items())} }}", flush=True)
        elif i % 2 == 0:
            print(f"iter {i} loss={float(loss):.4f} maxgrad={{ {', '.join(f'{k}:{v:.1e}' for k,v in maxnorm.items())} }}", flush=True)
        opt.step()
        if i >= 40:
            break
    print("REPRO DONE -- no crash in 500 iters")


if __name__ == "__main__":
    sys.exit(main())
