"""
ClassificationSAM: SAM ViT-B fine-tuneado para clasificacion fitosanitaria (SANO/ENFERMO).

Arquitectura dual:
  - logits_cls  [B, 2]        -> clasificacion sano/enfermo
  - logits_mask [B, 1, H, W]  -> segmentacion de copa (preserva calidad R1)

Transferencia de conocimiento:
  - Image Encoder: congelado excepto ultimos 2 bloques + neck
  - Prompt Encoder: congelado
  - Mask Decoder: entrenable, inicializado desde checkpoint R1
  - Classification Head: entrenable (inicializado aleatoriamente)
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry


class ClassificationHead(nn.Module):
    def __init__(self, in_features: int = 256, hidden: int = 128,
                 num_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]  ->  [B, num_classes]
        return self.fc(self.pool(x))


class ClassificationSAM(nn.Module):
    def __init__(
        self,
        sam_checkpoint: str | Path,
        r1_checkpoint:  str | Path | None = None,
        unfreeze_encoder_blocks: int = 2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        # ── SAM base ──────────────────────────────────────────────────────────
        self.sam = sam_model_registry["vit_b"](checkpoint=str(sam_checkpoint))

        # ── Transfer decoder weights from R1 ──────────────────────────────────
        if r1_checkpoint is not None:
            self._load_r1_decoder(Path(r1_checkpoint))

        # ── Classification head (over image encoder output: 256 channels) ─────
        self.classification_head = ClassificationHead(
            in_features=256,
            hidden=128,
            num_classes=num_classes,
        )

        # ── Freezing strategy ─────────────────────────────────────────────────
        # 1. Freeze all SAM parameters
        for param in self.sam.parameters():
            param.requires_grad = False

        # 2. Unfreeze mask decoder (initialised from R1)
        for param in self.sam.mask_decoder.parameters():
            param.requires_grad = True

        # 3. Unfreeze last N encoder blocks + neck
        if unfreeze_encoder_blocks > 0:
            self._unfreeze_encoder_tail(unfreeze_encoder_blocks)
        # Classification head is trainable by default (freshly initialised)

        self._print_param_summary()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _load_r1_decoder(self, r1_checkpoint: Path) -> None:
        ckpt       = torch.load(str(r1_checkpoint), map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        prefix     = "sam.mask_decoder."
        decoder_sd = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if decoder_sd:
            self.sam.mask_decoder.load_state_dict(decoder_sd, strict=True)
            print("[Model] Decoder R1 transferido correctamente.")
        else:
            print("[Model] WARN: No se encontraron pesos del decoder en r1_checkpoint.")

    def _unfreeze_encoder_tail(self, n_blocks: int) -> None:
        blocks = list(self.sam.image_encoder.blocks)
        for block in blocks[-n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in self.sam.image_encoder.neck.parameters():
            param.requires_grad = True

    def _print_param_summary(self) -> None:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] Parametros totales    : {total:,}")
        print(f"[Model] Entrenables           : {trainable:,}  ({100*trainable/total:.1f}%)")
        print(f"[Model] Congelados            : {total - trainable:,}")

    # ── Optimizer param groups ────────────────────────────────────────────────

    def get_param_groups(self, lr: float) -> list[dict]:
        cls_params     = list(self.classification_head.parameters())
        cls_ids        = {id(p) for p in cls_params}
        decoder_params = list(self.sam.mask_decoder.parameters())
        decoder_ids    = {id(p) for p in decoder_params}
        encoder_tail   = [
            p for p in self.sam.image_encoder.parameters()
            if p.requires_grad and id(p) not in cls_ids | decoder_ids
        ]
        groups = [
            {"params": cls_params,     "lr": lr * 2,   "name": "cls_head"},
            {"params": decoder_params, "lr": lr * 1,   "name": "decoder"},
        ]
        if encoder_tail:
            groups.append({"params": encoder_tail, "lr": lr * 0.1, "name": "encoder_tail"})
        return groups

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        images:      torch.Tensor,              # [B, 3, 1024, 1024] in [0, 1]
        output_size: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = images.shape[0]

        # SAM expects [0, 255] and applies its own normalization
        images_sam       = self.sam.preprocess(images * 255.0)
        image_embeddings = self.sam.image_encoder(images_sam)  # [B, 256, 64, 64]

        # Classification branch
        logits_cls = self.classification_head(image_embeddings)  # [B, 2]

        # Segmentation branch (keeps decoder active to preserve R1 quality)
        sparse_emb, dense_emb = self.sam.prompt_encoder(points=None, boxes=None, masks=None)
        dense_emb = dense_emb.expand(B, -1, image_embeddings.shape[-2], image_embeddings.shape[-1])
        low_res_masks, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )

        if output_size is not None:
            logits_mask = F.interpolate(low_res_masks, size=output_size,
                                        mode="bilinear", align_corners=False)
        else:
            logits_mask = low_res_masks

        return logits_cls, logits_mask  # [B, 2], [B, 1, H, W]

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def save_checkpoint(self, path: str | Path, metadata: dict | None = None) -> None:
        torch.save(
            {"model_state_dict": self.state_dict(), "metadata": metadata or {}},
            str(path),
        )

    @classmethod
    def load_checkpoint(
        cls,
        ft_checkpoint:           str | Path,
        sam_checkpoint:          str | Path,
        unfreeze_encoder_blocks: int = 2,
        num_classes:             int = 2,
    ) -> "ClassificationSAM":
        ckpt  = torch.load(str(ft_checkpoint), map_location="cpu", weights_only=False)
        model = cls(
            sam_checkpoint=sam_checkpoint,
            r1_checkpoint=None,   # fine-tuned weights loaded below
            unfreeze_encoder_blocks=unfreeze_encoder_blocks,
            num_classes=num_classes,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        return model
