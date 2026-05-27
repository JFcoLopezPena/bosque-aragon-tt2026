"""
SAM ViT-B fine-tuning para segmentación binaria de copas de árboles.
Congelado: image_encoder + prompt_encoder
Entrenable: mask_decoder únicamente
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
from segment_anything.modeling import Sam


class SegmentationSAM(nn.Module):
    """
    SAM ViT-B con mask_decoder entrenable para segmentación binaria copa/fondo.

    Entrada esperada en forward: [B, 3, 1024, 1024] en rango [0, 1].
    SAM normaliza internamente (equivalent to ImageNet normalization in [0,255] space).

    Salida: logits [B, 1, H, W] sin sigmoid.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        unfreeze_encoder_blocks: int = 0,
    ) -> None:
        super().__init__()

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint SAM no encontrado: {checkpoint_path}\n"
                "Descárgalo con:\n"
                "  Invoke-WebRequest https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -OutFile sam_vit_b_01ec64.pth"
            )

        print(f"[Model] Cargando SAM ViT-B desde {checkpoint_path} ...")
        self.sam: Sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint_path))

        self._freeze_all()
        self._unfreeze_mask_decoder()
        if unfreeze_encoder_blocks > 0:
            self._unfreeze_encoder_tail(unfreeze_encoder_blocks)
        self._print_param_summary()

    # ── Freeze / unfreeze ─────────────────────────────────────────────────────

    def _freeze_all(self) -> None:
        for p in self.sam.parameters():
            p.requires_grad = False

    def _unfreeze_mask_decoder(self) -> None:
        for p in self.sam.mask_decoder.parameters():
            p.requires_grad = True

    def _unfreeze_encoder_tail(self, n_blocks: int) -> None:
        """Descongela los ultimos n_blocks transformer blocks + neck del image encoder."""
        blocks = self.sam.image_encoder.blocks
        total  = len(blocks)
        for blk in blocks[max(0, total - n_blocks):]:
            for p in blk.parameters():
                p.requires_grad = True
        # neck = conv layers post-transformer
        for p in self.sam.image_encoder.neck.parameters():
            p.requires_grad = True
        print(f"[Model] Desbloqueados ultimos {n_blocks}/{total} bloques del encoder + neck")

    def _print_param_summary(self) -> None:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total     = trainable + frozen
        print(f"[Model] Entrenables : {trainable:,}  ({100 * trainable / total:.1f}%)")
        print(f"[Model] Congelados  : {frozen:,}  ({100 * frozen / total:.1f}%)")
        print(f"[Model] Total       : {total:,}")

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        images: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Args:
            images:      [B, 3, 1024, 1024] float32 en [0, 1]
            output_size: tamaño de salida (H, W). Si None → 256×256 (salida nativa)

        Returns:
            logits: [B, 1, H, W]
        """
        B      = images.shape[0]
        device = images.device

        # SAM normaliza con pixel_mean/pixel_std en espacio [0,255]
        images_255 = images * 255.0
        images_sam = self.sam.preprocess(images_255)  # normaliza + pad (pad=0 si ya es 1024)

        image_embeddings = self.sam.image_encoder(images_sam)  # [B, 256, 64, 64]

        sparse_emb, dense_emb = self.sam.prompt_encoder(
            points=None, boxes=None, masks=None
        )
        # dense_emb: [1, 256, 64, 64] → repetir para el batch
        dense_emb = dense_emb.expand(
            B, -1,
            image_embeddings.shape[-2],
            image_embeddings.shape[-1],
        )

        low_res_masks, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )  # [B, 1, 256, 256]

        if output_size is not None:
            logits = F.interpolate(low_res_masks, size=output_size,
                                   mode="bilinear", align_corners=False)
        else:
            logits = low_res_masks

        return logits  # [B, 1, H, W]

    # ── Utilidades ────────────────────────────────────────────────────────────

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def save_checkpoint(
        self,
        path: str | Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {"model_state_dict": self.state_dict()}
        if metadata:
            ckpt.update(metadata)
        torch.save(ckpt, path)
        print(f"[Model] Checkpoint guardado: {path}")

    @classmethod
    def load_checkpoint(
        cls,
        ft_checkpoint:  str | Path,
        sam_checkpoint: str | Path,
    ) -> "SegmentationSAM":
        ckpt = torch.load(ft_checkpoint, map_location="cpu", weights_only=False)
        n_unfreeze = ckpt.get("unfreeze_encoder_blocks", 0)
        model = cls(checkpoint_path=sam_checkpoint, unfreeze_encoder_blocks=n_unfreeze)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Model] Fine-tuned checkpoint cargado: {ft_checkpoint}")
        return model
