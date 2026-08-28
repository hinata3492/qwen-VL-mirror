# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Qwen3VLForConditionalGeneration

from config import (
    MODEL_NAME,
    QWEN_HIDDEN_DIM,
    SEG_HIDDEN_DIM,
)


# ============================================================
# Segmentation Head
# ============================================================

class MirrorSegHead(nn.Module):

    def __init__(
        self,
        visual_dim=4096,
        semantic_dim=4096,
        hidden_dim=256,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Visual feature projection
        # ----------------------------------------------------
        #
        # [B, 4096, H, W]
        #        ↓
        # [B, 256, H, W]
        #

        self.visual_proj = nn.Conv2d(
            visual_dim,
            hidden_dim,
            kernel_size=1,
        )

        # ----------------------------------------------------
        # Semantic feature projection
        # ----------------------------------------------------
        #
        # [B, 4096]
        #    ↓
        # [B, 256]
        #

        self.semantic_proj = nn.Linear(
            semantic_dim,
            hidden_dim,
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------
        #
        # visual 256ch
        # semantic 256ch
        #
        # concat
        #   ↓
        # 512ch
        #

        self.decoder = nn.Sequential(

            nn.Conv2d(
                hidden_dim * 2,
                hidden_dim,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                hidden_dim,
                hidden_dim // 2,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                hidden_dim // 2,
                1,
                kernel_size=1,
            ),
        )

    def forward(
        self,
        visual_feature,
        semantic_feature,
        output_size,
    ):
        """
        visual_feature:
            [B, 4096, H', W']

        semantic_feature:
            [B, 4096]

        output:
            [B, 1, H, W]
        """

        # ====================================================
        # Visual projection
        # ====================================================

        visual = self.visual_proj(
            visual_feature
        )

        # [B, 256, H', W']

        # ====================================================
        # Semantic projection
        # ====================================================

        semantic = self.semantic_proj(
            semantic_feature
        )

        # [B, 256]

        # 空間次元を追加
        semantic = semantic[
            :,
            :,
            None,
            None,
        ]

        # [B, 256, 1, 1]

        # Visual featureと同じ空間サイズへ展開
        semantic = semantic.expand(
            -1,
            -1,
            visual.shape[-2],
            visual.shape[-1],
        )

        # [B, 256, H', W']

        # ====================================================
        # Fusion
        # ====================================================

        fused = torch.cat(
            [
                visual,
                semantic,
            ],
            dim=1,
        )

        # [B, 512, H', W']

        # ====================================================
        # Decoder
        # ====================================================

        logits = self.decoder(
            fused
        )

        # [B, 1, H', W']

        # ====================================================
        # Upsampling
        # ====================================================

        logits = F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        # [B, 1, H, W]

        return logits


# ============================================================
# Qwen3-VL + Segmentation
# ============================================================

class QwenMirrorSegmentation(nn.Module):

    def __init__(self):
        super().__init__()

        # ====================================================
        # Qwen3-VL
        # ====================================================

        print(
            f"Loading Qwen model: "
            f"{MODEL_NAME}"
        )

        self.qwen = (
            Qwen3VLForConditionalGeneration
            .from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        )

        # ====================================================
        # Freeze Qwen
        # ====================================================

        for param in self.qwen.parameters():

            param.requires_grad = False

        self.qwen.eval()

        # ====================================================
        # Segmentation Head
        # ====================================================

        self.seg_head = MirrorSegHead(
            visual_dim=QWEN_HIDDEN_DIM,
            semantic_dim=QWEN_HIDDEN_DIM,
            hidden_dim=SEG_HIDDEN_DIM,
        )


    # ========================================================
    # Feature Extraction
    # ========================================================

    def extract_features(
        self,
        inputs,
        target_image_index=0,
    ):
        """
        Qwen3-VLから

            1. target image spatial feature
            2. multimodal semantic feature

        を取り出す。

        Qwen本体はfreezeなのでgradientは不要。
        """

        # ====================================================
        # Qwen forward
        # ====================================================

        with torch.no_grad():

            outputs = self.qwen(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # 最終hidden state
        hidden = outputs.hidden_states[-1]

        # shape:
        # [B, sequence_length, 4096]

        # ====================================================
        # Input information
        # ====================================================

        input_ids = (
            inputs["input_ids"]
        )

        image_grid_thw = (
            inputs["image_grid_thw"]
        )

        image_token_id = (
            self.qwen
            .config
            .image_token_id
        )

        # ====================================================
        # Semantic feature
        # ====================================================
        #
        # 最後のtokenは
        #
        # Image1
        # Image2
        # Prompt
        #
        # より後ろに存在するため、
        # causal attention上、
        # それらを見た表現になる。
        #

        semantic_feature = hidden[
            :,
            -1,
            :
        ]

        # [B, 4096]

        # ====================================================
        # Target image spatial feature
        # ====================================================

        batch_features = []

        batch_size = (
            input_ids.shape[0]
        )

        for b in range(
            batch_size
        ):

            # ------------------------------------------------
            # 全image tokenの位置
            # ------------------------------------------------

            image_positions = (
                input_ids[b]
                == image_token_id
            ).nonzero(
                as_tuple=False
            ).squeeze(-1)

            # ------------------------------------------------
            # Target imageのgrid情報
            # ------------------------------------------------
            #
            # 例:
            #
            # image_grid_thw:
            #
            # [[1, 30, 40],
            #  [1, 30, 40]]
            #
            # 1枚目:
            # T=1
            # H=30
            # W=40
            #

            grid_t, grid_h, grid_w = (
                image_grid_thw[
                    target_image_index
                ].tolist()
            )

            # 今回は静止画のみ
            if grid_t != 1:

                raise ValueError(
                    "Currently only static "
                    "images are supported. "
                    f"grid_t={grid_t}"
                )

            # ------------------------------------------------
            # Qwen3-VL 2x2 Patch Merger
            # ------------------------------------------------

            merged_h = (
                grid_h // 2
            )

            merged_w = (
                grid_w // 2
            )

            # 例:
            #
            # 30 x 40
            #     ↓
            # 15 x 20
            #
            # = 300 image tokens

            tokens_per_image = (
                grid_t
                * merged_h
                * merged_w
            )

            # ------------------------------------------------
            # Target imageより前にある画像token数
            # ------------------------------------------------

            start = 0

            for i in range(
                target_image_index
            ):

                t, h, w = (
                    image_grid_thw[i]
                    .tolist()
                )

                start += (
                    t
                    * (h // 2)
                    * (w // 2)
                )

            end = (
                start
                + tokens_per_image
            )

            # ------------------------------------------------
            # Target image token position
            # ------------------------------------------------

            positions = (
                image_positions[
                    start:end
                ]
            )

            # ------------------------------------------------
            # Target image tokens
            # ------------------------------------------------

            target_tokens = hidden[
                b,
                positions,
                :
            ]

            # shape:
            # [N, 4096]

            expected_tokens = (
                grid_t
                * merged_h
                * merged_w
            )

            # ------------------------------------------------
            # Safety check
            # ------------------------------------------------

            if (
                target_tokens.shape[0]
                != expected_tokens
            ):

                raise RuntimeError(
                    "Token count mismatch: "
                    f"got "
                    f"{target_tokens.shape[0]}, "
                    f"expected "
                    f"{expected_tokens}"
                )

            # ------------------------------------------------
            # Token sequence
            #
            # [N, C]
            #
            # ↓
            #
            # spatial feature
            #
            # [H', W', C]
            # ------------------------------------------------

            target_feature = (
                target_tokens.reshape(
                    merged_h,
                    merged_w,
                    hidden.shape[-1],
                )
            )

            # ------------------------------------------------
            # [H', W', C]
            #       ↓
            # [C, H', W']
            # ------------------------------------------------

            target_feature = (
                target_feature.permute(
                    2,
                    0,
                    1,
                )
            )

            target_feature = (
                target_feature.contiguous()
            )

            batch_features.append(
                target_feature
            )

        # ====================================================
        # Batch
        # ====================================================

        visual_feature = (
            torch.stack(
                batch_features,
                dim=0,
            )
        )

        # [B, 4096, H', W']

        # ====================================================
        # dtype
        # ====================================================
        #
        # Qwen:
        # bfloat16
        #
        # Seg Head:
        # float32
        #

        visual_feature = (
            visual_feature.float()
        )

        semantic_feature = (
            semantic_feature.float()
        )

        return (
            visual_feature,
            semantic_feature,
        )


    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        inputs,
        target_image_index=0,
        output_size=(480, 640),
    ):
        """
        通常のforward。

        test_model.pyなどでは
        今まで通り、

            model(inputs)

        と呼べる。

        train.pyのcache版では、
        extract_features() と
        seg_head() を分離して使う。
        """

        # ====================================================
        # Qwen feature extraction
        # ====================================================

        (
            visual_feature,
            semantic_feature,
        ) = self.extract_features(
            inputs=inputs,
            target_image_index=target_image_index,
        )

        # ====================================================
        # Segmentation
        # ====================================================

        logits = self.seg_head(
            visual_feature=visual_feature,
            semantic_feature=semantic_feature,
            output_size=output_size,
        )

        return logits