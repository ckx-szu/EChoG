import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision.models.video import r3d_18, R3D_18_Weights
class VideoInputAdapter(nn.Module):
    """Expand compact uint8 grayscale windows and normalize them directly on GPU."""

    def __init__(self, normalization: str) -> None:
        super().__init__()
        if normalization == "imagenet":
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        elif normalization == "kinetics":
            mean, std = (0.43216, 0.394666, 0.37645), (0.22803, 0.22145, 0.216989)
        else:
            raise ValueError(f"Unsupported video_normalization: {normalization}")
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1, 1), persistent=False)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        was_uint8 = video.dtype == torch.uint8
        needs_normalization = was_uint8 or video.shape[1] == 1
        video = video.float().div_(255.0) if was_uint8 else video.float()
        if video.shape[1] == 1:
            video = video.repeat(1, 3, 1, 1, 1)
        if needs_normalization:
            video = (video - self.mean) / self.std
        return video.contiguous(memory_format=torch.channels_last_3d)

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 7, stride=stride, padding=3, bias=False)
        self.norm1 = _group_norm(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 5, padding=2, bias=False)
        self.norm2 = _group_norm(out_channels)
        self.act = nn.GELU()
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                _group_norm(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + identity)

class ECGResNet1D(nn.Module):
    """Original lightweight ECG baseline."""

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, 15, stride=2, padding=7, bias=False),
            _group_norm(64),
            nn.GELU(),
            nn.MaxPool1d(5, stride=2, padding=2),
        )
        channels = (64, 128, 256, 512)
        stages = []
        in_channels = 64
        for stage_index, out_channels in enumerate(channels):
            stride = 1 if stage_index == 0 else 2
            stages.extend(
                [ResidualBlock1D(in_channels, out_channels, stride), ResidualBlock1D(out_channels, out_channels)]
            )
            in_channels = out_channels
        self.blocks = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(self.stem(x))
        return self.dropout(self.pool(x).flatten(1))

class VideoR3D18(nn.Module):
    """Original R3D-18 video baseline."""

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.3,
        weights_path: str | None = None,
        freeze_batchnorm_stats: bool = True,
    ) -> None:
        super().__init__()
        weights = R3D_18_Weights.KINETICS400_V1 if pretrained and not weights_path else None
        self.backbone = r3d_18(weights=weights)
        if weights_path:
            state = torch.load(weights_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.backbone.load_state_dict(state, strict=True)
        self.out_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        # Match the channels-last-3d input adapter before DDP constructs its
        # gradient buckets. This avoids the R3D Conv3d gradient-stride warning
        # and retains the faster Tensor Core layout on Ampere GPUs.
        self.backbone.to(memory_format=torch.channels_last_3d)
        self.dropout = nn.Dropout(dropout)
        self.freeze_batchnorm_stats = bool(freeze_batchnorm_stats)
        if self.freeze_batchnorm_stats:
            self._freeze_batchnorm_running_stats()

    def _freeze_batchnorm_running_stats(self) -> None:
        """Use the pretrained running statistics while keeping affine parameters trainable.

        R3D-18 contains BatchNorm3d throughout the backbone. Updating those
        statistics from two or three segments belonging to one patient makes
        train and evaluation behavior diverge, and under DDP gives every rank a
        different video encoder. Calling ``eval`` only on BatchNorm modules
        prevents running-stat updates without disabling gradients or the
        wrapper dropout.
        """
        for module in self.backbone.modules():
            if isinstance(module, _BatchNorm):
                module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_batchnorm_stats:
            self._freeze_batchnorm_running_stats()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.backbone(x))

def _mix_attention_and_extreme(
    attention_logits: torch.Tensor,
    extreme_logits: torch.Tensor,
    mix_logit: torch.Tensor,
    evidence_mode: str,
) -> torch.Tensor:
    if evidence_mode == "attention":
        return attention_logits
    if evidence_mode == "max":
        return extreme_logits
    if evidence_mode != "attention_max":
        raise ValueError(f"Unsupported aggregator_evidence: {evidence_mode}")
    gate = torch.sigmoid(mix_logit)
    return (1.0 - gate) * attention_logits + gate * extreme_logits

class FastSegmentAggregator(nn.Module):
    """Disease-specific attention over all segments covering one complete cine."""

    def __init__(
        self,
        feature_dim: int,
        position_dim: int,
        num_diseases: int,
        dropout: float,
        mode: str,
    ) -> None:
        super().__init__()
        if mode not in {"mean", "max", "attention"}:
            raise ValueError("fast_aggregation must be mean, max, or attention")
        self.mode = mode
        self.feature_score = (
            nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_diseases),
            )
            if mode == "attention"
            else None
        )
        self.position_score = (
            nn.Linear(position_dim, num_diseases, bias=False)
            if mode == "attention"
            else None
        )

    def forward(
        self,
        segment_logits: torch.Tensor,
        segment_features: torch.Tensor,
        segment_owner: torch.Tensor | None,
        segment_position: torch.Tensor | None,
        batch_size: int | None,
    ) -> torch.Tensor:
        if segment_owner is None:
            return segment_logits
        if batch_size is None:
            batch_size = int(segment_owner.max().item()) + 1
        scores = None
        if self.mode == "attention":
            scores = self.feature_score(segment_features)
            if segment_position is not None:
                scores = scores + self.position_score(segment_position)
        outputs = []
        for sample_index in range(batch_size):
            selected = segment_owner == sample_index
            logits = segment_logits[selected]
            if logits.numel() == 0:
                raise ValueError(f"No segments for DICOM sample {sample_index}")
            if self.mode == "mean":
                output = logits.mean(dim=0)
            elif self.mode == "max":
                output = logits.amax(dim=0)
            else:
                weights = torch.softmax(scores[selected], dim=0)
                output = torch.sum(weights * logits, dim=0)
            outputs.append(output)
        return torch.stack(outputs, dim=0)

def _mean_features_by_owner(
    features: torch.Tensor,
    owner: torch.Tensor | None,
    output_count: int,
) -> torch.Tensor:
    if owner is None:
        return features
    outputs = []
    for index in range(output_count):
        selected = features[owner == index]
        if selected.numel() == 0:
            raise ValueError(f"No segment features for DICOM {index}")
        outputs.append(selected.mean(dim=0))
    return torch.stack(outputs, dim=0)

class FastPatientAggregator(nn.Module):
    """Disease-wise DICOM MIL: one or more DICOMs -> one patient prediction."""

    def __init__(
        self,
        feature_dim: int,
        num_diseases: int,
        dropout: float,
        mode: str,
    ) -> None:
        super().__init__()
        if mode not in {"mean", "attention", "max", "attention_max"}:
            raise ValueError(
                "patient_mil_aggregation must be mean, attention, max, or attention_max"
            )
        self.mode = mode
        self.score = (
            nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_diseases),
            )
            if mode in {"attention", "attention_max"}
            else None
        )
        self.evidence_mix_logit = (
            nn.Parameter(torch.full((num_diseases,), -0.7))
            if mode == "attention_max"
            else None
        )

    def forward(
        self,
        dicom_logits: torch.Tensor,
        dicom_features: torch.Tensor,
        dicom_owner: torch.Tensor | None,
        batch_size: int | None,
    ) -> torch.Tensor:
        if dicom_owner is None:
            return dicom_logits
        if batch_size is None:
            batch_size = int(dicom_owner.max().item()) + 1
        scores = self.score(dicom_features) if self.score is not None else None
        outputs = []
        for patient_index in range(batch_size):
            selected = dicom_owner == patient_index
            logits = dicom_logits[selected]
            if logits.numel() == 0:
                raise ValueError(f"No DICOMs for patient bag {patient_index}")
            maximum = logits.amax(dim=0)
            if self.mode == "mean":
                output = logits.mean(dim=0)
            elif self.mode == "max":
                output = maximum
            else:
                weights = torch.softmax(scores[selected], dim=0)
                attention = torch.sum(weights * logits, dim=0)
                if self.mode == "attention":
                    output = attention
                else:
                    gate = torch.sigmoid(self.evidence_mix_logit)
                    output = (1.0 - gate) * attention + gate * maximum
            outputs.append(output)
        return torch.stack(outputs, dim=0)

class FastDICOMBaseline(nn.Module):
    """Full-cine segment -> DICOM -> patient hierarchical MIL baseline."""

    def __init__(
        self,
        modality: str,
        video_pretrained: bool,
        dropout: float,
        weights_path: str | None,
        video_normalization: str,
        segment_micro_batch: int,
        num_diseases: int,
        aggregation: str,
        position_dim: int,
        patient_aggregation: str,
        freeze_video_batchnorm_stats: bool,
    ) -> None:
        super().__init__()
        if modality not in {"echo", "ecg", "fusion"}:
            raise ValueError(f"Unsupported modality: {modality}")
        self.modality = modality
        self.segment_micro_batch = max(int(segment_micro_batch), 0)
        self.video_adapter = VideoInputAdapter(video_normalization) if modality in {"echo", "fusion"} else None
        self.video_encoder = (
            VideoR3D18(
                video_pretrained,
                dropout,
                weights_path,
                freeze_batchnorm_stats=freeze_video_batchnorm_stats,
            )
            if modality in {"echo", "fusion"}
            else None
        )
        self.ecg_encoder = ECGResNet1D(dropout) if modality in {"ecg", "fusion"} else None
        feature_dim = 1024 if modality == "fusion" else 512
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_diseases),
        )
        self.video_classifier = (
            nn.Sequential(nn.LayerNorm(512), nn.Dropout(dropout), nn.Linear(512, num_diseases))
            if modality == "fusion"
            else None
        )
        self.ecg_classifier = (
            nn.Sequential(nn.LayerNorm(512), nn.Dropout(dropout), nn.Linear(512, num_diseases))
            if modality == "fusion"
            else None
        )
        self.main_segment_aggregator = FastSegmentAggregator(
            feature_dim, position_dim, num_diseases, dropout, aggregation
        )
        self.video_segment_aggregator = (
            FastSegmentAggregator(512, position_dim, num_diseases, dropout, aggregation)
            if modality == "fusion"
            else None
        )
        self.ecg_segment_aggregator = (
            FastSegmentAggregator(512, position_dim, num_diseases, dropout, aggregation)
            if modality == "fusion"
            else None
        )
        self.main_patient_aggregator = FastPatientAggregator(
            feature_dim, num_diseases, dropout, patient_aggregation
        )
        self.video_patient_aggregator = (
            FastPatientAggregator(512, num_diseases, dropout, patient_aggregation)
            if modality == "fusion"
            else None
        )
        self.ecg_patient_aggregator = (
            FastPatientAggregator(512, num_diseases, dropout, patient_aggregation)
            if modality == "fusion"
            else None
        )

    def _encode(self, encoder: nn.Module, value: torch.Tensor) -> torch.Tensor:
        chunk = self.segment_micro_batch
        if chunk <= 0 or value.shape[0] <= chunk:
            return encoder(value)
        return torch.cat(
            [encoder(value[start : start + chunk]) for start in range(0, value.shape[0], chunk)],
            dim=0,
        )

    def forward(
        self,
        video=None,
        ecg=None,
        segment_owner=None,
        segment_position=None,
        dicom_owner=None,
        batch_size: int | None = None,
        return_aux: bool = False,
    ):
        video_features = ecg_features = None
        if self.video_encoder is not None:
            video_features = self._encode(self.video_encoder, self.video_adapter(video))
        if self.ecg_encoder is not None:
            ecg_features = self._encode(self.ecg_encoder, ecg)
        num_dicoms = int(dicom_owner.shape[0]) if dicom_owner is not None else int(batch_size)
        if self.modality == "echo":
            dicom_features = _mean_features_by_owner(video_features, segment_owner, num_dicoms)
            dicom_logits = self.main_segment_aggregator(
                self.classifier(video_features),
                video_features,
                segment_owner,
                segment_position,
                num_dicoms,
            )
            outputs = {
                "logits": self.main_patient_aggregator(
                    dicom_logits,
                    dicom_features,
                    dicom_owner,
                    batch_size,
                ),
                "dicom_logits": dicom_logits,
            }
        elif self.modality == "ecg":
            dicom_features = _mean_features_by_owner(ecg_features, segment_owner, num_dicoms)
            dicom_logits = self.main_segment_aggregator(
                self.classifier(ecg_features),
                ecg_features,
                segment_owner,
                segment_position,
                num_dicoms,
            )
            outputs = {
                "logits": self.main_patient_aggregator(
                    dicom_logits,
                    dicom_features,
                    dicom_owner,
                    batch_size,
                ),
                "dicom_logits": dicom_logits,
            }
        else:
            joint_features = torch.cat([video_features, ecg_features], dim=1)
            joint_dicom_features = _mean_features_by_owner(joint_features, segment_owner, num_dicoms)
            video_dicom_features = _mean_features_by_owner(video_features, segment_owner, num_dicoms)
            ecg_dicom_features = _mean_features_by_owner(ecg_features, segment_owner, num_dicoms)
            main_dicom_logits = self.main_segment_aggregator(
                self.classifier(joint_features),
                joint_features,
                segment_owner,
                segment_position,
                num_dicoms,
            )
            video_dicom_logits = self.video_segment_aggregator(
                self.video_classifier(video_features),
                video_features,
                segment_owner,
                segment_position,
                num_dicoms,
            )
            ecg_dicom_logits = self.ecg_segment_aggregator(
                self.ecg_classifier(ecg_features),
                ecg_features,
                segment_owner,
                segment_position,
                num_dicoms,
            )
            outputs = {
                "logits": self.main_patient_aggregator(
                    main_dicom_logits,
                    joint_dicom_features,
                    dicom_owner,
                    batch_size,
                ),
                "video_logits": self.video_patient_aggregator(
                    video_dicom_logits,
                    video_dicom_features,
                    dicom_owner,
                    batch_size,
                ),
                "ecg_logits": self.ecg_patient_aggregator(
                    ecg_dicom_logits,
                    ecg_dicom_features,
                    dicom_owner,
                    batch_size,
                ),
                "dicom_logits": main_dicom_logits,
            }
        return outputs if return_aux else outputs["logits"]

    def optimizer_parameter_groups(self, base_lr: float, video_lr_multiplier: float):
        return _optimizer_groups(self, self.video_encoder, base_lr, video_lr_multiplier)

def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def _optimizer_groups(model, video_encoder, base_lr: float, video_lr_multiplier: float):
    video_ids = {id(p) for p in video_encoder.parameters()} if video_encoder is not None else set()
    video_parameters, other_parameters = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (video_parameters if id(parameter) in video_ids else other_parameters).append(parameter)
    groups = []
    if video_parameters:
        groups.append(
            {
                "params": video_parameters,
                "lr": base_lr * video_lr_multiplier,
                "group_name": "video_backbone",
            }
        )
    if other_parameters:
        groups.append(
            {
                "params": other_parameters,
                "lr": base_lr,
                "group_name": "task_heads",
            }
        )
    return groups
