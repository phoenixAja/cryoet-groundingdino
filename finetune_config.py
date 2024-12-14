# Inherit base configuration from a pre-trained GroundingDINO model
_base_ = ["mmdet::grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py"]

# Set the root directory for all data files
data_root = "/content"

# Specify the language model to use for text processing
lang_model_name = "bert-base-uncased"

# Main model configuration dictionary
model = dict(
    # Specify model type as GroundingDINO, which combines visual and text processing
    type="GroundingDINO",
    # Number of object queries the model will predict
    num_queries=900,
    # Enable iterative bounding box refinement
    with_box_refine=True,
    # Use two-stage detection pipeline
    as_two_stage=True,
    # Configuration for input data preprocessing
    data_preprocessor=dict(
        # Type of preprocessor for detection tasks
        type="DetDataPreprocessor",
        # Mean values for image normalization (RGB)
        mean=[123.675, 116.28, 103.53],
        # Standard deviation values for image normalization
        std=[58.395, 57.12, 57.375],
        # Convert BGR to RGB color format
        bgr_to_rgb=True,
        # Disable mask padding
        pad_mask=False,
    ),
    # Language model configuration
    language_model=dict(
        # Use BERT model for text processing
        type="BertModel",
        # Name of the pre-trained BERT model to use
        name=lang_model_name,
        # Don't pad sequences to maximum length
        pad_to_max=False,
        # Use sub-sentence representations
        use_sub_sentence_represent=True,
        # Special tokens to handle in text processing
        special_tokens_list=["[CLS]", "[SEP]", ".", "?"],
        # Add pooling layer for sentence embeddings
        add_pooling_layer=True,
    ),
    # Visual backbone configuration (Swin Transformer)
    backbone=dict(
        # Use Swin Transformer architecture
        type="SwinTransformer",
        # Initial embedding dimensions
        embed_dims=96,
        # Number of transformer blocks in each stage
        depths=[2, 2, 6, 2],
        # Number of attention heads in each stage
        num_heads=[3, 6, 12, 24],
        # Size of local attention window
        window_size=7,
        # Multiplier for FFN layer dimension
        mlp_ratio=4,
        # Enable bias in attention
        qkv_bias=True,
        # Disable explicit scaling of attention
        qk_scale=None,
        # Dropout rate for regular connections
        drop_rate=0.0,
        # Dropout rate for attention
        attn_drop_rate=0.0,
        # Stochastic depth rate
        drop_path_rate=0.2,
        # Enable layer normalization in patches
        patch_norm=True,
        # Which stages to output features from
        out_indices=(1, 2, 3),
        # Disable gradient checkpointing
        with_cp=False,
        # Disable weight conversion
        convert_weights=False,
    ),
    # Neck configuration for feature processing
    neck=dict(
        # Use ChannelMapper to process backbone features
        type="ChannelMapper",
        # Input channel dimensions from backbone
        in_channels=[192, 384, 768],
        # Kernel size for convolutions
        kernel_size=1,
        # Output channel dimension
        out_channels=256,
        # Disable activation function
        act_cfg=None,
        # Enable bias in convolutions
        bias=True,
        # Group normalization configuration
        norm_cfg=dict(type="GN", num_groups=32),
        # Number of output feature levels
        num_outs=4,
    ),
    # Encoder configuration
    encoder=dict(
        # Number of encoder layers
        num_layers=6,
        # Visual processing configuration
        layer_cfg=dict(
            # Self-attention configuration for visual features
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            # Feed-forward network configuration
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
        ),
        # Text processing configuration
        text_layer_cfg=dict(
            # Self-attention for text features
            self_attn_cfg=dict(num_heads=4, embed_dims=256, dropout=0.0),
            # Feed-forward network for text
            ffn_cfg=dict(embed_dims=256, feedforward_channels=1024, ffn_drop=0.0),
        ),
        # Vision-language fusion configuration
        fusion_layer_cfg=dict(
            # Dimensions for visual features
            v_dim=256,
            # Dimensions for language features
            l_dim=256,
            # Embedding dimension for fusion
            embed_dim=1024,
            # Number of attention heads
            num_heads=4,
            # Initial values for layer scaling
            init_values=1e-4,
        ),
    ),
    # Decoder configuration
    decoder=dict(
        # Number of decoder layers
        num_layers=6,
        # Return intermediate decoder outputs
        return_intermediate=True,
        # Layer configuration
        layer_cfg=dict(
            # Self-attention for queries
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # Cross-attention for text features
            cross_attn_text_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # Cross-attention for image features
            cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # Feed-forward network configuration
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
        ),
        # Disable post-normalization
        post_norm_cfg=None,
    ),
    # Positional encoding configuration
    positional_encoding=dict(
        # Number of positional encoding dimensions
        num_feats=128,
        # Normalize positional encodings
        normalize=True,
        # Offset for encodings
        offset=0.0,
        # Temperature for scaling
        temperature=20,
    ),
    # Detection head configuration
    bbox_head=dict(
        # Use GroundingDINO head
        type="GroundingDINOHead",
        # Number of object classes to detect
        num_classes=6,
        # Synchronize classification average factor across GPUs
        sync_cls_avg_factor=True,
        # Contrastive learning configuration
        contrastive_cfg=dict(max_text_len=256),
        # Classification loss configuration
        # class_weight=[1.0, 3.0, 2.5, 2.5, 1.5, 3.0],  # Higher weight for underrepresented classes
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0
        ),
        # Bounding box regression loss
        loss_bbox=dict(type="L1Loss", loss_weight=5.0),
    ),
    # Denoising configuration
    dn_cfg=dict(
        # Scale for label noise
        label_noise_scale=0.5,
        # Scale for box coordinate noise
        box_noise_scale=1.0,
        # Group configuration for denoising
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100),
    ),
    # Training configuration
    train_cfg=dict(
        # Delete inherited training config
        _delete_=True,
        # Hungarian matching assigner configuration
        assigner=dict(
            type="HungarianAssigner",
            # Matching cost components
            match_costs=[
                dict(type="ClassificationCost", weight=2.0),
                dict(type="BBoxL1Cost", weight=5.0),
                dict(type="IoUCost", iou_mode="giou", weight=2.0),
            ],
        ),
    ),
    # Testing configuration
    test_cfg=dict(
        # Maximum detections per image
        max_per_img=300
    ),
)

# The rest of the configuration continues with data pipelines,
# optimizers, and training settings following the same pattern...
