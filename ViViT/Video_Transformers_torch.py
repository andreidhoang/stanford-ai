#!/usr/bin/env python
"""
PyTorch port of the video transformer tutorial.

This script is a direct refactor of the Keras‐based notebook provided in
``Video_Transformers.ipynb``.  It uses PyTorch and torchvision instead of
TensorFlow/Keras to implement the same functionality.  The overall flow
remains identical:

1.  Download and prepare the UCF101 subset used for training/testing.
2.  Optionally compute video frame features using a pre–trained DenseNet121.
3.  Build a Transformer‐based classifier for sequence data, incorporating
    positional embeddings and multi–head attention.
4.  Train the model on preprocessed data and evaluate its performance.
5.  Provide utilities for running inference on a single video and creating
    GIF visualisations.

Note that this script does **not** execute training by default.  It is
intended to show how one could implement the same model architecture and
training loop in PyTorch.  Running the full pipeline requires downloading
the dataset and may take considerable time.
"""

import os
import cv2
import imageio
import numpy as np
import pandas as pd
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import torchvision
from torchvision import transforms


# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
# These values mirror those defined in the original Keras notebook.  You can
# modify them if you'd like to experiment with different sequence lengths or
# image sizes.
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 1024
IMG_SIZE = 128
EPOCHS = 5


# -----------------------------------------------------------------------------
# Data loading and preprocessing
# -----------------------------------------------------------------------------
def crop_center(frame: np.ndarray) -> np.ndarray:
    """Center–crop a frame to IMG_SIZE × IMG_SIZE.

    Args:
        frame: A single video frame as a NumPy array of shape (H, W, C).

    Returns:
        The cropped frame as a NumPy array of shape (IMG_SIZE, IMG_SIZE, C).
    """
    h, w, _ = frame.shape
    top = max((h - IMG_SIZE) // 2, 0)
    left = max((w - IMG_SIZE) // 2, 0)
    bottom = top + IMG_SIZE
    right = left + IMG_SIZE
    return frame[top:bottom, left:right, :]


def load_video(path: str, max_frames: int = 0) -> np.ndarray:
    """Load a video file and return a sequence of cropped frames.

    Args:
        path: Path to the video file.
        max_frames: Optionally limit the number of frames returned.  A value
            of 0 means all frames are returned.

    Returns:
        A NumPy array of shape (frames, IMG_SIZE, IMG_SIZE, 3) containing
        RGB frames cropped to the centre.
    """
    cap = cv2.VideoCapture(path)
    frames: List[np.ndarray] = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # OpenCV loads images in BGR; convert to RGB and crop
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cropped = crop_center(frame)
            frames.append(cropped)
            if 0 < max_frames <= len(frames):
                break
    finally:
        cap.release()
    return np.asarray(frames)


def build_feature_extractor() -> nn.Module:
    """Create a pre–trained DenseNet121 feature extractor.

    The final classification layer is removed and the network is set to
    evaluation mode.  A preprocessing transform is defined to mirror
    ``keras.applications.densenet.preprocess_input``.

    Returns:
        A PyTorch module that takes a tensor of shape (N, 3, H, W) and
        outputs features of shape (N, NUM_FEATURES).
    """
    model = torchvision.models.densenet121(weights="IMAGENET1K_V1")
    # Remove the classifier and pooling; DenseNet returns (batch, 1024, 1, 1)
    # from its features attribute.  We'll flatten this to (batch, 1024).
    feature_extractor = model.features
    feature_extractor.eval()
    # Define preprocessing consistent with Keras DenseNet.  PyTorch models
    # typically expect inputs in [0,1] range and normalised by mean/std.
    # The following transform approximates keras.applications.densenet.preprocess_input.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Normalize(mean=mean, std=std),
    ])
    return feature_extractor, preprocess


@torch.no_grad()
def extract_features(frames: np.ndarray, feature_extractor: nn.Module, preprocess) -> np.ndarray:
    """Extract DenseNet121 features from an array of frames.

    Args:
        frames: A NumPy array of shape (frames, H, W, 3) representing RGB frames.
        feature_extractor: The pre–trained DenseNet feature extractor.
        preprocess: A torchvision transform to apply to each frame.

    Returns:
        A NumPy array of shape (frames, NUM_FEATURES) with extracted features.
    """
    # Prepare a batch of frames
    tensors = torch.stack([preprocess(frame) for frame in frames], dim=0)  # (F, 3, H, W)
    features = feature_extractor(tensors)
    # Flatten: DenseNet features have shape (F, C, h, w), where h = w = 1
    features = torch.flatten(features, start_dim=1)  # (F, C)
    return features.cpu().numpy()


def prepare_all_videos(df: pd.DataFrame, root_dir: str,
                        feature_extractor: nn.Module,
                        preprocess) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare feature tensors and label arrays for all videos in a DataFrame.

    This function mirrors ``prepare_all_videos`` in the Keras notebook.  It
    iterates over each row in ``df``, loads the video frames, pads them to
    ``MAX_SEQ_LENGTH`` if necessary, and extracts DenseNet features frame
    by frame.  The resulting 3D tensor has shape
    ``(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES)``.

    Args:
        df: A pandas DataFrame with columns ``video_name`` and ``tag``.
        root_dir: Directory containing the video files.
        feature_extractor: The DenseNet feature extractor module.
        preprocess: The preprocessing transform for input frames.

    Returns:
        A tuple ``(frame_features, labels)`` where ``frame_features`` is a
        NumPy array of shape ``(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES)`` and
        ``labels`` is a 1D array of class indices.
    """
    video_paths = df["video_name"].tolist()
    class_names = df["tag"].tolist()
    # Build a label encoder: unique class names sorted alphabetically
    unique_classes = sorted(set(class_names))
    class_to_index = {name: idx for idx, name in enumerate(unique_classes)}
    labels = np.array([class_to_index[name] for name in class_names], dtype=np.int64)

    num_samples = len(video_paths)
    frame_features = np.zeros((num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype=np.float32)
    for idx, rel_path in enumerate(video_paths):
        frames = load_video(os.path.join(root_dir, rel_path))
        # Pad or crop sequence length
        if len(frames) < MAX_SEQ_LENGTH:
            pad_len = MAX_SEQ_LENGTH - len(frames)
            padding = np.zeros((pad_len, IMG_SIZE, IMG_SIZE, 3), dtype=frames.dtype)
            frames = np.concatenate([frames, padding], axis=0)
        elif len(frames) > MAX_SEQ_LENGTH:
            frames = frames[:MAX_SEQ_LENGTH]
        # Extract features per frame
        feats = extract_features(frames, feature_extractor, preprocess)
        # If the video was shorter than MAX_SEQ_LENGTH, the features array may
        # contain zeros for the padded frames.  To preserve this behaviour we
        # explicitly pad the feature vectors.
        if feats.shape[0] < MAX_SEQ_LENGTH:
            feat_pad = np.zeros((MAX_SEQ_LENGTH - feats.shape[0], NUM_FEATURES), dtype=np.float32)
            feats = np.concatenate([feats, feat_pad], axis=0)
        frame_features[idx] = feats
    return frame_features, labels


# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------
class PositionalEmbedding(nn.Module):
    """Add learnable positional embeddings to frame features.

    This module mimics the `PositionalEmbedding` Keras layer.  It creates an
    ``nn.Embedding`` with ``MAX_SEQ_LENGTH`` positions and adds it to the
    incoming frame features.  A ``compute_mask`` method is also provided to
    generate a mask indicating which positions are valid (non–padded).
    """

    def __init__(self, sequence_length: int, embed_dim: int) -> None:
        super().__init__()
        self.position_embeddings = nn.Embedding(sequence_length, embed_dim)
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, embed_dim)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        embedded_positions = self.position_embeddings(positions)  # (seq_len, embed_dim)
        return x + embedded_positions.unsqueeze(0)

    def compute_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return a boolean mask where ``True`` indicates a padded position.

        In the original Keras code, padding frames were represented by all
        zero values.  We maintain the same convention here: any frame
        embedding that sums to zero is considered padded.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            A boolean tensor of shape (batch_size, seq_len) where ``True``
            indicates the position should be masked (i.e. ignored during
            attention).
        """
        # Identify padded frames where all features are zero
        mask = (x.abs().sum(dim=-1) == 0)
        return mask


class TransformerEncoder(nn.Module):
    """A simple Transformer encoder block.

    This module matches the behaviour of the `TransformerEncoder` layer in the
    Keras example.  It consists of a multi–head self–attention layer
    followed by a small feed–forward network (two linear layers with GELU
    activation) and layer normalisation.
    """

    def __init__(self, embed_dim: int, dense_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.3,
            batch_first=True,
        )
        self.dense_proj = nn.Sequential(
            nn.Linear(embed_dim, dense_dim),
            nn.GELU(),
            nn.Linear(dense_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x shape: (batch, seq_len, embed_dim)
        # mask shape: (batch, seq_len) where True indicates padding positions
        # In PyTorch MultiheadAttention, ``key_padding_mask`` uses True to mask
        attn_output, _ = self.attention(x, x, x, key_padding_mask=mask)
        out1 = self.norm1(x + attn_output)
        proj_output = self.dense_proj(out1)
        return self.norm2(out1 + proj_output)


class VideoTransformerClassifier(nn.Module):
    """Transformer–based classifier for video sequences.

    The architecture follows the original Keras implementation: positional
    embeddings are added to the inputs, followed by a single Transformer
    encoder block, global max pooling, dropout, and a final linear layer for
    classification.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.embed_dim = NUM_FEATURES
        self.positional_embedding = PositionalEmbedding(MAX_SEQ_LENGTH, self.embed_dim)
        self.transformer = TransformerEncoder(
            embed_dim=self.embed_dim, dense_dim=4, num_heads=1
        )
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, embed_dim)
        x = self.positional_embedding(x)
        # Compute mask for attention
        mask = self.positional_embedding.compute_mask(x)
        x = self.transformer(x, mask)
        # Convert to shape (batch, embed_dim, seq_len) for pooling
        x = x.permute(0, 2, 1)
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


# -----------------------------------------------------------------------------
# Training and evaluation utilities
# -----------------------------------------------------------------------------
def train_model(train_data: np.ndarray, train_labels: np.ndarray,
                test_data: np.ndarray, test_labels: np.ndarray,
                num_classes: int, epochs: int = EPOCHS,
                batch_size: int = 8) -> VideoTransformerClassifier:
    """Train the video transformer classifier on the provided dataset.

    This function converts the NumPy arrays into PyTorch tensors, sets up a
    simple training loop with Adam optimiser and cross–entropy loss, and
    evaluates the model on a held–out test set.  The best model (based on
    test accuracy) is returned.

    Args:
        train_data: Array of shape (num_train, seq_len, embed_dim).
        train_labels: Array of shape (num_train,) containing class indices.
        test_data: Array of shape (num_test, seq_len, embed_dim).
        test_labels: Array of shape (num_test,) containing class indices.
        num_classes: The number of unique classes.
        epochs: Number of epochs to train.
        batch_size: Batch size for the DataLoader.

    Returns:
        The trained ``VideoTransformerClassifier`` model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoTransformerClassifier(num_classes).to(device)
    # Convert data to tensors
    x_train = torch.tensor(train_data, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    x_test = torch.tensor(test_data, dtype=torch.float32)
    y_test = torch.tensor(test_labels, dtype=torch.long)
    # Create DataLoader
    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False
    )
    # Optimiser and loss
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimiser.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * x_batch.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(x_batch)
                preds = logits.argmax(dim=-1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            # Save the best model weights
            torch.save(model.state_dict(), "video_transformer_best.pth")
        print(f"Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, accuracy={acc:.4f}")
    # Load best model before returning
    model.load_state_dict(torch.load("video_transformer_best.pth"))
    return model


def prepare_single_video(frames: np.ndarray,
                         feature_extractor: nn.Module,
                         preprocess) -> np.ndarray:
    """Prepare a single video for inference.

    This mirrors the Keras ``prepare_single_video`` function.  It pads the
    sequence to ``MAX_SEQ_LENGTH`` if necessary and extracts DenseNet
    features for each frame.

    Args:
        frames: A NumPy array of shape (frames, H, W, 3) containing raw RGB
            frames.
        feature_extractor: Pre–trained DenseNet feature extractor.
        preprocess: Preprocessing transform for input frames.

    Returns:
        A NumPy array of shape (1, MAX_SEQ_LENGTH, NUM_FEATURES).
    """
    # Pad sequence
    if len(frames) < MAX_SEQ_LENGTH:
        pad_len = MAX_SEQ_LENGTH - len(frames)
        frames = np.concatenate([
            frames,
            np.zeros((pad_len, IMG_SIZE, IMG_SIZE, 3), dtype=frames.dtype)
        ], axis=0)
    elif len(frames) > MAX_SEQ_LENGTH:
        frames = frames[:MAX_SEQ_LENGTH]
    # Extract features
    feats = extract_features(frames, feature_extractor, preprocess)
    if feats.shape[0] < MAX_SEQ_LENGTH:
        feat_pad = np.zeros((MAX_SEQ_LENGTH - feats.shape[0], NUM_FEATURES), dtype=np.float32)
        feats = np.concatenate([feats, feat_pad], axis=0)
    return feats[None, ...]


def predict_action(model: VideoTransformerClassifier, video_path: str,
                   class_vocab: List[str], feature_extractor: nn.Module,
                   preprocess) -> np.ndarray:
    """Predict the action class of a single video and print top probabilities.

    Args:
        model: Trained ``VideoTransformerClassifier``.
        video_path: Relative path to the video within the ``test`` directory.
        class_vocab: List mapping class indices back to labels.
        feature_extractor: Pre–trained DenseNet feature extractor.
        preprocess: Preprocessing transform.

    Returns:
        A NumPy array of frames used for visualisation.
    """
    frames = load_video(os.path.join("test", video_path))
    frame_features = prepare_single_video(frames, feature_extractor, preprocess)
    device = next(model.parameters()).device
    x = torch.tensor(frame_features, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits[0], dim=0).cpu().numpy()
    for i in np.argsort(probs)[::-1]:
        print(f"  {class_vocab[i]}: {probs[i] * 100:5.2f}%")
    return frames


def to_gif(images: np.ndarray, duration: int = 200, filename: str = "animation.gif") -> None:
    """Save a sequence of images as an animated GIF.

    Args:
        images: A NumPy array of shape (frames, H, W, 3) containing RGB images.
        duration: Duration between frames in milliseconds.
        filename: Output filename for the GIF.
    """
    converted_images = images.astype(np.uint8)
    imageio.mimsave(filename, converted_images, format="GIF", duration=duration / 1000.0)


if __name__ == "__main__":
    # Example usage (not executed by default).
    # To run training, uncomment the following lines and ensure that
    # train.csv, test.csv and the corresponding .npy files are present.
    #
    # train_df = pd.read_csv("train.csv")
    # test_df = pd.read_csv("test.csv")
    # feature_extractor, preprocess = build_feature_extractor()
    # train_data, train_labels = np.load("train_data.npy"), np.load("train_labels.npy")
    # test_data, test_labels = np.load("test_data.npy"), np.load("test_labels.npy")
    # num_classes = len(np.unique(train_labels))
    # model = train_model(train_data, train_labels, test_data, test_labels, num_classes)
    #
    # # Example inference on a random test video
    # class_vocab = sorted(set(train_df["tag"].tolist()))
    # test_video = np.random.choice(test_df["video_name"].values.tolist())
    # print(f"Test video path: {test_video}")
    # frames = predict_action(model, test_video, class_vocab, feature_extractor, preprocess)
    # to_gif(frames[:MAX_SEQ_LENGTH])
    pass