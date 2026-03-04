"""
SKODA Mini Checkpoint Dataset Loader

OLD STRATIFIED SPLIT WITH SHUFFLE (for MicroBiConvLSTM paper reproducibility)

WARNING: This implementation uses the OLD stratified split with shuffle,
which can cause data leakage when using overlapping windows. This is
intentional to match the original MicroBiConvLSTM paper results.

For fair evaluation, use the nanoharmamba.data.skoda module which uses
temporal splits to prevent data leakage.

SKODA: Automotive assembly line gesture recognition
- 1 subject performing 10 manipulative gestures + Null class
- 10 sensors on right arm, 3-axis accelerometer each = 30 channels
- Sampling rate: ~98Hz
- Challenge: High Null class proportion + industrial machine vibration

SIGNAL RESCUE STRATEGY:
- 5Hz Low-Pass Butterworth Filter (removes machine vibration 50-60Hz)
- Human arms cannot manipulate objects faster than 2-3Hz
- Filter prevents industrial vibration aliasing into gesture band
- Label Smoothing 0.1 (fuzzy gesture boundaries)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import scipy.io as sio
from scipy.signal import butter, filtfilt


# Original label values from the dataset
ORIGINAL_LABELS = {
    32: 0,   # null -> 0
    48: 1,   # write on notepad
    49: 2,   # open hood
    50: 3,   # close hood
    51: 4,   # check gaps front door
    52: 5,   # open left front door
    53: 6,   # close left front door
    54: 7,   # close both left doors
    55: 8,   # check trunk gaps
    56: 9,   # open and close trunk
    57: 10,  # check steering wheel
}

# SKODA Activity Labels (mapped to 0-10)
SKODA_ACTIVITIES = {
    0: "null",
    1: "write_on_notepad",
    2: "open_hood",
    3: "close_hood",
    4: "check_gaps_front",
    5: "open_left_front_door",
    6: "close_left_front_door",
    7: "close_both_left_doors",
    8: "check_trunk_gaps",
    9: "open_close_trunk",
    10: "check_steering_wheel"
}

NUM_CLASSES = 11
NUM_SENSORS = 10
NUM_CHANNELS = 30  # 10 sensors × 3 axes (calibrated only)
SAMPLING_RATE = 98  # Hz


def butterworthLowpass(cutoff: float, fs: float, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Design a Butterworth low-pass filter."""
    nyq = 0.5 * fs
    normalCutoff = cutoff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog=False)
    return b, a


def applyLowpassFilter(data: np.ndarray, cutoff: float = 5.0, fs: float = SAMPLING_RATE, order: int = 4) -> np.ndarray:
    """Apply Butterworth low-pass filter to remove machine vibration."""
    b, a = butterworthLowpass(cutoff, fs, order)
    
    filteredData = np.zeros_like(data)
    for c in range(data.shape[1]):
        filteredData[:, c] = filtfilt(b, a, data[:, c])
    
    return filteredData.astype(np.float32)


def computeSkodaClassWeights(labels: np.ndarray, numClasses: int = NUM_CLASSES) -> torch.Tensor:
    """Compute class weights for imbalanced Skoda dataset."""
    classCounts = np.bincount(labels.astype(int), minlength=numClasses)
    classCounts = np.maximum(classCounts, 1)  # Avoid division by zero
    totalSamples = len(labels)
    weights = totalSamples / (numClasses * classCounts)
    return torch.tensor(weights, dtype=torch.float32)


class SkodaDataset(Dataset):
    """
    SKODA Mini Checkpoint Dataset.
    
    OLD IMPLEMENTATION WITH STRATIFIED SHUFFLE (for paper reproducibility).
    
    Args:
        root: Path to Skoda dataset folder
        split: 'train' or 'test'
        windowSize: Sliding window size (default 98 = ~1s at 98Hz)
        stride: Window stride (default 24 = 75% overlap for train)
        normalize: Whether to normalize data
        testRatio: Ratio for test split (default 0.2)
        transform: Optional transform
        seed: Random seed for reproducibility
        applyFilter: Apply 5Hz Butterworth low-pass filter (Signal Rescue)
        filterCutoff: Filter cutoff frequency in Hz (default 5Hz)
        allowOverlap: Use overlapping windows (75% overlap for train)
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        windowSize: int = 98,
        stride: int = 24,
        normalize: bool = True,
        testRatio: float = 0.2,
        transform=None,
        seed: int = 42,
        applyFilter: bool = True,
        filterCutoff: float = 5.0,
        allowOverlap: bool = True
    ):
        super().__init__()
        
        self.root = Path(root)
        self.split = split
        self.windowSize = windowSize
        # Overlapping windows for training (75% overlap), non-overlapping for test
        if allowOverlap:
            self.stride = stride if split == 'train' else windowSize
        else:
            self.stride = windowSize  # Non-overlapping for both splits
        self.normalize = normalize
        self.testRatio = testRatio
        self.transform = transform
        self.seed = seed
        self.applyFilter = applyFilter
        self.filterCutoff = filterCutoff
        
        # Will be set during loading
        self.mean = None
        self.std = None
        
        # Load data
        self.windows, self.labels = self._loadData()
        
        # Normalization stats (channel-wise Z-Score)
        if self.normalize and len(self.windows) > 0:
            self.mean = self.windows.mean(axis=(0, 1), keepdims=True)
            self.std = self.windows.std(axis=(0, 1), keepdims=True) + 1e-8
            self.windows = (self.windows - self.mean) / self.std
    
    def _findMatFile(self) -> Optional[Path]:
        """Find the right arm .mat file."""
        searchPaths = [
            self.root / "right_classall_clean.mat",
            self.root / "SkodaMiniCP_2015_08" / "right_classall_clean.mat",
        ]
        
        for path in searchPaths:
            if path.exists():
                return path
        
        # Search recursively
        for matFile in self.root.rglob("right_classall_clean.mat"):
            return matFile
        
        return None
    
    def _extractCalibratedAccel(self, rawData: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract calibrated acceleration data and labels from raw .mat data."""
        labels = rawData[:, 0].astype(np.int64)
        
        channels = []
        for s in range(NUM_SENSORS):
            baseCol = 1 + s * 7
            xCal = rawData[:, baseCol + 1]
            yCal = rawData[:, baseCol + 2]
            zCal = rawData[:, baseCol + 3]
            channels.extend([xCal, yCal, zCal])
        
        data = np.stack(channels, axis=1).astype(np.float32)
        return data, labels
    
    def _mapLabels(self, labels: np.ndarray) -> np.ndarray:
        """Map original labels (32, 48-57) to consecutive indices (0-10)."""
        mappedLabels = np.full_like(labels, -1)
        
        for origLabel, mappedLabel in ORIGINAL_LABELS.items():
            mappedLabels[labels == origLabel] = mappedLabel
        
        return mappedLabels
    
    def _loadData(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load SKODA data with OLD stratified split + shuffle."""
        
        matFile = self._findMatFile()
        
        if matFile is None:
            print(f"   Warning: SKODA right_classall_clean.mat not found in {self.root}")
            print(f"   Using synthetic data for testing pipeline")
            return self._generateSyntheticData()
        
        try:
            mat = sio.loadmat(str(matFile))
            
            dataKey = None
            for key in mat.keys():
                if 'right_classall_clean' in key.lower():
                    dataKey = key
                    break
            
            if dataKey is None:
                for key in mat.keys():
                    if not key.startswith('_'):
                        dataKey = key
                        break
            
            if dataKey is None:
                raise ValueError(f"Could not find data in {matFile}")
            
            rawData = mat[dataKey]
            print(f"   Loaded {matFile.name}: shape={rawData.shape}")
            
            data, labels = self._extractCalibratedAccel(rawData)
            labels = self._mapLabels(labels)
            
            validMask = labels >= 0
            data = data[validMask]
            labels = labels[validMask]
            
            if self.applyFilter:
                print(f"   Applying {self.filterCutoff}Hz low-pass filter (Signal Rescue)")
                data = applyLowpassFilter(data, cutoff=self.filterCutoff, fs=SAMPLING_RATE)
            
            print(f"   Data: {data.shape}, Labels unique: {np.unique(labels)}")
            
        except Exception as e:
            print(f"   Warning: Error loading SKODA data: {e}")
            import traceback
            traceback.print_exc()
            print(f"   Using synthetic data for testing pipeline")
            return self._generateSyntheticData()
        
        # OLD IMPLEMENTATION: Window FIRST, then stratified split WITH SHUFFLE
        # NOTE: This can cause data leakage with overlapping windows!
        # Adjacent windows share 75% of data and may end up in both train and test.
        
        allWindows, allLabels = self._segmentWindows(data, labels)
        
        if len(allLabels) == 0:
            return allWindows, allLabels
        
        # OLD: Stratified split WITH SHUFFLE (causes leakage with overlapping windows)
        np.random.seed(self.seed)
        
        trainIndices, testIndices = [], []
        
        for classIdx in range(NUM_CLASSES):
            classMask = allLabels == classIdx
            classIndices = np.where(classMask)[0]
            
            if len(classIndices) == 0:
                continue
            
            # OLD: Shuffle indices for this class (CAUSES DATA LEAKAGE!)
            np.random.shuffle(classIndices)
            
            nTrain = int(len(classIndices) * (1 - self.testRatio))
            trainIndices.extend(classIndices[:nTrain].tolist())
            testIndices.extend(classIndices[nTrain:].tolist())
        
        if self.split == 'train':
            indices = np.array(trainIndices)
        else:
            indices = np.array(testIndices)
        
        # OLD: Shuffle final indices to mix classes
        np.random.shuffle(indices)
        
        return allWindows[indices], allLabels[indices]
    
    def _generateSyntheticData(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic data for testing when real data unavailable."""
        np.random.seed(self.seed if self.split == 'train' else self.seed + 1)
        
        nWindows = 2000 if self.split == 'train' else 500
        
        windows = np.random.randn(nWindows, self.windowSize, NUM_CHANNELS).astype(np.float32)
        labels = np.zeros(nWindows, dtype=np.int64)
        labels[nWindows//2:] = np.random.randint(1, NUM_CLASSES, nWindows - nWindows//2)
        
        return windows, labels
    
    def _segmentWindows(
        self,
        data: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Segment continuous data into windows."""
        windows = []
        windowLabels = []
        
        for start in range(0, len(data) - self.windowSize + 1, self.stride):
            end = start + self.windowSize
            windowData = data[start:end]
            windowLabelArr = labels[start:end]
            
            uniqueLabels, counts = np.unique(windowLabelArr, return_counts=True)
            majorityLabel = uniqueLabels[counts.argmax()]
            
            if (windowLabelArr == majorityLabel).mean() >= 0.7:
                windows.append(windowData)
                windowLabels.append(int(majorityLabel))
        
        if len(windows) == 0:
            return np.array([]).reshape(0, self.windowSize, NUM_CHANNELS), np.array([])
        
        return np.stack(windows, axis=0), np.array(windowLabels, dtype=np.int64)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        x = torch.from_numpy(self.windows[idx])
        y = int(self.labels[idx])
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    @property
    def numClasses(self) -> int:
        return NUM_CLASSES
    
    @property
    def inputShape(self) -> Tuple[int, ...]:
        return (self.windowSize, NUM_CHANNELS)
    
    @property
    def classNames(self):
        return list(SKODA_ACTIVITIES.values())


def getSkodaLoaders(
    root: str = "./datasets/Skoda",
    batchSize: int = 64,
    windowSize: int = 98,
    stride: int = 24,
    numWorkers: int = 0,
    returnWeights: bool = False,
    allowOverlap: bool = True
) -> Tuple[DataLoader, DataLoader, Optional[torch.Tensor]]:
    """
    Get SKODA train/test data loaders.
    
    NOTE: Uses OLD stratified split with shuffle for MicroBiConvLSTM paper reproducibility.
    """
    
    trainDataset = SkodaDataset(
        root=root,
        split='train',
        windowSize=windowSize,
        stride=stride,
        normalize=True,
        allowOverlap=allowOverlap
    )
    
    testDataset = SkodaDataset(
        root=root,
        split='test',
        windowSize=windowSize,
        stride=windowSize,
        normalize=False,
        allowOverlap=allowOverlap
    )
    
    # Apply train normalization to test
    if len(trainDataset.windows) > 0 and len(testDataset.windows) > 0:
        testDataset.mean = trainDataset.mean
        testDataset.std = trainDataset.std
        testDataset.windows = (testDataset.windows - testDataset.mean) / testDataset.std
        testDataset.normalize = True
    
    classWeights = None
    if len(trainDataset.labels) > 0:
        classWeights = computeSkodaClassWeights(trainDataset.labels, numClasses=NUM_CLASSES)
    
    trainLoader = DataLoader(
        trainDataset,
        batch_size=batchSize,
        shuffle=True,
        num_workers=numWorkers,
        pin_memory=True,
        drop_last=True
    )
    
    testLoader = DataLoader(
        testDataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=numWorkers,
        pin_memory=True
    )
    
    if returnWeights:
        return trainLoader, testLoader, classWeights
    return trainLoader, testLoader


if __name__ == "__main__":
    print("Testing SKODA Dataset (OLD stratified split with shuffle)...")
    
    try:
        dataset = SkodaDataset(
            root='./datasets/Skoda',
            split='train'
        )
        
        print(f"Train samples: {len(dataset)}")
        print(f"Input shape: {dataset.inputShape}")
        print(f"Num classes: {dataset.numClasses}")
        
        if len(dataset) > 0:
            x, y = dataset[0]
            print(f"Sample shape: {x.shape}")
            print(f"Label: {y} ({SKODA_ACTIVITIES[y]})")
        
    except Exception as e:
        print(f"Error: {e}")
