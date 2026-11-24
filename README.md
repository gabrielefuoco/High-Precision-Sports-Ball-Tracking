# High-Precision Sports Ball Tracking

Detection and tracking of a football in video, built for the hard case: a small, fast object that is often occluded, blurry, or out of frame. A YOLO11s detector is kept deliberately simple; most of the accuracy comes from a post-processing pipeline that turns per-frame detections into a physically plausible trajectory.

## Table of contents

- [Results](#results)
- [How it works](#how-it-works)
- [Decisions](#decisions)
- [Repository map](#repository-map)
- [Running the notebooks](#running-the-notebooks)
- [Limitations](#limitations)
- [Documentation](#documentation)

## Results

Target was a mean squared error of 0.013 on the ground-truth position.

| Metric | Result |
| :--- | :---: |
| Box precision | 0.993 |
| Box recall | 0.968 |
| Final MSE | **0.0053** |
| MSE, clip 1 | 0.0078 |
| MSE, clip 2 | 0.0028 |

The final figure is the mean over the two test clips, and it is the number the post-processing sequence was selected against.

## How it works

### The detector

**YOLO11s**, chosen for the balance between weight and compute, trained on Kaggle with a P-100.

Two modifications target small objects specifically:

- the **P5 head (layer 22) is frozen**, so the capacity stays on the high-resolution features that actually carry a ball 20 pixels wide;
- training resolution is raised to **1088 px**, because at the default size the ball is a handful of pixels after the first downsampling.

Augmentation (flip, rotations, brightness variation, noise) was applied through Roboflow.

### The post-processing pipeline

Detection alone produces a cloud of points, not a trajectory. The pipeline applies, in order:

1. **outlier filter** — removes detections inconsistent with the recent path, using euclidean distance;
2. **linear interpolation** — fills the gaps left by occlusion and missed frames;
3. **parabolic trajectories** — reconstructs the segments where the ball leaves the field of view, the model's weakest case;
4. **validation and smoothing** — checks the interpolated segments before they replace real data.

The order of these steps was chosen empirically: it is the sequence that minimises the MSE on the test clips, not the one that seemed logical a priori.

### The evaluation

The metric compares the predicted position with the ground truth frame by frame, over the frames where both exist. Its coverage on the first clip is part of the result:

| Measure | Count | Share |
| :--- | :---: | :---: |
| Frame indices in range | 2,564 | — |
| Frames with a match (prediction and ground truth) | 237 | 68.5% of frames with data |
| Predictions without ground truth | 103 | 29.8% |
| Ground truth without prediction | 6 | 1.7% |
| Frame indices with no data at all | 2,219 | 86.5% of the range |

Most of the range carries no annotation at all, so the score is computed on the annotated core of the clip.

## Decisions

**A small detector plus heavy post-processing, instead of a bigger model.** The failure modes of the detector are temporal — blink and you lose the ball — and no increase in capacity fixes a trajectory with a hole in it. Interpolation and outlier rejection recover those frames at almost no cost.

**Freezing the P5 head.** It is a deliberate capacity reduction: for a 20-pixel object, the deepest features are noise, and leaving them trainable slowed convergence without improving recall.

**Lowering the confidence threshold and raising the detection cap to 4.** Normally a tracker keeps the most confident detection. Here the post-processing keeps the detection **closest to the last valid one**, because a confident detection can be a spectator's head or a line marking. That choice moves the selection criterion from the detector to the trajectory, where the physics lives.

**Parabolic reconstruction for out-of-field balls.** The model cannot see what has left the frame; a trajectory prior can. It is the only component that adds information the images do not contain, and it is confined to the segments where the ball is outside the field.

**A custom MSE instead of mAP as the final score.** mAP measures detection quality on annotated boxes; the deliverable is a continuous position over time. The MSE on the matched frames is the metric that reflects what the pipeline is for.

## Repository map

```
Notebook_Training_252053.ipynb   training workflow (Kaggle, GPU P-100)
Notebook_Test_252053.ipynb       inference, post-processing, MSE evaluation, annotated video
Relazione_Fuoco_Gabriele_252053.pdf   technical report: dataset, experiments, error analysis
```

## Running the notebooks

```bash
pip install ultralytics roboflow opencv-python gdown pyyaml
```

The notebooks were written for Kaggle: they download weights (`best.pt`, 19.3 MB) and the test clips from Google Drive at the start. **Weights and videos are not in this repository**, so a clean clone needs those links to be reachable before anything runs.

## Limitations

| Item | Impact | Status |
|---|---|---|
| Weights and test videos live on Google Drive | a clean clone cannot run without them | open |
| Two test clips only | the MSE is a mean over two samples, not a benchmark | accepted |
| 29.8% of predictions have no ground truth | the score says nothing about those frames | open |
| No automated tests | the pipeline is validated by inspecting plots and videos | open |
| One sport, one kind of broadcast | camera movement, zoom and field geometry are implicitly assuming football | accepted |
| Box precision and recall come from training validation | they are not re-measured by the test notebook | noted |

## Documentation

`Relazione_Fuoco_Gabriele_252053.pdf` is the technical report: dataset construction, the two training experiments, the post-processing design and the error analysis. The notebooks are the executable counterpart, with the evaluation cells that produced the numbers above.
