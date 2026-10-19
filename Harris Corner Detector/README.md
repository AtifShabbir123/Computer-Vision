# Computer Vision – Harris Corner Detector

## Files
| file / folder | what it is |
|---|---|
| `harris_assignment.py` | **the only code file**: Part A = detector + helper functions, Part B = all experiments |
| `report.docx` | the report (Word) |
| `results/` | all result images and plots, `t5_param_counts.csv`, `results.json` |
| `data/` | the test images (brick, cat, moon, generated checkerboard) |

## Requirements
Python 3.9+ with `numpy`, `matplotlib`, `opencv-python`, `scipy` (used only for checking/comparison).
`scikit-image` is needed only the first time to create the 3 sample photos (skipped if `data/brick.png`, `data/cat.png`, `data/moon.png` exist).

    pip install numpy matplotlib opencv-python scipy scikit-image

## How to run
    python harris_assignment.py            # all tasks, about 30 seconds
    python harris_assignment.py t1 t3      # only some parts: t1 t2 t3 t4 t5 bonus failure

All random numbers use fixed seeds, so the results are the same every run.

## Data
* Three grayscale 256×256 images: `brick` (man-made), `cat` (natural/textured), `moon` (low contrast). The PNG files can be changed in `data/` .

## Rules
Part A of the code uses only NumPy (plus cv2.resize / cv2.warpAffine to make transformed test images). It has no cv2.filter2D, GaussianBlur, Sobel, cornerHarris, no scipy.ndimage, and no loops over pixels (loops only over kernel offsets).
cv2 / scipy filters appear only in Part B, for checking (Task 1) and comparing (Task 5).

## Where each task is (function names)
* Task 1: `correlate2d`, `convolve2d`, `correlate_separable`, `gaussian_kernel1d/2d`, `CENTRAL_X`, `SOBEL_X`, `gaussian_derivative_kernel_x`, `task1`
* Task 2: `structure_tensor`, `harris_R`, `nms3x3`, `detect_corners`, `task2`
* Task 3: `eigenvalues_2x2`, `classify_R`, `rho_min`, `R_level_curve`, `task3`
* Task 4: `rotate_image`, `scale_image`, `add_brightness`, `mul_contrast`, `add_noise`, `repeatability`, `task4`
* Task 5, bonus, failure case: `task5`, `bonus`, `failure_case`
