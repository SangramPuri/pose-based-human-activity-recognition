# Project Audit

## What the supplied reference established

The reference project uses 2D human-pose information, temporal windows, an LSTM classifier, video analysis, and a Flask interface. It describes six activities and a 32-frame sequence representation.

## What this study implementation changes

- Uses a clean PyTorch implementation rather than reproducing the supplied source files.
- Makes the input dimension explicit: 34 values (17 landmarks × x/y).
- Separates data loading, model definition, training, pose integration, and video processing.
- Does not copy the reference author's branding or identity.
- Does not publish the reference project's reported accuracy as a result of this implementation.

## Before portfolio publication

1. Train the model.
2. Evaluate on held-out data.
3. Record accuracy and macro-F1.
4. Add your own confusion matrix and plots.
5. Test the video pipeline.
6. Write your own project explanation after understanding every component.
