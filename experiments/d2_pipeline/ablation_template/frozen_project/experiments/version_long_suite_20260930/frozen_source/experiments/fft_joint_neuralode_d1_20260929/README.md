# End-to-end learnable FFT + Neural ODE, Temperature

- One seed (`2024`) and the existing 511/110/109-day split.
- Task: next-day displacement prediction conditional on measured next-day Temperature temperature. This is not unknown-future temperature forecasting.
- Inputs to `model.forward`: only the preceding 30 days of temperature/displacement and the explicitly known next-day temperature. Future displacement is used only in the loss.
- Cutoffs initialize from the completed 30-day Stage-1 experiment (43.313 h and 7.377 h). They are frozen for five warm-up epochs and then updated by validation-selected end-to-end prediction training.
- Total loss: observed displacement prediction MSE + 0.02 Stage-1 decomposition loss + 0.1 LF/MF component supervision + 0.001 Neural-ODE residual penalty.
- The LF branch is a slow Neural ODE; the MF branch is a Bouc--Wen Neural ODE with a bounded neural residual. HF is not predicted.
- Best checkpoint selection uses all-day validation displacement MSE. First/second 12-hour metrics are reported but do not select the model. Test is sealed.
