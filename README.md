# ECG Heartbeat Classification: GRU vs. 1D CNN on MIT-BIH

Classifying **87,554 single heartbeats** from the **MIT-BIH Arrhythmia** dataset into 5 classes: normal, supraventricular, ventricular, fusion and unclassifiable beats. The dataset is highly imbalanced: 83% of the beats are normal.

**Result:** a **1D CNN with class weights** reaches **96.4% accuracy** and **0.926 balanced accuracy**, and catches 84–99% of every abnormal beat type. Predicting "normal" for every beat gives a 0.828 accuracy baseline but a balanced accuracy of only 0.200.

## Debugging the data first

My first models did not beat the baseline: GRU, 1D CNN and even a Random Forest all predicted only "normal". The loss stayed at ln(5) ≈ 1.61, the value you get when a model assigns equal probability to every class and learns nothing from the input.

Since three very different models failed in the same way, I checked the data instead of the models:
- In the provided `ECG_X.npy` / `ECG_Y.npy` files, **all 5 classes had the same average signal length (~111.7) and amplitude (~0.29)**. Real arrhythmia classes differ in shape and duration.
- A Random Forest on the raw signals scored a **balanced accuracy of 0.20**, which is chance level for 5 classes.
- The class counts matched the original MIT-BIH training set exactly. So the signals were the right ones, but they were not aligned with their labels.

I reloaded the data from the original source (Kaggle, `shayanfazeli/heartbeat`, `mitbih_train.csv`). The classes became clearly different (for example, fusion beats average 78.8 samples vs. ~111–118 for the others), and the same Random Forest check rose to 0.74. A diagnostic cell at the end of the notebook reproduces this check.

## Approach
1. **Variable-length sequences:** removed the zero padding at the end of each beat, then re-padded with `-1` to 187 samples, so the GRU can use a `Masking` layer.
2. **Stratified 80/20 train/test split**, so rare classes appear in the test set in the same proportion.
3. **Class weights** (`compute_class_weight('balanced')`) to counter the 83% / 0.7% imbalance.
4. **Models:**
   - **GRU:** Masking → GRU(20) → GRU(20) → Dense(50) → Dropout → Softmax
   - **1D CNN:** Conv1D(32) → MaxPool → Conv1D(64) → MaxPool → Conv1D(128) → Dense(64) → Dropout → Softmax
   - Both trained with Adam and EarlyStopping (patience 5).

## Results (test set, 17,511 beats)

| Model | Accuracy | Balanced accuracy | Macro F1 |
|---|---|---|---|
| Baseline (always "normal") | 0.828 | 0.200 | 0.181 |
| Random Forest (data check, 20k training samples, no class weights) | 0.970 | 0.780 | 0.851 |
| GRU + class weights | 0.874 | 0.873 | 0.665 |
| **1D CNN + class weights** | **0.964** | **0.926** | **0.837** |

**Recall per class (1D CNN):** normal 0.97 · supraventricular 0.87 · ventricular 0.96 · fusion 0.84 · unclassifiable 0.99

**Trade-off:** class weights push the models to flag rare beats more often. This raises recall for the rare classes but lowers their precision: 0.60 for supraventricular and 0.44 for fusion beats with the CNN. The Random Forest shows the opposite trade-off: high precision, but it misses about half of the fusion beats. In a screening setting, missing an abnormal beat is usually more costly than a false alarm, so the CNN's balance is the better fit.

## Tech
Python · TensorFlow / Keras (GRU, Conv1D, Masking) · scikit-learn (class weights, metrics, Random Forest) · pandas · NumPy · Matplotlib · Google Colab (T4 GPU)

## Files
- `classify_electrocardiograms.ipynb` – full analysis, including the data diagnostics
- `tests/` – automated checks

Built during the Workintech Data Science & Machine Learning program.
