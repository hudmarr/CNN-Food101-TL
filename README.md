
# Beating the DeepFood Paper on the Food101 Dataset


This project challenges the **DeepFood research paper**'s results on the Food101 dataset, which achieved a validation accuracy of 77.4%. Through a series of experiments utilizing **EfficientNet** models, this notebook successfully surpassed the benchmark, achieving a validation accuracy of **88.55%**.


## Purpose
The primary reason for undertaking this project was to test and strengthen my knowledge of the following concepts:
- Convolutional Neural Networks (CNNs)
- Mixed precision training
- Transfer learning
- Fine-tuning techniques
- Use of callbacks (e.g., ModelCheckpoint, EarlyStopping, LearningRateScheduler)
- Data processing and pipeline optimization

## Notebook Structure

### 1. **Data Preparation**
- Downloaded and prepared the Food101 dataset using TensorFlow Datasets.
- Preprocessing steps included resizing images, adjusting data types for mixed precision, and batching for optimized training.

### 2. **Model Architectures and Experiments**
The notebook contains three main experiments with different configurations of **EfficientNet** models:

#### Experiment 1: EfficientNetB0 with Full Unfreezing
- Fine-tuned the **EfficientNetB0** model by initially freezing all layers, then unfreezing all layers at once.
- Utilized callbacks including **ModelCheckpoint**, **EarlyStopping**, and **LearningRateScheduler**.
- **Results**:
  - Validation accuracy: **82.28%**
  - Training time: ~52 minutes (13 minutes feature extraction + 39 minutes fine-tuning)

#### Experiment 2: Incremental Unfreezing of EfficientNetB0
- Experimented with gradually unfreezing layers (20, 50, 100, and then the rest).
- While promising in theory, this approach underperformed compared to the first experiment.
- **Results**:
  - Validation accuracy: **81.11%**
  - Training time: ~124 minutes

#### Experiment 3: EfficientNetB4 with Full Unfreezing
- Repeated the first experiment's methodology but used the **EfficientNetB4** model for greater capacity.
- **Results**:
  - Validation accuracy: **88.55%**
  - Training time: ~4 hours

### 3. **Model Evaluation**
- Visualized training progress using TensorBoard.
- Performed predictions on random images for qualitative analysis.

### 4. **Conclusions**
- The EfficientNetB4 model outperformed both the baseline and the earlier EfficientNetB0 configurations.
- **Key takeaways**:
  - Full unfreezing was more effective than incremental unfreezing for this dataset.
  - Choosing a larger, more powerful architecture significantly improved performance.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/repo-name.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook in a Jupyter environment.

## Results Summary
| Experiment | Model           | Validation Accuracy | Training Time |
|------------|------------------|---------------------|---------------|
| 1          | EfficientNetB0  | 82.28%             | ~52 minutes   |
| 2          | EfficientNetB0  | 81.11%             | ~124 minutes  |
| 3          | EfficientNetB4  | 88.55%             | ~4 hours      |

## Additional Notes
- This notebook is GPU-optimized and tested on a T4 graphics card.
- Future work could include exploring other architectures or advanced optimization techniques to further improve results.

Feel free to contribute or fork the repository to test your own ideas!
