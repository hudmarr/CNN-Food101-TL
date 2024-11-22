
# Respository of Neural Network functions I find helpful

import tensorflow as tf





# Imports an image and resizes it to be used. If scale is false, image is not normalized.
import io

def import_prep_image(filename, img_shape=224, scale = True):
  """
  Imports an image from a filename, turns it into a tensor and reshapes into
  (img_shape, img_shape, 3).

  Parameters
  ----------
  filename (str): filename of image
  img_shape (int): size to resize to, default 224
  scale (bool): whether to scale or not
  """

  # Read in image
  image = tf.io.read_file(filename)
  # Turn into tensor
  image = tf.image.decode_jpeg(image)
  # Resize
  image = tf.image.resize(image, [img_shape, img_shape])

  # Normalize if wanted
  if scale:
    image = image/255.
  else:
    return image
  







# Predict + Plot it
import matplotlib.pyplot as plt

def pred_plus_plot(model, image_path, class_names, show_axis=False):
    """
    Predicts the class of an image using a trained model and plots the image with the predicted class as the title.

    Parameters
    ----------
    model: Trained model used for making predictions.
    image_path: Path to the target image.
    class_names: List of class names corresponding to the model's outputs.
    show_axis: Whether to display the plot axis (default=False).

    """
    # Load and preprocess the image
    img = import_prep_image(image_path)

    # Make a prediction
    prediction = model.predict(tf.expand_dims(img, axis=0))

    # Determine the predicted class
    if len(prediction[0]) > 1:  # For multi-class classification
        predicted_class = class_names[prediction.argmax()]
    else:  # For binary classification
        predicted_class = class_names[int(tf.round(prediction)[0][0])]

    # Plot the image with the predicted class
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_class}")
    if not show_axis:
        plt.axis(False)
    plt.show()







# Makes tensorboard callback with files saved to "log_dir/experiment_name/current_datetime/"

import datetime

def make_callback_TB(log_dir, experiment_name):
    """
    Creates a TensorBoard callback to store log files.

    Log files are saved to:
        "log_dir/experiment_name/current_datetime/"

    Parameters
    ----------
    log_dir: Directory to store TensorBoard log files.
    experiment_name: Name of the experiment directory (e.g., "model_1").
    """
    # Define the directory for log files
    save_path = f"{log_dir}/{experiment_name}/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Create the TensorBoard callback
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=save_path)
    
    print(f"Saving TensorBoard log files to: {save_path}")
    return tb_callback







# Basically just unzips a folder

import zipfile

def unzip_folder(zip_path):
    """
    Unzips a folder

    Parameters
    ----------
    zip_path: str, path to file
    """
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall()






# Takes in the original history and new history and makes a graph comparing them
import matplotlib.pyplot as plt

def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compares two TensorFlow model History objects.

    Parameters
    ----------
    original_history: History object from the initial model training.
    new_history: History object from the continued model training.
    initial_epochs: Number of epochs in original_history (new_history plot starts from this point).
    """
    # Extract metrics from original history
    train_acc = original_history.history["accuracy"]
    train_loss = original_history.history["loss"]
    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_train_acc = train_acc + new_history.history["accuracy"]
    total_train_loss = train_loss + new_history.history["loss"]
    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Create plots
    plt.figure(figsize=(8, 8))

    # Accuracy plot
    plt.subplot(2, 1, 1)
    plt.plot(total_train_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.axvline(x=initial_epochs - 1, linestyle='--', color='gray', label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Loss plot
    plt.subplot(2, 1, 2)
    plt.plot(total_train_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.axvline(x=initial_epochs - 1, linestyle='--', color='gray', label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')

    plt.tight_layout()
    plt.show()







# Makes two plots that compare the loss curves for the training and validation metrics.

import matplotlib.pyplot as plt

def loss_plot(history):
    """
    Plots loss and accuracy curves for training and validation.

    Parameters
    ----------
    history: TensorFlow model History object containing training metrics.
    """
    # Extract metrics from history
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(train_loss))  # Number of epochs

    # Plot training and validation loss
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot training and validation accuracy
    plt.figure()
    plt.plot(epochs, train_accuracy, label='Training Accuracy')
    plt.plot(epochs, val_accuracy, label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()







# Breaks down a directory listing all of the contents

import os

def break_down_dir(directory):
    """
    Analyzes the contents of a directory, listing subdirectories and file counts.

    Parameters
    ----------
    directory: str, path to the directory to be analyzed

    Outputs
    -------
    Prints the number of subdirectories, the number of files in each subdirectory, 
    and the name of each subdirectory.
    """
    for dirpath, dirnames, filenames in os.walk(directory):
        print(f"There are {len(dirnames)} subdirectories and {len(filenames)} files in '{dirpath}'.")








# Creates a confusion matrix based on user needs. Matrix is a remix of Scikit-Learn's example

import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def create_confusion_matrix(true_labels, pred_labels, class_names=None, size=(10, 10), font_size=15, normalize=False, save_image=False, x_rotate=False):
    """
    Creates a confusion matrix comparing actual labels and predicted labels.

    Parameters
    ----------
    true_labels: Array of actual labels (same shape as pred_labels).
    pred_labels: Array of predicted labels (same shape as true_labels).
    class_names: List of class names (optional). If None, integer labels are used.
    size: Tuple indicating the size of the plot (default=(10, 10)).
    font_size: Font size for text in the plot (default=15).
    normalize: Whether to show percentages instead of raw counts (default=False).
    save_image: Whether to save the confusion matrix as an image (default=False).
    x_rotate: Whether to rotate x-axis labels for better readability (default=False), make True for models with many classes

    Returns
    -------
    A confusion matrix plot comparing true_labels and pred_labels.

    Example:
        create_confusion_matrix(
            true_labels=test_labels,
            pred_labels=test_preds,
            class_names=label_names,
            size=(12, 12),
            font_size=12,
            x_rotate=True
        )
    """
    # Generate the confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # Normalize if needed
    num_classes = cm.shape[0]  # Total number of classes

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=size)
    color_map = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(color_map)

    # Set up labels
    labels = class_names if class_names else np.arange(num_classes)

    # Add axis titles and tick labels
    ax.set(title="Confusion Matrix",
           xlabel="Predicted",
           ylabel="Actual",
           xticks=np.arange(num_classes),
           yticks=np.arange(num_classes),
           xticklabels=labels,
           yticklabels=labels)

    # Position x-axis labels at the bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Rotate labels if x_rotate is True
    if x_rotate:
        plt.xticks(rotation=70, fontsize=font_size)
    plt.yticks(fontsize=font_size)

    # Determine threshold for text color
    threshold = (cm.max() + cm.min()) / 2.

    # Add text to each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, f"{cm[i, j]} ({cm_normalized[i, j]*100:.1f}%)",
                     ha="center",
                     color="white" if cm[i, j] > threshold else "black",
                     fontsize=font_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                     ha="center",
                     color="white" if cm[i, j] > threshold else "black",
                     fontsize=font_size)

    # Save the confusion matrix as an image if requested
    if save_image:
        fig.savefig("confusion_matrix.png")
