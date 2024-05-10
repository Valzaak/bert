def custom_compute_metrics(p):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.preprocessing import MultiLabelBinarizer

    mlb = MultiLabelBinarizer()

    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)

    mask = labels != -100

    true_predictions = predictions[mask]
    true_labels = labels[mask]

    mlb.fit([[i for i in range(13)]])
    true_predictions_one_hot = mlb.transform(true_predictions.reshape(-1, 1))
    true_labels_one_hot = mlb.transform(true_labels.reshape(-1, 1))

    precision = precision_score(
        true_labels_one_hot, true_predictions_one_hot, zero_division=0, average="micro"
    )
    recall = recall_score(
        true_labels_one_hot, true_predictions_one_hot, zero_division=0, average="micro"
    )
    f1 = f1_score(
        true_labels_one_hot, true_predictions_one_hot, zero_division=0, average="micro"
    )
    accuracy = accuracy_score(true_labels_one_hot, true_predictions_one_hot)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }
