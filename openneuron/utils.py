import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Функция форматирования строк
def format(value, decimals=2, edge_items=2):
    if value is None:
        return 'None'
    elif isinstance(value, (int, np.int32, np.int64, float, np.float32, np.float64)):
        return f'{np.round(value, decimals)}'
    elif isinstance(value, np.ndarray):
        if len(value) > 2 * edge_items:
            start_str = np.array2string(value[:edge_items], formatter={'float_kind': lambda x: f'{np.round(x, decimals)}'})
            end_str = np.array2string(value[-edge_items:], formatter={'float_kind': lambda x: f'{np.round(x, decimals)}'})
            return f'{start_str[:-1]} ... {end_str[1:]}'
        else:
            return np.array2string(value, formatter={'float_kind': lambda x: f'{np.round(x, decimals)}'})
    else:
        raise TypeError('The input must be an int, np.int32, np.int64, float, np.float32, np.float64, or np.ndarray.')


# Оценка точности
def network_scores(X, y, predictions):
    # predictions = predictions.reshape(y.shape)
    print('Scores:')
    print('Incorrect Predictions (Significant Difference) on Test Data:')
    all_predictions_correct = True

    # Функция для форматирования массивов
    def format_array(array, decimals=2):
        if isinstance(array, np.ndarray):
            if len(array) > 4:
                return f"[{array[0]:.{decimals}f} {array[1]:.{decimals}f} ... {array[-2]:.{decimals}f} {array[-1]:.{decimals}f}]"
            else:
                return f"[{' '.join(f'{x:.{decimals}f}' for x in array)}]"
        return array
    
    for y_pred, y_true, x in zip(predictions, y, X):
        predicted_class = np.argmax(y_pred)
        true_class = np.argmax(y_true)

        if predicted_class != true_class:
            # x_string = ' '.join([f'{x:.2f}' for x in x])
            x_formatted = format_array(x)
            y_pred_formatted = format_array(y_pred)
            y_true_formatted = format_array(y_true, decimals=0)
            print(f'Network inputs: {x_formatted}, predicted: {y_pred_formatted}, true: {y_true_formatted}, predicted class: {predicted_class}, true class: {true_class}')
            all_predictions_correct = False
        
    if all_predictions_correct:
        print('All predictions are correct.')
    
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y, axis=1)
    accuracy = accuracy_score(true_classes, predicted_classes) * 100  # преобразуем точность в проценты
    precision = precision_score(true_classes, predicted_classes, average='weighted', zero_division=1)
    recall = recall_score(true_classes, predicted_classes, average='weighted')
    f1 = f1_score(true_classes, predicted_classes, average='weighted', zero_division=1)

    print(f'Accuracy: {accuracy:.2f}%')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')