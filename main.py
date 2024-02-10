import json
import numpy as np
import json
from prettytable import PrettyTable

# Caricare le predizioni da un file JSON
with open('predictions_zs.json', 'r') as file:
    predictions = json.load(file)

# Caricare i ground truths da un file JSON
with open('ground_truths.json', 'r') as file:
    ground_truths = json.load(file)


# Funzione per calcolare l'IoU
def bbox_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0
    return iou


# Organizzare le predizioni e i ground truths per categoria
def organize_by_category(data):
    organized_data = {}
    for item in data:
        category = item['category_id']
        if category not in organized_data:
            organized_data[category] = []
        organized_data[category].append(item)
    return organized_data


# Calcolare TP e FP per ogni categoria
def calculate_tp_fp(predictions, ground_truths, iou_threshold=0.5):
    predictions_by_category = organize_by_category(predictions)
    ground_truths_by_category = organize_by_category(ground_truths)

    ap_per_category = {}
    for category, preds in predictions_by_category.items():
        if category not in ground_truths_by_category:
            continue  # Skip categories without ground truths

        print(f'Categoria {category}')

        gts = ground_truths_by_category[category]
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))

        matched_gts = set()
        for i, pred in enumerate(sorted(preds, key=lambda x: x['score'], reverse=True)):

            if i % 1000 == 0:
                print(f'Predizione {i}')

            best_iou = iou_threshold
            best_gt_id = None
            for j, gt in enumerate(gts):
                if j in matched_gts:
                    continue  # Ground truth already matched

                iou = bbox_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_id = j

            if best_gt_id is not None:
                tp[i] = 1
                matched_gts.add(best_gt_id)
            else:
                fp[i] = 1

        # Calcolare precision e recall per interpolazione a 11 punti
        cumsum_fp = np.cumsum(fp)
        cumsum_tp = np.cumsum(tp)
        recall = cumsum_tp / len(gts)
        precision = cumsum_tp / (cumsum_tp + cumsum_fp)

        # Applicare interpolazione
        precision_at_recall_levels = []
        for threshold in np.linspace(0, 1, 11):
            relevant_precisions = precision[recall >= threshold]
            max_precision = max(relevant_precisions) if len(relevant_precisions) > 0 else 0
            precision_at_recall_levels.append(max_precision)

        print(f'Valore per la categoria: {np.mean(precision_at_recall_levels)}')
        ap_per_category[category] = np.mean(precision_at_recall_levels)

    return ap_per_category


def get_category_name(index):
    # Apertura e lettura del file JSON
    with open('categories.json', 'r') as json_file:
        data = json.load(json_file)

    # Ricerca della categoria basata sull'indice/id
    for category in data["categories"]:
        if category["id"] == index:
            return category["name"]

    # Se l'indice non corrisponde a nessuna categoria, si ritorna None o un messaggio di errore
    return "Categoria non trovata"


if __name__ == '__main__':
    ap_per_category = calculate_tp_fp(predictions, ground_truths)
    x = PrettyTable()
    x.field_names = ["Class", "AP"]
    for index, value in enumerate(ap_per_category):
        x.add_row([get_category_name(index), ap_per_category[index]])
    print(x)
