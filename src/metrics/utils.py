import editdistance

# Based on seminar materials

# Don't forget to support cases when target_text == ''


def calc_cer(target_text, predicted_text) -> float:
    if target_text == "":
        if predicted_text == "":
            return 0
        else:
            return 1
    edits = editdistance.eval(target_text, predicted_text)
    return edits / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    target_words = target_text.split()
    predicted_words = predicted_text.split()

    if len(target_words) == 0:
        if len(predicted_words) == 0:
            return 0
        else:
            return 1

    edits = editdistance.eval(target_words, predicted_words)
    return edits / len(target_words)
