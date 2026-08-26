import csv

def get_csv_writer(path, fieldnames):
    with open(path, 'w') as w:
        csv.writer(w).writerow(fieldnames)
    def append_func(rows):
        if not rows:
            return
        with open(path, 'a') as a:
            csv.writer(a).writerows(rows)
    return append_func