"""Build unlabelled corner pairs from the upload-list photos (downloaded locally; the download step is not published
because it holds the official storage address).

Each order in the list has one upload session (one day of uploads), almost always the return photos. For a car rented
more than once that day, the previous renter's return photo of a corner is the closest thing to the next renter's
pickup photo, so  (order n-1 return, order n return)  of the same car and ImageType 1-4 is a 借→還-like pair: any
difference is damage from rental n or from the gap between the two rentals. No labels; used for flag-rate estimates
and as a distillation pool. ImageType is what the customer picked and is often wrong (README), as in production.

Also lists the ImageType 5-9 ("other") photos, which often are customer close-ups, for weak labels.

Run: python experiments/xlsx_pairs.py   -> <IRENT_XLSX_PHOTOS>/xlsx_pairs.json, xlsx_other_photos.json
"""
import csv
import json
import os
import pathlib
import re
from collections import defaultdict

DEST = pathlib.Path(os.environ.get("IRENT_XLSX_PHOTOS", str(pathlib.Path.home() / "irent_data" / "xlsx_photos")))
TS = re.compile(r"_(\d{14})\.\w+$")


def main():
    rows = [r for r in csv.DictReader(open(DEST / "manifest.csv", encoding="utf-8")) if r["status"] in ("ok", "exists")]
    by_order = defaultdict(dict)
    car_of, t_of = {}, {}
    other = []
    for r in rows:
        o, t = r["order_number"], int(r["ImageType"])
        path = str(DEST / o / r["Image"])
        ts = TS.search(r["Image"]).group(1)
        car_of[o] = r["CarNo"]
        t_of[o] = min(t_of.get(o, ts), ts)
        if t in (1, 2, 3, 4):
            by_order[o].setdefault(t, path)          # first photo of that type if a customer uploaded twice
        elif 5 <= t <= 9:
            other.append({"order": o, "car": r["CarNo"], "type": t, "path": path, "ts": ts})
    by_car = defaultdict(list)
    for o in t_of:
        by_car[car_of[o]].append(o)
    pairs, gaps = [], []
    for car, orders in by_car.items():
        orders.sort(key=lambda o: t_of[o])
        for o1, o2 in zip(orders, orders[1:]):
            gaps.append((int(t_of[o2][8:10]) * 60 + int(t_of[o2][10:12])) - (int(t_of[o1][8:10]) * 60 + int(t_of[o1][10:12])))
            for k in (1, 2, 3, 4):
                if k in by_order[o1] and k in by_order[o2]:
                    pairs.append({"a": by_order[o1][k], "b": by_order[o2][k], "name": f"xlsx/{car}/{o1}->{o2}/{k}",
                                  "plate": car, "y": -1})
    (DEST / "xlsx_pairs.json").write_text(json.dumps(pairs, ensure_ascii=False), encoding="utf-8")
    (DEST / "xlsx_other_photos.json").write_text(json.dumps(other, ensure_ascii=False), encoding="utf-8")
    multi = sum(len(v) > 1 for v in by_car.values())
    gaps.sort()
    print(f"photos {len(rows)}, orders {len(t_of)}, cars {len(by_car)} ({multi} rented more than once)")
    print(f"consecutive order pairs {len(gaps)}, corner pairs {len(pairs)}, "
          f"cars in pairs {len({p['plate'] for p in pairs})}; median gap between upload sessions {gaps[len(gaps) // 2]} min")
    print(f"ImageType 5-9 photos {len(other)} from {len({o['order'] for o in other})} orders")


if __name__ == "__main__":
    main()
