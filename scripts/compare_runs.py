import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
print(f"{'mode':<8}{'subset':<14}" + "".join(f"@{k:<12}" for k in a["ks"]))
for mode in a["results"]:
    for subset in a["results"][mode]:
        cells = []
        for k in a["ks"]:
            x, y = a["results"][mode][subset][str(k)], b["results"][mode][subset][str(k)]
            cells.append(f"{x:.3f}/{y:.3f}".ljust(13))
        print(f"{mode:<8}{subset:<14}" + "".join(cells))