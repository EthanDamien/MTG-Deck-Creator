"""Archetype deck-building templates with slot counts and wave assignments."""

TEMPLATES: dict[str, dict] = {
    "tribal": {
        "ratios": {
            "commander": 1,
            "theme": 26,
            "ramp": 14,
            "draw": 9,
            "removal": 8,
            "wipes": 4,
            "protection": 4,
            "lands": 34,
        },
        "waves": {
            1: ["commander", "theme"],
            2: ["ramp", "draw", "protection"],
            3: ["removal", "wipes"],
            4: ["lands"],
        },
    },
    "midrange": {
        "ratios": {
            "commander": 1,
            "theme": 20,
            "ramp": 14,
            "draw": 10,
            "removal": 10,
            "wipes": 4,
            "protection": 5,
            "lands": 36,
        },
        "waves": {
            1: ["commander", "theme"],
            2: ["ramp", "draw", "protection"],
            3: ["removal", "wipes"],
            4: ["lands"],
        },
    },
    "aggro": {
        "ratios": {
            "commander": 1,
            "theme": 30,
            "ramp": 10,
            "draw": 8,
            "removal": 8,
            "wipes": 2,
            "protection": 5,
            "lands": 36,
        },
        "waves": {
            1: ["commander", "theme"],
            2: ["ramp", "draw", "protection"],
            3: ["removal", "wipes"],
            4: ["lands"],
        },
    },
    "control": {
        "ratios": {
            "commander": 1,
            "theme": 12,
            "ramp": 12,
            "draw": 14,
            "removal": 12,
            "wipes": 6,
            "protection": 5,
            "lands": 38,
        },
        "waves": {
            1: ["commander", "theme"],
            2: ["ramp", "draw", "protection"],
            3: ["removal", "wipes"],
            4: ["lands"],
        },
    },
    "combo": {
        "ratios": {
            "commander": 1,
            "theme": 22,
            "ramp": 14,
            "draw": 14,
            "removal": 6,
            "wipes": 2,
            "protection": 5,
            "lands": 36,
        },
        "waves": {
            1: ["commander", "theme"],
            2: ["ramp", "draw", "protection"],
            3: ["removal", "wipes"],
            4: ["lands"],
        },
    },
}

# 60-card versions: halve land counts, trim theme
TEMPLATES_60: dict[str, dict] = {
    style: {
        "ratios": {
            slot: (count // 2 if slot == "lands" else max(1, count // 2))
            for slot, count in tmpl["ratios"].items()
            if slot != "commander"
        },
        "waves": {k: [s for s in v if s != "commander"] for k, v in tmpl["waves"].items()},
    }
    for style, tmpl in TEMPLATES.items()
}


def get_template(style: str, fmt: str) -> dict:
    """Return ratios + waves for the given archetype + format."""
    base = TEMPLATES_60 if fmt == "60card" else TEMPLATES
    return base.get(style, base["midrange"])
