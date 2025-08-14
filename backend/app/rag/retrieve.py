SEED_DOCS = [
    "ICAR advisory: Irrigate wheat at CRI stage; avoid waterlogging.",
    "Raitamitra: Tomato prices peak post-monsoon."
]
def answer(query: str):
    return (f"Best practice for '{query}' found in advisories.",
            ["ICAR Advisory (2023)", "Raitamitra Stats (2024)"])
