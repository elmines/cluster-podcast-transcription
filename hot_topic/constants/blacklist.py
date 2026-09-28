import re

_AD_PATTS = [
    r"\.com",
    r"\.edu",
    r"Rasmussen University",
    r"Arizona State University",
    r"US Bank Business Essential",
    r"Alpha Insurance",
    r"Hartford",
    r"OnDeck",
    r"American Airlines Advantage Business Program",
    r"Davis Gainesville Chevrolet GMC",
    r"Grainger",
    r"Kalshi", # This may be too strong--a lot of podcasts probably talk about Kalshi
    r"Vanta ",
    r"V Pizza",
    r"Coke Florida",
    r"Burrito Factory",
    r"American Express Business Gold Card",
    r"Spurrier's Grit-Iron Grill in Gainesville",
    r"Original American Kitchen",
    r"ACAS powers",
    r"Hey Gainesville",
    r"reminds you that April is Child Abuse Prevention Month",

    r"This isn't just a road\.",
    r"It's our workplace\.",
    r"Slow down\.",
    r"Keep your distance\.",
    r"Stay focused\.",
    r"Your choices matter\.",
    r"Let's get everyone home safely\.",

    r"Warning, this product contains nicotine\.",
    r"Nicotine is an addictive chemical\.",

    r"Opioid addiction is claiming lives right here in Alachua County",
    r"Visit HopeAlachua\.com",

    r"Ice cold Coca-Cola and football\.",
    r"That's a championship combo",

    r"Find your seat and start now at Rasmussen\.edu",

    r"And with Fin, we've built the number one AI agent for customer service",

    r"Work in Progress is a podcast to help skilled migrants rebuild their careers in a new country",

    r"I'm Jameeda Jamil and guests on my new podcast, Wrong Turns, share their most mortifying and hilarious disaster stories",

    r"Listen now wherever you get your podcasts",

    r"\[MUSIC\]",
    r"\[MUSIC PLAYING\]",
    r"\(slow music\)",
    r"\(upbeat music\)",
    r"\(singing in foreign language\)",
    r"\(soft music\)",
]

AD_PATT = re.compile('|'.join(_AD_PATTS))

__ALL__ = ["AD_PATT"]