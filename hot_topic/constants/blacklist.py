import re

_CASE_INSENS_PATTS = [
    r"rasmussen", # Too aggressive?
    r"T-Mobile",
    r"T-Satellite",
    r"Chevy Silverado",
    r"GMC Sierra",
    r"Burrito Factory",

    r"(Original|All) American Kitchen",
    r"Arizona State",

    r"V(e|i)rbo", # too aggressive?
    r"MyFICO",
    r"Orderly Meds",
    r"Hollywood Feed",
    r"Amazon Hub Delivery",
    r"Support After Abortion",

    r"Ellie's Eden",
    r"Ellie Zedin",

    r"US Bank Business Essential",
    r"Alpha Insurance",
    r"Hartford",
    r"OnDeck",
    r"American Airlines Advantage Business Program",
    r"Davis Gainesville Chevrolet GMC",
    r"Grainger",
    r"Kalshi", # This may be too strong--a lot of podcasts probably talk about Kalshi
    r"V Pizza",
    r"Coke Florida",
    r"American Express Business Gold Card",
    r"Spurrier's Grit-Iron Grill",
    r"ACAS powers",
    r"reminds you that April is Child Abuse Prevention Month",
    r"Instagram Teen Accounts",
    r"Because your playbook ensures your arena is always ready for tip-off",
    r"Offering the products you need all in one place",
    r"Azure Well",
    r"Warning, this product contains nicotine\.",
    r"Nicotine is an addictive chemical\.",

    r"Ice(-| )cold Coca(-| )Cola and football",
    r"championship combo",

    r"And with Fin, we've built the number one AI agent for customer service",
    r"Work in Progress is a podcast to help skilled migrants rebuild their careers in a new country",
    r"guests on my new podcast",

    r"alachua"
    r"gainesville",

    r"Know a local business that will make a great partner",
    r"A local coffee shop owner, florist, automotive shop, dry cleaner, you name it"

    r"isn't just a road, it's our workplace",
    r"Your choices matter",

    # music (starting with a bracket or paren)
    r"[\[(](music|upbeat|singing|rock|sirens wailing)",
    r"♪",
    r"are sold near you",
    r"MX Business Gold Card",
    r"opioid addiction is claiming lives",
    r"Topo Chico",
    r"ACAST",
    r"packed with bold ingredients",
    r"Coligan",
    r"Vanta",
    r"(wherever|anywhere|everywhere) you (get|find) your podcast",
    r"Advantage Business Program",
    r"US Bank",
    r"Kraft Mac and Cheese",
    r"Perfect Bistro",
    r"get everyone home safely"
    r"Weight Watchers",
    r"Target Zero Initiative",
    r"Celtic Bank",
    r"Simply Money",
    r"beat any price",
    r"sell more so you save more",
    r"you can get back to what matters most",
    r"epending on certain loan attributes",
    r"uilt for business",
    r"(V|B|Bee) Pizza",
    r"This isn't just a road\.",
    r"It's our workplace\.",
    r"Slow down\.",
    r"Keep your distance\.",
    r"Stay focused\.",
    r"Your choices matter\.",
    r"Let's get everyone home safely\.",
    r"(C|k)alshi",
    r"AirMed",
]

_CASE_AD_PATTS = [
    r"Vanta",
    r"ASU",
    r"On Deck",
    r"Found",
    r"Good and the Beautiful's Reading",
    # Mis-transcription of "V Pizza"
    r"The Pizza",
    r"Alpha", # For alpha insurance
    r"Lem is",

    r"Burrito", # capitalized--probably ad for burrito factory

    r"contains? nicotine",
    r"an addictive chemical",

    r"Zin", # nicotine patch
    r"Warning\.",
]


_WHITELIST = [
    r"@rasmussen_pole", # Is @ reserved in regex syntax?
    r"Slow music\.",
    r"Zeldin",
    r"zeldin",
]

AD_PATT = re.compile("(?i:" + "|".join(_CASE_INSENS_PATTS) + ")" + "|" + "|".join(_CASE_AD_PATTS))

__ALL__ = ["AD_PATT"]
