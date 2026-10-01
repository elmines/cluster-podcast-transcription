import re

_AD_PATTS = [
    r"\.com",
    r"\.edu",
    r"\.org",

    r"Rasmussen University",
    r"Rasmussen", # Too aggressive?

    r"T-Mobile",
    r"T-Satellite",
    r"Lem is",

    r"VerboCare",
    r"Verbo Care",
    r"Verbo", # too aggressive?
    r"Virbo",


    r"MyFICO",
    r"Orderly Meds",
    r"Hollywood Feed",
    r"Found",
    r"Weight Watchers",
    r"Instagram Teen Accounts",
    r"Instagram teen accounts",
    r"Because your playbook ensures your arena is always ready for tip-off",
    r"Offering the products you need all in one place",
    r"Amazon Hub Delivery",
    r"Support After Abortion",
    r"Ellie's Eden",
    r"Azure Well",
    r"Good and the Beautiful's Reading",

    r"Arizona State University",
    r"ASU",
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

    r"Zin", # nicotine patch

    r"Warning, this product contains nicotine\.",
    r"Nicotine is an addictive chemical\.",

    r"Opioid addiction is claiming lives right here in Alachua County",
    r"Visit HopeAlachua\.com",

    r"Ice cold Coca-Cola and football\.",
    r"That's a championship combo",

    r"Find your seat and start now at Rasmussen\.edu",

    r"And with Fin, we've built the number one AI agent for customer service",

    r"Work in Progress is a podcast to help skilled migrants rebuild their careers in a new country",

    # r"I'm Jameeda Jamil and guests on my new podcast, Wrong Turns, share their most mortifying and hilarious disaster stories",
    r"guests on my new podcast",

    # r"Listen now wherever you get your podcasts",
    r"wherever you get your podcasts",

    r"\[MUSIC\]",
    r"\[MUSIC PLAYING\]",
    r"\(slow music\)",
    r"\(upbeat music\)",
    r"\(singing in foreign language\)",
    r"\(soft music\)",
]

_AD_PATTS_2 = [
    r"dot org",
    r"dot edu",
    r"dot com",
    r"\[upbeat music\]",
    r"\[singing in foreign language\]",
    r"\[MUSIC",
    r"\(MUSIC\)",
    r"\(MUSIC PLAYS\)",
    r"\(MUSIC PLAYING\)",
    r"\(MUSIC CONTINUES\)",
    r"\(upbeat drum music\)",
    r"\(upbeat jazz music\)",
    r"\(upbeat rock music\)",
    r"\(sirens wailing\)"
    r"\(upbeat bluegrass music\)",
    r"\(rock music\)",
    r"♪",
    r"a championship combo",
    r"isn't just a road, it's our workplace",
    r"Your choices matter",
    r"rasmussen",
    r"Calshi",
    r"Gainesville",
    r"gainesville",
    r"Burrito", # Capital only for Burrito factory
    r"contains? nicotine",
    r"an addictive chemical",
    r"Warning\.",
    r"Alpha insurance",
    r"(a|A)mazon (h|H)ub (d|D)elivery",
    r"Know a local business that will make a great partner",
    r"A local coffee shop owner, florist, automotive shop, dry cleaner, you name it"
    r"Arizona State",
    r"Original American kitchen",
    r"are sold near you",
    r"Ellie Zedin",
    r"Ellie's Eden",
    r"AirMed",
    r"packed with bold ingredients",
    r"Coligan",
    r"(The|V|B|Bee) Pizza",
    r"(V|v)anta",
    r"((w|W)herever|(a|A)nywhere|(e|E)verywhere) you (get|find) your podcast",
    r"myFICO",
    r"Advantage Business Program",
    r"US Bank",
    r"Kraft Mac and Cheese",
    r"Perfect Bistro",
    r"glp1",
    r"(W|w)eight (W|w)atchers",
    r"get everyone home safely"
    r"Target Zero Initiative",
    r"All American Kitchen",
    r"On Deck",
    r"Celtic Bank",
    r"Simply Money",
    r"beat any price",
    r"Chevy Silverado",
    r"GMC Sierra",
    r"sell more so you save more",
    r"you can get back to what matters most",
    r"epending on certain loan attributes",
    r"(m|M)(x|X) (b|B)usiness (g|G)old (c|C)ard",
    r"uilt for business",
    r"(t|T)opo (c|C)hico",
    r"(A|a)(CAST|cast)",
    r"opioid addiction is claiming lives",

]

_WHITELIST = [
    r"@rasmussen_pole", # Is @ reserved in regex syntax?
    r"Slow music\.",
    r"Zeldin",
    r"zeldin",
]

AD_PATT = re.compile('|'.join(_AD_PATTS + _AD_PATTS_2))

__ALL__ = ["AD_PATT"]
