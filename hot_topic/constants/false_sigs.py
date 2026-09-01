
_CONSUMER = "Consumer Ad"
_PODCAST = "Ad for Another Podcast"
_MUSIC = "Music"

_AD_STRINGS = [
    (_CONSUMER, "Ice cold Coca-Cola and football. That's a championship combo."),
    (_CONSUMER, "Find your seat and start now at Rasmussen.edu."),
    (_CONSUMER, "And with Fin, we've built the number one AI agent for customer service."),
    (_PODCAST, "Work in Progress is a podcast to help skilled migrants rebuild their careers in a new country."),
    (_PODCAST, "I'm Jameeda Jamil and guests on my new podcast, Wrong Turns, share their most mortifying and hilarious disaster stories."),
    (_PODCAST, "Listen now wherever you get your podcasts."),
    (_MUSIC, "[MUSIC]"),
    (_MUSIC, "[MUSIC PLAYING]"),
    (_MUSIC, "(slow music)"),
    (_MUSIC, "(upbeat music)"),
    (_MUSIC, "(singing in foreign language)"),
    (_MUSIC, "(soft music)"),
]
NOISE_PROMPT = "\n".join( f"{k} : {v}" for k,v in _AD_STRINGS)

__all__ = [
        "NOISE_PROMPT"
]
