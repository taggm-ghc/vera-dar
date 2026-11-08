"""SELFQ-RELEVANCE-1108 fix: build config/wordlists/english_words.txt from dwyl/english-words words_alpha.txt (The
Unlicense; provenance in config/wordlists/README.md). Keeps lower-case alphabetic lines only, de-duplicated and sorted;
no word is added or removed by hand.
Usage: python3 -I scripts/build_english_wordlist.py words_alpha.txt config/wordlists/english_words.txt"""
import re
import sys

src, out = sys.argv[1:3]
words = sorted({w.strip() for w in open(src, encoding="utf-8") if re.fullmatch(r"[a-z]+", w.strip())})
open(out, "w", encoding="utf-8").write("\n".join(words) + "\n")
print(len(words))
