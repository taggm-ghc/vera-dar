# General English word list (SELFQ-RELEVANCE-1108 fix)

`english_words.txt`: 370,105 lower-case alphabetic English words, one per line. `vera/spell_clarify.py` treats a
question word in this list as correctly spelled, so it never triggers the clarification round (live regression on
b12c881: "lava" paused as a typo). It is never used to suggest corrections and never used as evidence.

## Source

- dwyl/english-words, file `words_alpha.txt`, fetched on 2026-11-08T10:46Z from
  `https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt`, after a separate robots check of
  that host (`https://raw.githubusercontent.com/robots.txt`: HTTP 404, no robots file, so no `*`, ClaudeBot,
  Claude-User or anthropic-ai group restricts the path); identical to a first fetch at 10:44Z. sha256
  `3ed0c94610d8bcf7c11bbb49c56aa49c7234d32b66824df91f554169e572da48`.
- Licence: The Unlicense (public-domain dedication, no attribution or notice required), fetched the same day from
  `https://raw.githubusercontent.com/dwyl/english-words/master/LICENSE.md`, sha256
  `b5065838cbac452dfc855ba6e6e031481ad2c68406f70d21ead9321374653e6c`; reproduced exactly in
  `LICENSE-dwyl-english-words.md`.
- Why: an open, public-domain list with nothing the repo's secret/host scan blocks; large enough that ordinary and
  scientific words (lava, quarks, photosynthesis, tectonics) and common proper nouns (London, Paris, Einstein) count as
  correctly spelled. The previous SCOWL-based list was replaced because its notice file did not match the data release
  and required a third-party contact line verbatim.
- Built with `python3 -I scripts/build_english_wordlist.py words_alpha.txt config/wordlists/english_words.txt`
  (lower-case alphabetic lines only, de-duplicated, sorted; no word added or removed by hand); output sha256
  `1198a3786bed510b4f8127f73a74c38596088a220d318d962e689dc6f0792599`.

## Known effect

The list is broad and includes rare words, so a typo that happens to be a rare real word (for example "vare") is not
paused and is searched as typed. Typos that are not English words ("vrea", "anwser") still pause when a near known word
exists within the calibrated bounds.
