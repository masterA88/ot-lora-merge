"""Apply Hilmi's authorship/style rules to the LaTeX sources (idempotent).

Rules (all projects):
  - author name = "Hilmi" only (no full name)
  - no email; use website link https://master-hilmi.vercel.app/
  - no em-dashes / '--' in PROSE: spaced ' -- ' -> comma; value ranges -> 'to';
    table placeholder ' --- ' -> 'n/a'.
  Preserved on purpose: hyphenated names (OT-TIES, ViT-B/32, low-rank, rank-$r$),
  proper-name en-dashes with no surrounding spaces (Gromov--Wasserstein),
  cross-reference ranges (}--\\ref), and command flags (--index-url).
"""
import re, sys

WEBSITE = r"\url{https://master-hilmi.vercel.app/}"

def fix_author(s):
    s = s.replace("Hilmi Nur Ardian\\\\\n\\texttt{master.ardian@gmail.com}",
                  "Hilmi\\\\\n" + WEBSITE)
    # safety net if the line break differs
    s = s.replace("\\texttt{master.ardian@gmail.com}", WEBSITE)
    s = s.replace("Hilmi Nur Ardian", "Hilmi")
    return s

def fix_dashes(s):
    # 1) value ranges wrapped in math: $a$--$b$  ->  $a$ to $b$
    s = re.sub(r"\$([^$]*)\$--\$([^$]*)\$", r"$\1$ to $\2$", s)
    # 2) table placeholder cells:  ' --- '  ->  ' n/a '   (note the surrounding spaces;
    #    the long comment rule '----------' has no surrounding single spaces and is untouched)
    s = s.replace(" --- ", " n/a ")
    # 3) bare numeric ranges: 264--359, 2--3, 5--6  ->  '264 to 359' etc.
    s = re.sub(r"(?<=\d)--(?=\d)", " to ", s)
    # 4) prose em-dash style ' -- '  ->  ', '
    s = s.replace(" -- ", ", ")
    return s

for path in sys.argv[1:]:
    txt = open(path, encoding="utf-8").read()
    before = txt
    txt = fix_author(txt)
    txt = fix_dashes(txt)
    open(path, "w", encoding="utf-8").write(txt)
    # report
    rem = txt.count(" -- ") + len(re.findall(r"(?<=\d)--(?=\d)", txt)) + txt.count(" --- ")
    print(f"{path}: email_present={'master.ardian' in txt} fullname_present={'Nur Ardian' in txt} "
          f"prose_dash_remaining={rem} changed={before!=txt}")
