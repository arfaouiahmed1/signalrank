#!/usr/bin/env python3
"""
Extract diverse CV samples from the two Kaggle resume datasets + keep Ahmed's CV.
- snehaanbhawal/resume-dataset: Resume/Resume.csv (2484, 25 categories)
- saugataroyarghya/resume-dataset: resume_data.csv (with skills, but we use snehaanbhawal as cleaner)
Saves to data/sample/cv_<category>.txt and data/sample/cv_ahmed.txt (existing).
Usage: python scripts/extract_resume_cvs.py
"""
import pandas as pd
from pathlib import Path
import re

OUT = Path("data/sample")
SRC1 = Path(r"C:\Users\ahmed\.cache\kagglehub\datasets\snehaanbhawal\resume-dataset\versions\1\Resume\Resume.csv")
SRC2 = Path(r"C:\Users\ahmed\.cache\kagglehub\datasets\saugataroyarghya\resume-dataset\versions\1\resume_data.csv")

OUT.mkdir(parents=True, exist_ok=True)

# Keep Ahmed's CV as is (already exists)
# Extract 3 diverse CVs from snehaanbhawal
if SRC1.exists():
    df = pd.read_csv(SRC1)
    # Pick 1 representative per category: longest Resume_str as most detailed
    categories = ["INFORMATION-TECHNOLOGY", "HR", "SALES", "TEACHER", "ENGINEERING"]
    for cat in categories:
        sub = df[df["Category"] == cat]
        if sub.empty:
            continue
        # pick longest
        idx = sub["Resume_str"].str.len().idxmax()
        row = sub.loc[idx]
        text = row["Resume_str"]
        # clean: remove excessive whitespace, html artifacts
        text = re.sub(r"\s+", " ", text).strip()
        # truncate to 4000 chars for eval
        text = text[:4000]
        out_path = OUT / f"cv_{cat.lower().replace('-','_')}.txt"
        out_path.write_text(f"# CV — {cat} (from snehaanbhawal/resume-dataset, ID {row['ID']})\n\n" + text, encoding="utf-8")
        print(f"Wrote {out_path} ({len(text)} chars)")

# Also extract 2 from saugataroyarghya for variety (if needed)
if SRC2.exists():
    try:
        df2 = pd.read_csv(SRC2, nrows=1000)  # sample first 1k due to size
        # Use career_objective + skills as CV text
        # Pick 2 random diverse: one Big Data, one HR-like
        # Filter where skills contains Python vs HR
        py = df2[df2["skills"].astype(str).str.contains("Python", case=False, na=False)]
        hr = df2[df2["skills"].astype(str).str.contains("HR|Recruitment", case=False, na=False)]
        for name, sub in [("cv_bigdata", py), ("cv_hr2", hr)]:
            if sub.empty:
                continue
            row = sub.iloc[0]
            text = f"Career Objective: {row.get('career_objective','')}\nSkills: {row.get('skills','')}\nExperience: {row.get('positions','')} at {row.get('professional_company_names','')} ({row.get('related_skils_in_job','')})\nResponsibilities: {row.get('responsibilities','')}\nMatched Score: {row.get('matched_score','')}"
            text = re.sub(r"\s+", " ", text).strip()[:4000]
            out_path = OUT / f"{name}.txt"
            if not out_path.exists():  # don't overwrite if already from snehaanbhawal
                out_path.write_text(f"# CV — {name} (from saugataroyarghya/resume-dataset)\n\n" + text, encoding="utf-8")
                print(f"Wrote {out_path}")
    except Exception as e:
        print(f"SRC2 extract failed: {e}")

# List all
print("\nAll CVs in data/sample:")
for p in sorted(OUT.glob("cv_*.txt")):
    print(f" - {p.name} ({p.stat().st_size} bytes)")

# Also create a manifest
manifest = [{"file": p.name, "path": str(p), "size": p.stat().st_size} for p in sorted(OUT.glob("cv_*.txt"))]
import json
(Path("data/sample/manifest.json")).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("Wrote manifest.json")
