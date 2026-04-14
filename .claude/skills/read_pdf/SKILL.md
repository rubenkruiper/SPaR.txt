---
name: read_pdf
description: Extract text from a PDF file and write it to /tmp/spar_input.txt, ready for use by the /spar skill or any other text-processing step. Pass the path to the PDF as the argument, e.g. `/read_pdf data/my_doc.pdf`.
argument-hint: path/to/document.pdf
---

Extract text from the PDF at `$ARGUMENTS` and write it to `/tmp/spar_input.txt`.

## Step 1 — Verify the file exists

```bash
ls "$ARGUMENTS"
```

If the file does not exist, stop and tell the user. Do not proceed.

## Step 2 — Extract text

```bash
poetry run python -c "
import pdfplumber
from pathlib import Path

pdf_path = Path('$ARGUMENTS')
pages = []
with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            pages.append(text.strip())

full_text = '\n\n'.join(pages)
Path('/tmp/spar_input.txt').write_text(full_text)
print(f'Pages: {len(pdf.pages)}')
print(f'Pages with text: {len(pages)}')
print(f'Characters extracted: {len(full_text)}')
print()
print('First 500 characters:')
print(full_text[:500])
"
```

## Step 3 — Report and hand off

Tell the user how many pages were extracted and confirm the text is ready at `/tmp/spar_input.txt`. If any pages returned no text, note that the PDF may be partially scanned — those pages are silently skipped.

The extracted text is now ready to pass to `/spar`.
