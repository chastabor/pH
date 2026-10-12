---
name: categorize
version: 1.0.0
description: "Hand to a child to sort a text, an image or a file into one of the categories you name: it returns the category and why, in a checked shape. Name the categories and the thing to sort in the child's prompt."
argument-hint: "<the categories, and the text or file to sort>"
model: classify
output: {"type": "object", "properties": {"category": {"type": "string"}, "reason": {"type": "string"}}, "required": ["category", "reason"], "additionalProperties": false}
---

# Sorting into the categories you were given

Your request names the categories and the thing to sort. Pick exactly one category,
and spell it exactly as the request spells it: whoever asked will compare it to their
own list.

- **A text** is in the request, or in a file it names: read the file.
- **An image, a PDF, audio or video** is in a file the request names: attach it to
  see it, with `attach` or, in a Python cell, `await tools.attach(path=...)`.

When nothing fits well, choose the closest category and say in the reason what did
not fit. Do not invent a category the request did not name.

Return the category and one sentence saying why, as the value you owe.
