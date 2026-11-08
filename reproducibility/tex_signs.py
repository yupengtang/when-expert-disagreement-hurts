"""Typeset signed numbers in generated LaTeX as math, so interval bounds match the $+x$ point estimates."""
import re

SIGNED = re.compile(r"(?<![\w\\{}^.$-])([+-])(\d+(?:\.\d+)?)(?![\w.])")


def math_signs(text):
    """Wrap +x and -x outside $...$ in math mode; hyphenated names, dates, ranges and lengths are left alone."""
    parts = re.split(r"(\$[^$]*\$)", text)
    return "".join(part if part.startswith("$") else SIGNED.sub(r"$\1\2$", part) for part in parts)
