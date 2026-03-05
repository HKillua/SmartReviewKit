"""Math formula utilities: OMML-to-LaTeX conversion, Unicode normalisation.

This module provides three core functions used by document loaders to extract
and normalise mathematical content from Office documents and PDFs:

- ``omml_to_latex``  – convert an Office MathML XML element to LaTeX
- ``unicode_math_to_latex`` – map common Unicode math symbols to LaTeX commands
- ``normalize_latex`` – unify delimiter styles ($, $$, \\(, \\[)
"""

from __future__ import annotations

import re
import logging
from typing import Optional

try:
    from lxml import etree
    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Namespace helpers
# ---------------------------------------------------------------------------
OMML_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
_NS = {"m": OMML_NS}

def _tag(local: str) -> str:
    return f"{{{OMML_NS}}}{local}"


# ---------------------------------------------------------------------------
# OMML → LaTeX conversion
# ---------------------------------------------------------------------------
_ACCENT_MAP = {
    "\u0302": r"\hat",
    "\u0303": r"\tilde",
    "\u0304": r"\bar",
    "\u0305": r"\overline",
    "\u0307": r"\dot",
    "\u0308": r"\ddot",
    "\u20D7": r"\vec",
}

def omml_to_latex(element) -> str:  # noqa: C901 – complexity justified for tag handling
    """Convert an OMML ``<m:oMath>`` XML element to a LaTeX string.

    The implementation covers the most common OMML constructs found in
    educational course-ware (fractions, scripts, radicals, delimiters,
    matrices, accents, functions, and plain runs).  Unknown tags are handled
    by recursively extracting their text content so nothing is silently lost.

    Args:
        element: An ``lxml.etree._Element`` whose tag is ``{OMML_NS}oMath``
                 or any child node.

    Returns:
        LaTeX string.  Empty string if *element* contains no useful content.
    """
    if not LXML_AVAILABLE:
        return ""
    if element is None:
        return ""

    tag = etree.QName(element.tag).localname if isinstance(element.tag, str) else ""

    # --- fractions --------------------------------------------------------
    if tag == "f":
        num = _child_latex(element, "num")
        den = _child_latex(element, "den")
        return rf"\frac{{{num}}}{{{den}}}"

    # --- superscript / subscript ------------------------------------------
    if tag == "sSup":
        base = _child_latex(element, "e")
        sup = _child_latex(element, "sup")
        return rf"{{{base}}}^{{{sup}}}"

    if tag == "sSub":
        base = _child_latex(element, "e")
        sub = _child_latex(element, "sub")
        return rf"{{{base}}}_{{{sub}}}"

    if tag == "sSubSup":
        base = _child_latex(element, "e")
        sub = _child_latex(element, "sub")
        sup = _child_latex(element, "sup")
        return rf"{{{base}}}_{{{sub}}}^{{{sup}}}"

    # --- radicals ---------------------------------------------------------
    if tag == "rad":
        deg = _child_latex(element, "deg")
        body = _child_latex(element, "e")
        if deg:
            return rf"\sqrt[{deg}]{{{body}}}"
        return rf"\sqrt{{{body}}}"

    # --- delimiters (parentheses, brackets, braces) -----------------------
    if tag == "d":
        inner_parts = []
        for child in element:
            lt = etree.QName(child.tag).localname if isinstance(child.tag, str) else ""
            if lt == "e":
                inner_parts.append(_children_latex(child))
        beg_chr = _get_val(element, "dPr/begChr", "(")
        end_chr = _get_val(element, "dPr/endChr", ")")
        sep_chr = _get_val(element, "dPr/sepChr", ",")
        content = sep_chr.join(inner_parts)
        return rf"\left{beg_chr}{content}\right{end_chr}"

    # --- n-ary operators (sum, product, integral) -------------------------
    if tag == "nary":
        char = _get_val(element, "naryPr/chr", "∫")
        sub = _child_latex(element, "sub")
        sup = _child_latex(element, "sup")
        body = _child_latex(element, "e")
        op = _NARY_MAP.get(char, char)
        parts = [op]
        if sub:
            parts.append(rf"_{{{sub}}}")
        if sup:
            parts.append(rf"^{{{sup}}}")
        parts.append(rf" {body}")
        return "".join(parts)

    # --- matrices ---------------------------------------------------------
    if tag == "m":
        rows = []
        for mr in element.findall(_tag("mr")):
            cells = []
            for me in mr.findall(_tag("e")):
                cells.append(_children_latex(me))
            rows.append(" & ".join(cells))
        body = r" \\ ".join(rows)
        return rf"\begin{{matrix}}{body}\end{{matrix}}"

    # --- accents ----------------------------------------------------------
    if tag == "acc":
        char = _get_val(element, "accPr/chr", "\u0302")
        body = _child_latex(element, "e")
        cmd = _ACCENT_MAP.get(char, r"\hat")
        return rf"{cmd}{{{body}}}"

    # --- functions (sin, cos, log …) --------------------------------------
    if tag == "func":
        fname = _child_latex(element, "fName")
        body = _child_latex(element, "e")
        return rf"\{fname}{{{body}}}" if fname else body

    # --- bar (overline / underline) ---------------------------------------
    if tag == "bar":
        pos = _get_val(element, "barPr/pos", "top")
        body = _child_latex(element, "e")
        if pos == "bot":
            return rf"\underline{{{body}}}"
        return rf"\overline{{{body}}}"

    # --- runs (leaf text) -------------------------------------------------
    if tag == "r":
        parts = []
        for child in element:
            lt = etree.QName(child.tag).localname if isinstance(child.tag, str) else ""
            if lt == "t":
                parts.append(child.text or "")
        return "".join(parts)

    # --- recursive fallback for container tags (oMath, e, num, den …) -----
    return _children_latex(element)


def _children_latex(parent) -> str:
    """Concatenate LaTeX conversion of all children."""
    parts = []
    for child in parent:
        parts.append(omml_to_latex(child))
    return "".join(parts)


def _child_latex(parent, local_name: str) -> str:
    """Get LaTeX for a specific named child."""
    child = parent.find(_tag(local_name))
    if child is None:
        return ""
    return _children_latex(child)


def _get_val(parent, path: str, default: str = "") -> str:
    """Read the ``m:val`` attribute along *path* (slash-separated local names)."""
    node = parent
    for part in path.split("/"):
        node = node.find(_tag(part))
        if node is None:
            return default
    return node.get(_tag("val"), node.get("val", default))


_NARY_MAP = {
    "∑": r"\sum",
    "∏": r"\prod",
    "∫": r"\int",
    "∮": r"\oint",
    "∬": r"\iint",
    "∭": r"\iiint",
    "⋃": r"\bigcup",
    "⋂": r"\bigcap",
}


# ---------------------------------------------------------------------------
# Unicode math → LaTeX mapping
# ---------------------------------------------------------------------------
_UNICODE_TO_LATEX: dict[str, str] = {
    # Greek lowercase
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
    "ε": r"\epsilon", "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta",
    "ι": r"\iota", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu",
    "ν": r"\nu", "ξ": r"\xi", "π": r"\pi", "ρ": r"\rho",
    "σ": r"\sigma", "τ": r"\tau", "υ": r"\upsilon", "φ": r"\phi",
    "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
    # Greek uppercase
    "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda",
    "Ξ": r"\Xi", "Π": r"\Pi", "Σ": r"\Sigma", "Φ": r"\Phi",
    "Ψ": r"\Psi", "Ω": r"\Omega",
    # Relations
    "≤": r"\leq", "≥": r"\geq", "≠": r"\neq", "≈": r"\approx",
    "≡": r"\equiv", "≪": r"\ll", "≫": r"\gg", "∝": r"\propto",
    "⊂": r"\subset", "⊃": r"\supset", "⊆": r"\subseteq", "⊇": r"\supseteq",
    "∈": r"\in", "∉": r"\notin", "∋": r"\ni",
    # Operators
    "±": r"\pm", "∓": r"\mp", "×": r"\times", "÷": r"\div",
    "·": r"\cdot", "∗": r"\ast", "⊕": r"\oplus", "⊗": r"\otimes",
    "∧": r"\wedge", "∨": r"\vee", "¬": r"\neg",
    # Arrows
    "→": r"\rightarrow", "←": r"\leftarrow", "↔": r"\leftrightarrow",
    "⇒": r"\Rightarrow", "⇐": r"\Leftarrow", "⇔": r"\Leftrightarrow",
    "↑": r"\uparrow", "↓": r"\downarrow",
    # Miscellaneous
    "∞": r"\infty", "∂": r"\partial", "∇": r"\nabla",
    "∅": r"\emptyset", "∀": r"\forall", "∃": r"\exists",
    "ℕ": r"\mathbb{N}", "ℤ": r"\mathbb{Z}", "ℚ": r"\mathbb{Q}",
    "ℝ": r"\mathbb{R}", "ℂ": r"\mathbb{C}",
    "√": r"\sqrt", "∑": r"\sum", "∏": r"\prod",
    "∫": r"\int", "∮": r"\oint",
    # Superscript digits
    "⁰": "^{0}", "¹": "^{1}", "²": "^{2}", "³": "^{3}",
    "⁴": "^{4}", "⁵": "^{5}", "⁶": "^{6}", "⁷": "^{7}",
    "⁸": "^{8}", "⁹": "^{9}", "ⁿ": "^{n}",
    # Subscript digits
    "₀": "_{0}", "₁": "_{1}", "₂": "_{2}", "₃": "_{3}",
    "₄": "_{4}", "₅": "_{5}", "₆": "_{6}", "₇": "_{7}",
    "₈": "_{8}", "₉": "_{9}",
}

_UNICODE_PATTERN = re.compile(
    "|".join(re.escape(k) for k in sorted(_UNICODE_TO_LATEX, key=len, reverse=True))
)


def unicode_math_to_latex(text: str) -> str:
    """Replace common Unicode math symbols with their LaTeX equivalents.

    Only symbols *outside* existing ``$...$`` / ``$$...$$`` delimiters are
    replaced, so already-formatted LaTeX is left untouched.

    >>> unicode_math_to_latex("α + β = γ")
    '\\\\alpha + \\\\beta = \\\\gamma'
    """
    if not text:
        return text

    def _replace(m: re.Match) -> str:
        return _UNICODE_TO_LATEX.get(m.group(0), m.group(0))

    # Split around LaTeX delimiters to avoid double-converting
    parts = re.split(r"(\$\$.*?\$\$|\$.*?\$)", text, flags=re.DOTALL)
    result = []
    for i, part in enumerate(parts):
        if i % 2 == 0:  # outside delimiters
            result.append(_UNICODE_PATTERN.sub(_replace, part))
        else:
            result.append(part)
    return "".join(result)


# ---------------------------------------------------------------------------
# LaTeX delimiter normalisation
# ---------------------------------------------------------------------------

def normalize_latex(text: str) -> str:
    r"""Unify LaTeX delimiter styles to ``$`` (inline) and ``$$`` (display).

    Conversions performed:
    - ``\( ... \)``  →  ``$ ... $``
    - ``\[ ... \]``  →  ``$$ ... $$``

    Existing ``$`` / ``$$`` delimiters are left as-is.
    """
    if not text:
        return text
    text = re.sub(r"\\\(\s*", "$", text)
    text = re.sub(r"\s*\\\)", "$", text)
    text = re.sub(r"\\\[\s*", "$$", text)
    text = re.sub(r"\s*\\\]", "$$", text)
    return text


# ---------------------------------------------------------------------------
# High-level helper used by loaders
# ---------------------------------------------------------------------------

def postprocess_math(text: str) -> str:
    """Apply full math post-processing pipeline to extracted text.

    Steps: Unicode symbol mapping → delimiter normalisation → whitespace cleanup.
    """
    text = unicode_math_to_latex(text)
    text = normalize_latex(text)
    return text
