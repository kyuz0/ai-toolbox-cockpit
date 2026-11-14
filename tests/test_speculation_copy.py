"""Product terminology gate: expose standard speculative-decoding names."""
from pathlib import Path
import re


def test_no_obscure_speculation_names_in_product_copy():
    root = Path(__file__).resolve().parents[1]
    paths = [root / 'README.md', *(root / 'docs').rglob('*.md')]
    paths += [p for p in (root / 'ai_toolbox_cockpit').rglob('*') if p.suffix in {'.py', '.json'}]
    forbidden = re.compile(r'target[ -]only|bundled draft', re.I)
    violations = [str(p.relative_to(root)) for p in paths if forbidden.search(p.read_text())]
    assert not violations, violations
