"""Extract an unambiguous marked final number without inspecting its gold answer."""
import re
from latent_loop.mathematics import rational,display

NUMBER=r'-?(?:\d[\d,]*(?:\.\d+)?|\.\d+)'
FRACTION=rf'{NUMBER}(?:\s*/\s*{NUMBER})?'


def final_number(text):
    text=text.replace('−','-')
    if '####' in text:tail=text.rsplit('####',1)[1]
    else:
        markers=list(re.finditer(r'final\s+answer\s*[:=]',text,re.I))
        if markers:tail=text[markers[-1].end():]
        else:
            boxed=list(re.finditer(r'\\boxed\s*\{',text))
            if not boxed:return None
            tail=text[boxed[-1].end():]
    def latex_fraction(match):
        numerator=rational(match[2].replace(',',''));denominator=rational(match[3].replace(',',''))
        if not denominator:return 'undefined'
        value=numerator/denominator
        if match[1]=='-':value=-value
        return '('+display(value)+')'
    tail=re.sub(rf'(-?)\s*\\(?:dfrac|tfrac|frac)\s*\{{\s*({NUMBER})\s*\}}\s*\{{\s*({NUMBER})\s*\}}',latex_fraction,tail)
    tail=re.sub(rf'-\s*\(\s*({FRACTION})\s*\)',lambda m:'('+display(-rational(m[1].replace(',','').replace(' ','')))+')',tail)
    matches=re.findall(FRACTION,tail)
    # Multiple distinct quantities after a final marker are ambiguous. Reject
    # instead of choosing whichever number happens to match the ground truth.
    try:values={rational(m.replace(',','').replace(' ','')) for m in matches}
    except ValueError:return None
    return display(values.pop()) if len(values)==1 else None
