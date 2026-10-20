from if_release.runtime import expand as _release_expand
import os
from pathlib import Path
root = Path(__file__).resolve().parent
os.chdir(str(root))
os.environ['PATH'] = _release_expand('${COMPILER}/compiler/2024.0/bin;${MSVC}/VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64;') + os.environ['PATH']
code = (root / 'thermal_driver.py').read_text(encoding='utf-8')
exec(compile(code, str(root / 'thermal_driver.py'), 'exec'))
