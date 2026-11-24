"""Configure evo's own backend setting in the isolated scratch home."""
import json
import os
from pathlib import Path
from evo.tools import settings

scratch = Path(os.environ['TMPDIR']).resolve()
path = Path(settings.DEFAULT_PATH).resolve()
if not path.is_relative_to(scratch):
    raise RuntimeError('Refusing to modify evo settings outside job scratch')
data = json.loads(path.read_text())
data['plot_backend'] = 'Agg'
settings.write_to_json_file(str(path), data)
(scratch / 'results/evo_settings.json').write_text(json.dumps(data, indent=2))
print('Configured evo Agg backend in isolated scratch home')
