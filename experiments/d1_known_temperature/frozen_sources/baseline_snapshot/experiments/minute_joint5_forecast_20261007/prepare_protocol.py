"""Build the joint five-channel minute protocol and save an independent report."""
from datetime import datetime, timezone, timedelta
import json
from data import DEFAULT_CONFIG, PROTOCOL, dump, make_protocol
protocol = make_protocol(DEFAULT_CONFIG.copy())
stamp = datetime.now(timezone(timedelta(hours=8)))
report = dict(created_at_beijing=stamp.isoformat(), protocol=str(PROTOCOL), counts=protocol['counts'], training_selection=protocol['training_selection'], reference_contract=protocol['reference_contract'], history_points=protocol['history_points'], future_points=protocol['future_points'], channels=protocol['channels'], units=protocol['units'], source_arrays_modified=False, old_protocol_modified=False)
path = PROTOCOL.parent / ('protocol_report_' + stamp.strftime('%Y%m%d_%H%M%S') + '.json')
dump(path, report)
print('SAVED ' + json.dumps(dict(protocol=str(PROTOCOL), report=str(path))))
