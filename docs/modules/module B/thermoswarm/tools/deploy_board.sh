#!/usr/bin/env bash
# Install the onboard agent on one SKYRIS Technic over SSH.
#   ./tools/deploy_board.sh T1 technic-1 <server-ip> tr256i     (ssh alias from ~/.ssh/config)
#   ./tools/deploy_board.sh T3 orangepi@<drone-ip> <server-ip> (user@ip)
# args: <id> <ssh_target> <server_ip> [thermal_driver=none] [ros_setup=$HOME/technic_ws/devel/setup.bash]
# The remote user and home directory are taken from the ssh login itself.
set -euo pipefail
ID=$1; TARGET=$2; SERVER=$3; DRIVER=${4:-none}; SETUP_ARG=${5:-}
HERE=$(cd "$(dirname "$0")/.." && pwd)
RUSER=$(ssh "$TARGET" 'whoami'); RHOME=$(ssh "$TARGET" 'echo $HOME')
DEST=$RHOME/thermoswarm
SETUP=${SETUP_ARG:-$RHOME/technic_ws/devel/setup.bash}
echo ">>> $ID: $TARGET (user $RUSER), installing to $DEST"
ssh "$TARGET" "mkdir -p $DEST"
scp -q "$HERE"/board/*.py "$HERE"/board/aruco_map.txt "$HERE"/board/agent.example.json "$HERE"/board/thermoswarm-agent.service "$TARGET:$DEST/"
echo ">>> $ID: writing agent.json (server $SERVER, thermal $DRIVER)"
ssh "$TARGET" "cd $DEST && python3 - '$ID' '$SERVER' '$DRIVER'" <<'PY'
import json, sys
c = json.load(open('agent.example.json'))
c.update(id=sys.argv[1], server=sys.argv[2])
c['thermal']['driver'] = sys.argv[3]
c['thermal'].pop('_driver_for_T1', None)
json.dump(c, open('agent.json', 'w'), indent=2)
print(json.dumps(c, indent=2))
PY
echo ">>> $ID: self-test (8 s, does NOT arm)"
ssh -t "$TARGET" "sudo systemctl stop thermoswarm-agent" || true
ssh "$TARGET" "source $SETUP && cd $DEST && python3 agent.py --check" || echo "!!! self-test failed: read the output above"
if [ "$DRIVER" != "none" ]; then
  echo ">>> $ID: thermal camera probe"
  ssh "$TARGET" "cd $DEST && timeout 30 python3 agent.py --probe-thermal" || echo "!!! thermal probe failed: see README section 4"
fi
echo ">>> $ID: installing systemd service (autostart)"
ssh -t "$TARGET" "sed -e 's#User=orangepi#User=$RUSER#' -e 's#/home/orangepi/thermoswarm#$DEST#' -e 's#/home/orangepi/technic_ws/devel/setup.bash#$SETUP#' $DEST/thermoswarm-agent.service | sudo tee /etc/systemd/system/thermoswarm-agent.service >/dev/null && sudo systemctl daemon-reload && sudo systemctl enable --now thermoswarm-agent && sleep 2 && systemctl --no-pager -n 5 status thermoswarm-agent"
echo ">>> $ID: done. Logs: ssh $TARGET journalctl -u thermoswarm-agent -f"
