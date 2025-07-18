#!/bin/bash

cd `dirname $0`
cd ..

./scripts/stop.sh

VISION_ARGS=()
BRAIN_ARGS=()
GAME_ARGS=()
for arg in "$@"; do
  case "$arg" in
    robot:=*|rerun:=*)
      VISION_ARGS+=("$arg")
      BRAIN_ARGS+=("$arg")
      ;;
    tree:=*|role:=*|attack:=*|pos:=*)
      BRAIN_ARGS+=("$arg")
      ;;
    port:=*)
      GAME_ARGS+=("$arg")
      ;;
    *)
      echo "⚠️ Unrecognized or unscoped argument: $arg" >&2
      ;;
  esac
done

source ./install/setup.bash
export FASTRTPS_DEFAULT_PROFILES_FILE=./configs/fastdds.xml

nohup ros2 launch vision launch.py "${VISION_ARGS[@]}" > vision.log 2>&1 &
nohup ros2 launch brain launch.py "${BRAIN_ARGS[@]}" > brain.log 2>&1 &
nohup ros2 launch game_controller launch.py "${GAME_ARGS[@]}" > game_controller.log 2>&1 &
