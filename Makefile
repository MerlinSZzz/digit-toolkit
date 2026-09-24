# DIGIT tactile sensor toolkit -- simple commands for a new visitor.
#
# Everything is a thin wrapper over `python -m digit ...` so the same features
# are available from the shell.  Run `make` or `make help` for the list.
#
# The `digit` conda env is created by `make setup`; override PY if yours lives
# elsewhere:  make check PY=/path/to/python

SHELL := /bin/bash
CONDA ?= $(HOME)/miniconda3/bin/conda
ENV_PY := $(HOME)/miniconda3/envs/digit/bin/python
PY ?= $(shell if [ -x "$(ENV_PY)" ]; then echo "$(ENV_PY)"; else echo python3; fi)

# common parameters (override on the command line, e.g. `make led LEVEL=8`)
SERIAL ?=
NODE ?=
RES ?= 640x480
FPS ?= 30
LED ?=
LEVEL ?= 15
R ?= 15
G ?= 15
B ?= 15
SECONDS ?= 5
FRAMES ?= 90
MODE ?= dashboard
THRESHOLD ?= 10
MIN_AREA ?= 40
NAME ?=
SESSION ?=
OUT ?= .logs/view_$(MODE).png
PROCESS ?= dashboard
DATASET ?= datasets/synthetic.npz
MODEL ?= models/digit_classifier.pkl
LABELS ?= circle square
ROOT ?= sessions
CLASSES ?= position
N ?= 10
METHOD ?= svm
REFERENCE ?=

# optional flags forwarded to the CLI
SERIAL_ARG := $(if $(SERIAL),--serial $(SERIAL),)
NODE_ARG := $(if $(NODE),--node $(NODE),)
LED_ARG := $(if $(LED),--led $(LED),)
NAME_ARG := $(if $(NAME),--name $(NAME),)
SESSION_ARG := $(if $(SESSION),$(SESSION),)
FAKE_ARG := $(if $(FAKE),--fake,)
FAKE_REF_ARG := $(if $(FAKE_REFERENCE),--fake-reference $(FAKE_REFERENCE),)
REF_ARG := $(if $(REFERENCE),--reference $(REFERENCE),)

.PHONY: help setup check info list devices modes snapshot reference led led-rgb fps \
        resolution view view-diff view-contact view-depth view-normal dashboard \
        save record sessions play monitor monitor-json eval synth train predict \
        collect benchmark reset install-udev udev-check test selftest demo

help:
	@echo "DIGIT tactile sensor toolkit -- make targets"
	@echo ""
	@echo "  setup            create the 'digit' conda env and install deps (one time)"
	@echo "  check            sensor found? frames arriving? measured fps?"
	@echo "  info / list / modes   device, serial and supported V4L2 modes"
	@echo ""
	@echo "  view             live dashboard (MODE=camera|diff|contact|depth|normal)"
	@echo "  save             render MODE headless and write OUT (default .logs/view_\$$MODE.png)"
	@echo "  view-diff view-contact view-depth view-normal   quick single views"
	@echo "  monitor          headless touch/slip monitor (SECONDS=10)"
	@echo "  monitor-json     same, one JSON object per frame"
	@echo ""
	@echo "  snapshot         save one frame to snapshots/"
	@echo "  reference        capture a no-contact reference (touch nothing!)"
	@echo "  led              set LED level: make led LEVEL=8  (also led-rgb R= G= B=)"
	@echo "  fps              set/verify frame rate: make fps FPS=15"
	@echo "  resolution       set/verify resolution: make resolution RES=320x240"
	@echo ""
	@echo "  record           record a session: make record SECONDS=10"
	@echo "  sessions         list recorded sessions"
	@echo "  play             replay: make play SESSION=sessions/... [PROCESS=dashboard]"
	@echo ""
	@echo "  eval             synthetic recognition benchmark (touch/localise/classify)"
	@echo "  demo             sensor-free: render every processing view to docs/images/"
	@echo "  synth            build a synthetic feature dataset"
	@echo "  train            train/evaluate a classifier on a dataset"
	@echo "  predict          live prediction with a trained model"
	@echo "  collect          interactively collect real labelled samples"
	@echo ""
	@echo "  benchmark        measure fps at every supported mode"
	@echo "  test             run the sensor-free unit tests (pytest)"
	@echo "  selftest         sensor-free end-to-end smoke test"
	@echo "  reset            soft USB reset (like a replug)"
	@echo "  install-udev     install 99-digit.rules so reset works without sudo"
	@echo ""
	@echo "Common overrides: SERIAL= NODE= RES= FPS= LED= LEVEL= SECONDS= FRAMES= MODE="
	@echo "                  THRESHOLD= OUT= DATASET= MODEL= SESSION= PY="

setup:
	@$(CONDA) env list | grep -qE '^digit[[:space:]]' || $(CONDA) create -y -n digit python=3.10
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	@echo "digit env ready. Run: make check"

check:
	$(PY) -m digit check $(SERIAL_ARG) $(NODE_ARG) $(LED_ARG) --resolution "$(RES)" --fps $(FPS)

info:
	$(PY) -m digit info $(SERIAL_ARG) $(NODE_ARG)

list devices:
	$(PY) -m digit list

modes:
	$(PY) -m digit modes $(NODE_ARG)

benchmark:
	$(PY) -m digit benchmark $(SERIAL_ARG) $(NODE_ARG) --frames $(FRAMES) --out .logs/benchmark.json

snapshot:
	$(PY) -m digit snapshot $(SERIAL_ARG) $(NODE_ARG) $(LED_ARG) --resolution "$(RES)"

reference:
	$(PY) -m digit reference $(SERIAL_ARG) $(NODE_ARG) $(LED_ARG) --resolution "$(RES)" --out reference.png

led:
	$(PY) -m digit led $(SERIAL_ARG) $(NODE_ARG) $(LEVEL) $(LED_ARG) --resolution "$(RES)"
led-rgb:
	$(PY) -m digit led $(SERIAL_ARG) $(NODE_ARG) --rgb $(R) $(G) $(B) $(LED_ARG) --resolution "$(RES)"

fps:
	$(PY) -m digit fps $(SERIAL_ARG) $(NODE_ARG) $(FPS) --resolution "$(RES)"

resolution:
	$(PY) -m digit resolution $(SERIAL_ARG) $(NODE_ARG) "$(RES)" --fps $(FPS)

# interactive live view (opens a window when DISPLAY is set)
view:
	$(PY) -m digit view $(SERIAL_ARG) $(NODE_ARG) $(LED_ARG) --resolution "$(RES)" --fps $(FPS) \
		--mode $(MODE) --threshold $(THRESHOLD) --min-area $(MIN_AREA) $(FAKE_ARG) $(FAKE_REF_ARG)

# render a view headless and save it (used for screenshots / reports)
save:
	$(PY) -m digit view $(SERIAL_ARG) $(NODE_ARG) $(LED_ARG) --resolution "$(RES)" --fps $(FPS) \
		--mode $(MODE) --out "$(OUT)" --frames $(FRAMES) --no-show --threshold $(THRESHOLD) --min-area $(MIN_AREA) \
		$(FAKE_ARG) $(FAKE_REF_ARG)

# sensor-free demo: render every processing view from a synthetic animated contact
demo:
	$(PY) -m digit demo $(REF_ARG) --out "$(OUT)" --frames $(FRAMES)

dashboard:
	$(MAKE) save MODE=dashboard OUT=.logs/view_dashboard.png FAKE=$(FAKE) FAKE_REFERENCE=$(FAKE_REFERENCE)
view-diff:
	$(MAKE) save MODE=diff OUT=.logs/view_diff.png FAKE=$(FAKE) FAKE_REFERENCE=$(FAKE_REFERENCE)
view-contact:
	$(MAKE) save MODE=contact OUT=.logs/view_contact.png FAKE=$(FAKE) FAKE_REFERENCE=$(FAKE_REFERENCE)
view-depth:
	$(MAKE) save MODE=depth OUT=.logs/view_depth.png FAKE=$(FAKE) FAKE_REFERENCE=$(FAKE_REFERENCE)
view-normal:
	$(MAKE) save MODE=normal OUT=.logs/view_normal.png FAKE=$(FAKE) FAKE_REFERENCE=$(FAKE_REFERENCE)

record:
	$(PY) -m digit record $(SERIAL_ARG) $(NODE_ARG) $(LED_ARG) --resolution "$(RES)" --fps $(FPS) \
		--seconds $(SECONDS) $(NAME_ARG) --root $(ROOT) $(FAKE_ARG) $(FAKE_REF_ARG)

sessions:
	$(PY) -m digit sessions --root $(ROOT)

play:
	$(PY) -m digit play "$(SESSION_ARG)" --root $(ROOT) --process $(PROCESS) \
		$(if $(OUT),--out $(OUT),) --fps $(FPS)

monitor:
	$(PY) -m digit monitor $(SERIAL_ARG) $(NODE_ARG) $(LED_ARG) --resolution "$(RES)" --fps $(FPS) \
		--seconds $(SECONDS) --threshold $(THRESHOLD) --min-area $(MIN_AREA) $(FAKE_ARG) $(FAKE_REF_ARG)

monitor-json:
	$(PY) -m digit monitor $(SERIAL_ARG) $(NODE_ARG) $(LED_ARG) --resolution "$(RES)" --fps $(FPS) \
		--seconds $(SECONDS) --threshold $(THRESHOLD) --min-area $(MIN_AREA) --json $(FAKE_ARG) $(FAKE_REF_ARG)

eval:
	$(PY) -m digit eval $(SERIAL_ARG) $(NODE_ARG) --threshold $(THRESHOLD) --min-area $(MIN_AREA) \
		$(if $(REFERENCE),--reference $(REFERENCE),) --out .logs/eval.json

synth:
	$(PY) -m digit synth --synthetic-reference --classes $(CLASSES) --n $(N) --out $(DATASET)

train:
	$(PY) -m digit train $(DATASET) --method $(METHOD) --model $(MODEL) --out .logs/train.json

predict:
	$(PY) -m digit predict $(SERIAL_ARG) $(NODE_ARG) $(MODEL) --resolution "$(RES)" --frames $(FRAMES) $(FAKE_ARG) $(FAKE_REF_ARG)

collect:
	$(PY) -m digit collect $(SERIAL_ARG) $(NODE_ARG) $(LABELS) --dataset $(DATASET) --frames $(FRAMES)

reset:
	$(PY) -m digit reset $(SERIAL_ARG)

install-udev:
	sudo cp 99-digit.rules /etc/udev/rules.d/99-digit.rules
	sudo udevadm control --reload
	sudo udevadm trigger
	@echo "Now unplug and replug the DIGIT sensor once."

udev-check:
	@if [ -f /etc/udev/rules.d/99-digit.rules ]; then echo "99-digit.rules installed"; else echo "99-digit.rules NOT installed (run: make install-udev)"; fi
	@ls -l /dev/bus/usb/*/* 2>/dev/null | grep -i . | tail -n 3 || true

test:
	$(PY) -m pytest -q tests

selftest:
	$(PY) -m digit selftest
